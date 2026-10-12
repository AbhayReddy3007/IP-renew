"""
patents.py — Patent-level data transforms:
  - Restricting patent rows to the patent_master-approved shortlist
  - Deduplication of patents recorded under multiple number formats
  - Filling blank Controlling Patent Expiry Year from patent_discovery / Filing Date
  - Building the d2 table (d1 + patent_discovery enrichment)
"""

import re
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery

from utils import (
    _find_col, _normalize_patent_number, _is_blank, _filing_year,
    _jurisdiction_token, _ip_dimension_score,
)
from bq_io import _bq_client, _load_table_from_bq, _upload_to_bigquery


# ── patent_master approval filter ─────────────────────────────────────────────
# Mirrors IP_refactored.chunking.patent_filter.get_approved_patent_numbers: a
# patent is only "approved" when patent_status = 'PDF Downloaded' and
# confidence > 0.3. Everything downstream (dedup, expiry sourcing, Years to
# Entry, d2) should only ever see this shortlisted set.

_PATENT_MASTER_QUERY_BY_MOLECULES = """
SELECT molecule_name, patent_number
FROM `{table}`
WHERE lower(molecule_name) IN UNNEST(@molecule_names)
AND patent_status = 'PDF Downloaded'
AND confidence > 0.3
"""

_PATENT_MASTER_QUERY_ALL = """
SELECT molecule_name, patent_number
FROM `{table}`
WHERE patent_status = 'PDF Downloaded'
AND confidence > 0.3
"""


def _load_approved_patent_numbers(project: str, dataset: str, table: str, drug_names,
                                   credentials_path: Optional[str]) -> Dict[str, set]:
    """Query patent_master for approved patent numbers (patent_status =
    'PDF Downloaded' and confidence > 0.3), grouped by normalised drug/
    molecule name (lower-cased, stripped — matching `lower(molecule_name) =`).

    One query covers every requested drug rather than one query per drug.
    Returns {} (and prints a warning) on query failure.
    """
    fq_table = f"{project}.{dataset}.{table}"
    drug_names = sorted({str(d).strip().lower() for d in drug_names if str(d).strip()})

    client = _bq_client(project, credentials_path)
    try:
        if drug_names:
            query = _PATENT_MASTER_QUERY_BY_MOLECULES.format(table=fq_table)
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("molecule_names", "STRING", drug_names)]
            )
            print(f"\n  [patent_master] Querying {fq_table} for {len(drug_names)} drug(s)' approved patents...")
            rows = client.query(query, job_config=job_config).result()
        else:
            query = _PATENT_MASTER_QUERY_ALL.format(table=fq_table)
            print(f"\n  [patent_master] Querying {fq_table} for ALL molecules' approved patents...")
            rows = client.query(query).result()
    except Exception as e:
        print(f"  [WARN] [patent_master] Query failed against {fq_table}: {e} — skipping approval filter.")
        return {}

    approved: Dict[str, set] = {}
    for row in rows:
        if not row.molecule_name or not row.patent_number:
            continue
        pn_norm = _normalize_patent_number(row.patent_number)
        if not pn_norm:
            continue
        approved.setdefault(str(row.molecule_name).strip().lower(), set()).add(pn_norm)

    print(f"  [patent_master] {sum(len(v) for v in approved.values())} approved patent number(s) "
          f"across {len(approved)} drug(s).")
    return approved


def _filter_to_approved_patents(final_df: pd.DataFrame, project: str, dataset: str, table: str,
                                 credentials_path: Optional[str]) -> pd.DataFrame:
    """Restrict final_df to only the patent_master-approved shortlist
    (patent_status = 'PDF Downloaded' and confidence > 0.3) for that drug,
    before any patent-level computation (dedup, expiry sourcing, Years to
    Entry, d2) runs.

    Forecasted rows (Type == 'Forecasted') and rows with a blank Patent
    Number are left untouched — patent_master only tracks real, discovered
    patents, so it has no opinion on forecasted/placeholder rows. Every
    other ("Existing") row with a Patent Number is dropped unless its
    normalised patent number is in the approved set for that drug.
    """
    required = ["Patent Number", "Drug Name"]
    missing = [c for c in required if c not in final_df.columns]
    if missing:
        print(f"  [WARN] Cannot apply patent_master approval filter — missing: {missing}")
        return final_df

    pn = final_df["Patent Number"]
    has_pn = pn.notna() & (pn.astype(str).str.strip() != "")

    if "Type" in final_df.columns:
        is_forecasted = final_df["Type"].astype(str).str.strip().str.lower() == "forecasted"
    else:
        is_forecasted = pd.Series(False, index=final_df.index)

    eligible = has_pn & ~is_forecasted
    if not eligible.any():
        print("  [patent_master filter] No existing-patent rows with a Patent Number — nothing to filter.")
        return final_df

    drug_names = final_df.loc[eligible, "Drug Name"].dropna().astype(str).unique()
    approved_by_drug = _load_approved_patent_numbers(project, dataset, table, drug_names, credentials_path)
    if not approved_by_drug:
        print("  [WARN] [patent_master filter] No approved patents returned — "
              "all existing-patent rows will be dropped.")

    drug_norm = final_df["Drug Name"].astype(str).str.strip().str.lower()
    pn_norm   = pn.map(_normalize_patent_number)

    approved_mask = pd.Series(
        [pn_norm.iat[i] in approved_by_drug.get(drug_norm.iat[i], set())
         for i in range(len(final_df))],
        index=final_df.index,
    )

    drop_mask = eligible & ~approved_mask
    n_dropped = int(drop_mask.sum())
    if n_dropped:
        dropped_drugs = sorted(final_df.loc[drop_mask, "Drug Name"].astype(str).str.strip().unique())
        _shown = ", ".join(dropped_drugs[:10])
        _more  = f" (+{len(dropped_drugs) - 10} more)" if len(dropped_drugs) > 10 else ""
        print(f"  [patent_master filter] Dropped {n_dropped} existing-patent row(s) not approved in "
              f"patent_master, across {len(dropped_drugs)} drug(s): {_shown}{_more}")
        final_df = final_df[~drop_mask].reset_index(drop=True)
    else:
        print(f"  [patent_master filter] All {int(eligible.sum())} existing-patent row(s) already approved.")

    return final_df


# ── Deduplication ─────────────────────────────────────────────────────────────

def _completeness(row) -> int:
    score = 0
    if not _is_blank(row.get("Approval Date")):
        score += 1
    for col in ("Est. Approval Year", "Exclusivity Year",
                "Controlling Patent Expiry Year", "PTE (months)"):
        if pd.notna(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]):
            score += 1
    return score


def _dedupe_duplicate_patents(final_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that represent the same physical patent recorded under
    different number formats (with/without underscores, kind-code suffix)."""
    required = ["Patent Number", "Drug Name", "Jurisdiction", "Filing Date"]
    if any(c not in final_df.columns for c in required):
        return final_df

    pn_norm   = final_df["Patent Number"].map(_normalize_patent_number)
    drug_norm = final_df["Drug Name"].astype(str).map(lambda s: str(s).strip().lower())
    dup_key   = (drug_norm
                 + "|" + final_df["Jurisdiction"].astype(str).str.strip().str.upper()
                 + "|" + final_df["Filing Date"].astype(str).str.strip()
                 + "|" + pn_norm)

    eligible = pn_norm != ""
    if "Type" in final_df.columns:
        eligible = eligible & (final_df["Type"].astype(str).str.strip().str.lower() != "forecasted")

    group_sizes = dup_key[eligible].value_counts()
    dup_keys = set(group_sizes[group_sizes > 1].index)
    if not dup_keys:
        return final_df

    drop_indices = []
    n_groups = 0
    for key in dup_keys:
        group_idx = final_df.index[eligible & (dup_key == key)]
        if len(group_idx) < 2:
            continue
        scores   = {i: _completeness(final_df.loc[i]) for i in group_idx}
        keep_idx = max(scores, key=scores.get)
        others   = [i for i in group_idx if i != keep_idx]
        for col in final_df.columns:
            if _is_blank(final_df.at[keep_idx, col]):
                for oi in others:
                    val = final_df.at[oi, col]
                    if not _is_blank(val):
                        final_df.at[keep_idx, col] = val
                        break
        drop_indices.extend(others)
        n_groups += 1

    if drop_indices:
        print(f"  Deduplicated {len(drop_indices)} duplicate patent row(s) across {n_groups} patent(s).")
        final_df = final_df.drop(index=drop_indices).reset_index(drop=True)
    return final_df


# ── Expiry from patent_discovery ──────────────────────────────────────────────

def _apply_patent_discovery_expiry(final_df: pd.DataFrame,
                                    patent_discovery_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Source Controlling Patent Expiry Year authoritatively from
    patent_discovery.expiry_date (matched by Patent Number).

    Resolution order per row:
      1. patent_discovery.expiry_date, matched on that row's own Patent Number.
      2. Existing ("non-Forecasted") rows only: the MAX patent_discovery-resolved
         expiry year among that same drug's OTHER Existing patents in the SAME
         Jurisdiction — i.e. the latest-expiring sibling patent "covers" this
         one when its own patent hasn't been discovered/matched yet.
      3. Filing Date + 20 — used for Forecasted rows unconditionally (they have
         no real patent number to look up), and for Existing rows only as a
         last resort when that drug+jurisdiction has no patent_discovery-
         resolved expiry at all to take a max from.
      4. Left blank if even Filing Date is unusable.
    """
    expiry_col, pn_col, filing_col = (
        "Controlling Patent Expiry Year", "Patent Number", "Filing Date")
    missing = [c for c in (expiry_col, pn_col, filing_col) if c not in final_df.columns]
    if missing:
        print(f"  [WARN] Cannot source {expiry_col!r} from patent_discovery — missing: {missing}")
        return final_df

    pd_pn_col = pd_expiry_col = None
    if patent_discovery_df is not None and not patent_discovery_df.empty:
        pd_pn_col     = _find_col(patent_discovery_df, ["patent", "number"])
        pd_expiry_col = _find_col(patent_discovery_df, ["expiry", "date"])

    def _expiry_year_from_date(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return np.nan
        if isinstance(val, pd.Timestamp):
            return float(val.year) if not pd.isna(val) else np.nan
        m = re.match(r"(\d{4})", str(val).strip())
        return float(m.group(1)) if m else np.nan

    # ── Step 1: direct match on this row's own Patent Number ─────────────────
    pd_expiry_year = pd.Series(np.nan, index=final_df.index)
    if pd_pn_col and pd_expiry_col:
        pdd = patent_discovery_df.copy()
        pdd["_pn_norm"] = pdd[pd_pn_col].map(_normalize_patent_number)
        expiry_map = (pdd[pdd["_pn_norm"] != ""]
                      .groupby("_pn_norm")[pd_expiry_col].first()
                      .to_dict())
        pn_norm = final_df[pn_col].map(_normalize_patent_number)
        pd_expiry_year = pn_norm.map(expiry_map).map(_expiry_year_from_date)
    else:
        print("  [WARN] patent_discovery missing patent_number/expiry_date — "
              "no row can be directly matched; falling through to drug-max / Filing Date + 20.")

    n_from_pd = int(pd_expiry_year.notna().sum())
    if n_from_pd:
        final_df.loc[pd_expiry_year.notna(), expiry_col] = pd_expiry_year[pd_expiry_year.notna()]

    still_blank = pd_expiry_year.isna()

    if "Type" in final_df.columns:
        is_forecasted = final_df["Type"].astype(str).str.strip().str.lower() == "forecasted"
    else:
        is_forecasted = pd.Series(False, index=final_df.index)
    is_existing = ~is_forecasted

    existing_unresolved   = still_blank & is_existing
    forecasted_unresolved = still_blank & is_forecasted  # always falls to Filing Date + 20 below

    # ── Step 2: Existing rows — max patent_discovery expiry among that drug's
    #            other Existing patents in the same jurisdiction ────────────
    n_from_drug_max = 0
    remaining_existing_unresolved = existing_unresolved
    if existing_unresolved.any():
        if "Drug Name" in final_df.columns and "Jurisdiction" in final_df.columns:
            drug_key  = final_df["Drug Name"].astype(str).str.strip().str.lower()
            jur_key   = final_df["Jurisdiction"].astype(str).str.strip().str.upper()
            group_key = drug_key + "|" + jur_key

            resolved_existing_mask = pd_expiry_year.notna() & is_existing
            max_by_group = (
                pd.DataFrame({"key": group_key[resolved_existing_mask],
                              "val": pd_expiry_year[resolved_existing_mask]})
                .groupby("key")["val"].max()
                .to_dict()
            )
            drug_max = group_key.map(max_by_group)

            apply_mask = existing_unresolved & drug_max.notna()
            n_from_drug_max = int(apply_mask.sum())
            if n_from_drug_max:
                final_df.loc[apply_mask, expiry_col] = drug_max[apply_mask]

            remaining_existing_unresolved = existing_unresolved & drug_max.isna()
        else:
            print("  [WARN] Missing 'Drug Name'/'Jurisdiction' — cannot compute the drug+jurisdiction "
                  "max fallback; existing rows go straight to Filing Date + 20 instead.")

    # ── Step 3: Filing Date + 20 — Forecasted rows always; Existing rows only
    #            when their drug+jurisdiction had no patent_discovery match at all ─
    needs_filing_fallback = forecasted_unresolved | remaining_existing_unresolved
    filing_years  = final_df.loc[needs_filing_fallback, filing_col].map(_filing_year)
    fallback_mask = filing_years.notna()
    n_fallback = int(fallback_mask.sum())
    if n_fallback:
        fallback_idx = filing_years[fallback_mask].index
        final_df.loc[fallback_idx, expiry_col] = filing_years[fallback_mask] + 20

    n_unresolved = int(needs_filing_fallback.sum()) - n_fallback
    print(f"  [patent_discovery expiry] From patent_discovery: {n_from_pd} row(s); "
          f"drug+jurisdiction max (existing rows): {n_from_drug_max} row(s); "
          f"Filing Date + 20 fallback: {n_fallback} row(s); "
          f"unresolved: {n_unresolved} row(s).")
    return final_df


# ── d2 builder ────────────────────────────────────────────────────────────────

def _build_d2(project: str, dataset: str, d1_table: str,
              patent_discovery_table: str, d2_table: str,
              credentials_path: Optional[str]) -> None:
    """Read d1, enrich with patent_discovery.expiry_date, recompute
    Years to Entry and Avg Years to Entry / IP Dimension Score, write to d2."""
    print(f"\n  [d2] Building {d2_table!r} from {d1_table!r} + {patent_discovery_table!r} ...")

    d1_df = _load_table_from_bq(project, dataset, d1_table, d1_table, credentials_path)
    if d1_df.empty:
        print(f"  [WARN] {d1_table!r} returned no rows — nothing to write to {d2_table!r}.")
        return

    pd_df = _load_table_from_bq(project, dataset, patent_discovery_table,
                                  "patent_discovery", credentials_path)

    pd_pn_col     = _find_col(pd_df, ["patent", "number"]) if not pd_df.empty else None
    pd_expiry_col = _find_col(pd_df, ["expiry", "date"])   if not pd_df.empty else None

    d1_pn_col   = _find_col(d1_df, ["patent", "number"])
    type_col    = _find_col(d1_df, ["type"])
    filing_col  = _find_col(d1_df, ["filing", "date"]) or _find_col(d1_df, ["filingdate"])
    cpey_col    = _find_col(d1_df, ["controlling", "expiry"])
    excl_col    = _find_col(d1_df, ["exclusivity", "year"])
    yte_col     = _find_col(d1_df, ["years", "entry"])
    avg_col     = _find_col(d1_df, ["avgyears"]) or _find_col(d1_df, ["avg", "years"])
    score_col   = _find_col(d1_df, ["ipdimension"]) or _find_col(d1_df, ["ip", "dimension"])
    drug_col    = _find_col(d1_df, ["drug", "name"]) or _find_col(d1_df, ["drugname"])
    jur_col     = _find_col(d1_df, ["jurisdiction"])

    d2_df = d1_df.copy()

    # Step 1: join expiry_date from patent_discovery
    if not pd_df.empty and pd_pn_col and pd_expiry_col and d1_pn_col:
        pd_df = pd_df.copy()
        pd_df["_pn_norm"] = pd_df[pd_pn_col].map(_normalize_patent_number)
        d2_df["_pn_norm"] = d2_df[d1_pn_col].map(_normalize_patent_number)
        expiry_map = (pd_df[pd_df["_pn_norm"] != ""]
                      .groupby("_pn_norm")[pd_expiry_col].first().to_dict())
        d2_df["expiry_date"] = d2_df["_pn_norm"].map(expiry_map)
        n_matched = int(d2_df["expiry_date"].notna().sum())
        print(f"  [d2] Matched {n_matched}/{len(d2_df)} rows to patent_discovery on patent_number.")
        d2_df = d2_df.drop(columns=["_pn_norm"])
    else:
        d2_df["expiry_date"] = np.nan
        print("  [WARN] [d2] Skipping join — missing required columns or empty table.")

    # Step 2: set Controlling_Patent_Expiry_Year
    if not cpey_col:
        cpey_col = "Controlling_Patent_Expiry_Year"
        d2_df[cpey_col] = np.nan

    def _expiry_year_from_date(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return np.nan
        if isinstance(val, pd.Timestamp):
            return float(val.year) if not pd.isna(val) else np.nan
        m = re.match(r"(\d{4})", str(val).strip())
        return float(m.group(1)) if m else np.nan

    existing_mask   = pd.Series(True, index=d2_df.index)
    forecasted_mask = pd.Series(False, index=d2_df.index)
    if type_col and type_col in d2_df.columns:
        forecasted_mask = d2_df[type_col].astype(str).str.strip().str.lower() == "forecasted"
        existing_mask   = ~forecasted_mask

    existing_expiry_year = d2_df["expiry_date"].map(_expiry_year_from_date)
    d2_df.loc[existing_mask, cpey_col] = existing_expiry_year[existing_mask]
    n_existing_matched = int(existing_expiry_year[existing_mask].notna().sum())
    print(f"  [d2] Existing rows: {cpey_col!r} = year(expiry_date) for "
          f"{n_existing_matched}/{int(existing_mask.sum())} rows.")

    n_forecasted = int(forecasted_mask.sum())
    if n_forecasted > 0 and filing_col and filing_col in d2_df.columns:
        filing_years = d2_df.loc[forecasted_mask, filing_col].map(_filing_year)
        d2_df.loc[forecasted_mask, cpey_col] = filing_years + 20
        print(f"  [d2] Forecasted rows: {cpey_col!r} = Filing_Year + 20 for "
              f"{int(filing_years.notna().sum())}/{n_forecasted} rows.")

    # Step 3: recompute Years_to_Entry
    if yte_col and yte_col in d2_df.columns:
        current_year = datetime.now().year
        expiry_num = pd.to_numeric(d2_df[cpey_col], errors="coerce")
        excl_num   = (pd.to_numeric(d2_df[excl_col], errors="coerce").fillna(0)
                      if excl_col and excl_col in d2_df.columns
                      else pd.Series(0, index=d2_df.index))
        effective_expiry = np.maximum(expiry_num.fillna(0), excl_num)
        valid_mask = effective_expiry >= (current_year - 50)
        d2_df.loc[valid_mask, yte_col] = (effective_expiry[valid_mask] - current_year).round(1)
        print(f"  [d2] Recalculated {yte_col!r} for {int(valid_mask.sum())} rows.")

    # Step 4: recalculate Avg_Years_to_Entry_US_EP and IP_Dimension_1_Score
    if (drug_col and jur_col and yte_col and avg_col and score_col
            and all(c in d2_df.columns for c in (drug_col, jur_col, yte_col, avg_col, score_col))):
        yte_num_d2 = pd.to_numeric(d2_df[yte_col], errors="coerce")
        drugs_in_d2 = d2_df[drug_col].dropna().astype(str).str.strip().unique()
        n_avg_updated = 0
        still_missing_avg = []
        for drug in drugs_in_d2:
            drug_mask = d2_df[drug_col].astype(str).str.strip().str.lower() == drug.strip().lower()
            jt = d2_df.loc[drug_mask, jur_col].map(_jurisdiction_token)
            us_mask = drug_mask & (jt == ("US", 5))
            eu_mask = drug_mask & (jt == ("EU", 10))
            us_max  = yte_num_d2[us_mask].max() if us_mask.any() else np.nan
            eu_max  = yte_num_d2[eu_mask].max() if eu_mask.any() else np.nan
            vals    = [v for v in (us_max, eu_max) if pd.notna(v)]
            if not vals:
                still_missing_avg.append(drug)
                continue
            avg_val = sum(vals) / len(vals)
            d2_df.loc[drug_mask, avg_col]   = round(avg_val, 1)
            d2_df.loc[drug_mask, score_col] = _ip_dimension_score(avg_val)
            n_avg_updated += 1
        print(f"\n  [d2] Avg YTE / IP Dimension Score updated for "
              f"{n_avg_updated}/{len(drugs_in_d2)} drug(s)."
              + (f" No US/EU YTE data for: {', '.join(sorted(still_missing_avg)[:10])}"
                 + (" ..." if len(still_missing_avg) > 10 else "")
                 if still_missing_avg else ""))

    _upload_to_bigquery(d2_df, project, dataset, d2_table,
                         credentials_path, write_disposition="WRITE_TRUNCATE")
    print(f"  [d2] Done — {d2_table!r} written ({len(d2_df)} rows, {len(d2_df.columns)} columns).")
