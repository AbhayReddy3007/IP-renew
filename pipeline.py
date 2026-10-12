"""
pipeline.py — Top-level combine() orchestrator.

Imports from all other modules and runs the full pipeline:
  1. Load data (BigQuery or local files)
  2. Skip forecasted_loe rows for Marketed/Pre-registration drugs
  3. Align and concat the two tables
  4. Select final output columns
  5. Restrict to the patent_master-approved patent shortlist
  6. Deduplicate patents
  7. Reset always-recalculated fields
  8. Source Controlling Patent Expiry Year from patent_discovery
  9. Calculate base Years to Entry
 10. Overwrite Phase / derived fields from clinical_efficacy / vw_drug_details
 11. Override Marketed/Pre-registration rows with real LOE / approval dates
 12. Fill missing Avg Years to Entry / IP Dimension Score
 13. Export to Excel
 14. Upload to BigQuery (d1)
 15. Build d2

Step 5 ensures every patent-level computation from that point on (dedup,
expiry sourcing, Years to Entry, d2) only ever sees patents patent_master
marks approved (patent_status = 'PDF Downloaded', confidence > 0.3) —
mirroring IP_refactored.chunking.patent_filter's approval criteria.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd

from config import (
    GCS_CREDENTIALS, DROP_COLUMNS,
    _ALWAYS_RECALCULATED_FIELDS, _TEXT_ALWAYS_RECALCULATED_FIELDS,
)
from utils import (
    _find_col, _norm, _normalize_drug_str,
    _align_columns, _ensure_column, _select_final_columns,
    _filter_to_drug, _jurisdiction_token, _ip_dimension_score,
)
from bq_io import _load_table_from_bq, _load_table_from_file, _upload_to_bigquery
from clinical import _apply_clinical_efficacy
from loe_dates import (
    _apply_marketed_approval_dates,
    _override_avg_years_from_loe,
    _find_loe_columns,
)
from patents import (
    _filter_to_approved_patents,
    _dedupe_duplicate_patents,
    _apply_patent_discovery_expiry,
    _build_d2,
)


# ── Avg / IP Dimension score helpers ─────────────────────────────────────────

def _fill_missing_avg_score(final_df: pd.DataFrame) -> pd.DataFrame:
    """For drugs still missing Avg Years to Entry (US & EP), calculate a
    fallback from the max US and EU Years to Entry values."""
    drug_col, jurisdiction_col, yte_col = "Drug Name", "Jurisdiction", "Years to Entry"
    avg_col, score_col = "Avg Years to Entry (US & EP)", "IP Dimension 1 Score"

    missing = [c for c in (drug_col, jurisdiction_col, yte_col, avg_col, score_col)
               if c not in final_df.columns]
    if missing:
        print(f"  [WARN] Cannot compute fallback Avg/IP Dimension score — missing: {missing}")
        return final_df

    avg_num = pd.to_numeric(final_df[avg_col], errors="coerce")
    missing_mask  = avg_num.isna()
    missing_drugs = final_df.loc[missing_mask, drug_col].dropna().astype(str).str.strip().unique()
    if len(missing_drugs) == 0:
        return final_df

    yte_num  = pd.to_numeric(final_df[yte_col], errors="coerce")
    n_filled = 0
    still_missing = []
    for drug in missing_drugs:
        drug_mask = final_df[drug_col].astype(str).str.strip().str.lower() == drug.strip().lower()
        jt = final_df.loc[drug_mask, jurisdiction_col].map(_jurisdiction_token)
        us_mask = drug_mask & (jt == ("US", 5))
        eu_mask = drug_mask & (jt == ("EU", 10))
        us_max  = yte_num[us_mask].max() if us_mask.any() else np.nan
        eu_max  = yte_num[eu_mask].max() if eu_mask.any() else np.nan
        vals    = [v for v in (us_max, eu_max) if pd.notna(v)]
        if not vals:
            still_missing.append(drug)
            continue
        avg_val = sum(vals) / len(vals)
        final_df.loc[drug_mask, avg_col]   = round(avg_val, 1)
        final_df.loc[drug_mask, score_col] = _ip_dimension_score(avg_val)
        n_filled += 1

    print(f"\n  [Avg/IP Dimension fallback] {len(missing_drugs)} drug(s) missing from "
          f"forecasted_loe. Calculated fallback for {n_filled}."
          + (f" No US/EU YTE data for: {', '.join(sorted(still_missing)[:10])}"
             + (" ..." if len(still_missing) > 10 else "")
             if still_missing else " All filled."))
    return final_df


# ── Reset always-recalculated fields ─────────────────────────────────────────

def _reset_always_recalculated_fields(final_df: pd.DataFrame) -> pd.DataFrame:
    """Blank Phase / Years to Entry / Avg Years to Entry / Controlling Patent
    Expiry Year / IP Dimension 1 Score unconditionally before any calculation
    so nothing is ever carried over from the source tables."""
    present = [c for c in _ALWAYS_RECALCULATED_FIELDS if c in final_df.columns]
    missing = [c for c in _ALWAYS_RECALCULATED_FIELDS if c not in final_df.columns]
    for col in present:
        if col in _TEXT_ALWAYS_RECALCULATED_FIELDS:
            final_df[col] = final_df[col].astype(object)
            final_df[col] = None
        else:
            final_df[col] = np.nan
    print(f"  [reset] Cleared {len(present)} field(s): {present}."
          + (f" Not present (nothing to clear): {missing}." if missing else ""))
    return final_df


# ── Marketed-anywhere skip helper ─────────────────────────────────────────────

def _marketed_or_prereg_drug_set(drug_details_df: pd.DataFrame) -> set:
    """Return the set of normalised drug names that are Marketed or
    Pre-registration in at least one jurisdiction (per vw_drug_details)."""
    if drug_details_df is None or drug_details_df.empty:
        return set()

    from utils import _norm_phase
    name_col = (_find_col(drug_details_df, ["cleaned", "generic", "name"])
                or _find_col(drug_details_df, ["generic", "name"])
                or _find_col(drug_details_df, ["drug", "name"])
                or _find_col(drug_details_df, ["drugname"]))
    phase_col = (_find_col(drug_details_df, ["highest", "development", "stage"])
                 or _find_col(drug_details_df, ["development", "stage"]))
    if not name_col or not phase_col:
        return set()

    name_norm  = drug_details_df[name_col].astype(str).map(_normalize_drug_str)
    phases     = drug_details_df[phase_col].map(_norm_phase)
    skip_mask  = phases.isin(["approved/marketed", "pre-registration"])
    return set(name_norm[skip_mask].unique()) - {""}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def combine(args) -> None:
    credentials_path = args.credentials or GCS_CREDENTIALS or None
    target_project   = args.target_project or args.project
    target_dataset   = args.target_dataset or args.dataset
    output           = args.output

    # ── 1. Load source tables ─────────────────────────────────────────────────
    if args.file1:
        df1 = _load_table_from_file(args.file1, "file 1 (forecasted_loe)")
    else:
        df1 = _load_table_from_bq(args.project, args.dataset, args.forecast_table,
                                   "Table 1 (forecasted_loe)", credentials_path)

    if args.file2:
        df2 = _load_table_from_file(args.file2, "file 2 (Master_LOE)")
    else:
        df2 = _load_table_from_bq(args.project, args.dataset, args.master_table,
                                   "Table 2 (Master_LOE)", credentials_path)

    if df1.empty:
        print("[ERROR] forecasted_loe returned no rows."); sys.exit(1)
    if df2.empty:
        print("[ERROR] Master_LOE returned no rows."); sys.exit(1)

    print(f"\n  Raw columns in file 1 (forecasted_loe): {list(df1.columns)}")
    print(f"  Raw columns in file 2 (Master_LOE):     {list(df2.columns)}")

    df1 = _ensure_column(df1, "no_of_forecasted_patents",
                          [["noofforecastedpatents"],
                           ["forecasted", "patents"],
                           ["forecastedcount"],
                           ["patents"]],
                          label="forecasted_loe")
    df2 = _ensure_column(df2, "Pediatric_Exclusivity", ["pediatric"], label="loe_table_2")
    df2 = _ensure_column(df2, "Approval_Date", ["approval", "date"],
                          exclude_tokens=["source", "year"], label="loe_table_2")

    master_loe_original_columns = list(df2.columns)

    # ── Drug filter ───────────────────────────────────────────────────────────
    if args.drug:
        print(f"\n  Filtering to drug: {args.drug!r}")
        df1 = _filter_to_drug(df1, args.drug, "Table 1 (forecasted_loe)")
        df2 = _filter_to_drug(df2, args.drug, "Table 2 (Master_LOE)")
        if df1.empty and df2.empty:
            print(f"[ERROR] No rows matched drug={args.drug!r}."); sys.exit(1)

    # ── 2. Skip forecasted_loe rows for Marketed/Pre-registration drugs ───────
    drug_details_df: Optional[pd.DataFrame] = None
    if not args.no_marketed_skip:
        drug_details_df = _load_table_from_bq(
            args.clinical_project, args.clinical_dataset,
            args.drug_details_table, "vw_drug_details_full", credentials_path)
        skip_set = _marketed_or_prereg_drug_set(drug_details_df)
        df1_drug_col = _find_col(df1, ["drug"])
        if skip_set and df1_drug_col and not df1.empty:
            df1_drug_norm = df1[df1_drug_col].astype(str).map(_normalize_drug_str)
            skip_mask     = df1_drug_norm.isin(skip_set)
            n_dropped     = int(skip_mask.sum())
            if n_dropped:
                dropped_names = sorted(df1.loc[skip_mask, df1_drug_col].astype(str).str.strip().unique())
                df1 = df1[~skip_mask]
                _shown = ", ".join(dropped_names[:10])
                _more  = f" (+{len(dropped_names) - 10} more)" if len(dropped_names) > 10 else ""
                print(f"\n  Skipping Table 1 rows for {len(dropped_names)} Marketed/Pre-registration "
                      f"drug(s): {_shown}{_more} — dropped {n_dropped} forecasted row(s).")

    # ── 3. Align + concat ─────────────────────────────────────────────────────
    df1 = _align_columns(df1, df2)

    drug_col  = _find_col(df1, ["drug"])
    avg_col   = _find_col(df1, ["avgyears"], prefer_tokens=["usep"])
    score_col = _find_col(df1, ["ipdimension"])

    for col_name, col_val in (("drug", drug_col), ("avg", avg_col), ("score", score_col)):
        if not col_val:
            print(f"[ERROR] Could not find a '{col_name}' column in file 1."); sys.exit(1)

    avg_map:   Dict[str, float] = {str(k).strip().lower(): v for k, v in
                                    df1.groupby(drug_col)[avg_col].first().to_dict().items()}
    score_map: Dict[str, float] = {str(k).strip().lower(): v for k, v in
                                    df1.groupby(drug_col)[score_col].first().to_dict().items()}

    combined = pd.concat([df1, df2], ignore_index=True, sort=False)
    print(f"\n  Combined rows: {len(combined)} | Combined columns: {len(combined.columns)}")

    # Fill blank Type with "Existing"
    type_col = _find_col(combined, ["type"])
    if type_col:
        blank_mask = combined[type_col].isna() | (combined[type_col].astype(str).str.strip() == "")
        n_blank = int(blank_mask.sum())
        combined.loc[blank_mask, type_col] = "Existing"
        print(f"  Filled {n_blank} blank {type_col!r} value(s) with 'Existing'")

    # Set Grant Date = Filing Date for forecasted rows
    filing_col = _find_col(combined, ["filing"], prefer_tokens=["date"])
    grant_col  = _find_col(combined, ["grant"],  prefer_tokens=["date"])
    if type_col and filing_col and grant_col:
        forecasted_mask = combined[type_col].astype(str).str.strip().str.lower() == "forecasted"
        combined[grant_col] = combined[grant_col].astype(object)
        combined.loc[forecasted_mask, grant_col] = combined.loc[forecasted_mask, filing_col]
        print(f"  Set {grant_col!r} = {filing_col!r} for {int(forecasted_mask.sum())} forecasted row(s)")

    # Drop unwanted columns
    to_drop = [c for c in combined.columns if _norm(c) in {_norm(d) for d in DROP_COLUMNS}]
    if to_drop:
        combined = combined.drop(columns=to_drop)
        print(f"  Dropped column(s): {', '.join(to_drop)}")

    # Stamp avg/score from file 1 across all rows
    combined_drug_col  = drug_col  if drug_col  in combined.columns else _find_col(combined, ["drug"])
    combined_avg_col   = avg_col   if avg_col   in combined.columns else _find_col(combined, ["avgyears"], ["usep"])
    combined_score_col = score_col if score_col in combined.columns else _find_col(combined, ["ipdimension"])

    if not combined_drug_col:
        print("[ERROR] Could not find 'Drug' column in combined dataframe."); sys.exit(1)

    key = combined[combined_drug_col].astype(str).str.strip().str.lower()
    combined[combined_avg_col]   = key.map(avg_map)
    combined[combined_score_col] = key.map(score_map)
    print(f"  Stamped {combined_avg_col!r} and {combined_score_col!r} from file 1 across all rows.")

    unmatched = combined.loc[combined[combined_avg_col].isna(), combined_drug_col].dropna().unique()
    if len(unmatched):
        print(f"  [WARN] {len(unmatched)} drug(s) in file 2 had no match in file 1: "
              f"{', '.join(sorted(map(str, unmatched))[:10])}"
              + (" ..." if len(unmatched) > 10 else ""))

    # ── 4. Select final columns ───────────────────────────────────────────────
    print("\n  Selecting final output columns:")
    final_df, used_cols = _select_final_columns(combined)

    # Preserve all loe_table_2 columns not already captured by FINAL_COLUMNS
    leftover_master_cols = [c for c in master_loe_original_columns
                             if c in combined.columns and c not in used_cols]
    if leftover_master_cols:
        for c in leftover_master_cols:
            final_df[c] = combined[c].values
        print(f"  Kept {len(leftover_master_cols)} additional loe_table_2 column(s): "
              f"{', '.join(leftover_master_cols)}")

    # ── 5. Restrict to patent_master-approved patents ─────────────────────────
    if not args.no_patent_filter:
        final_df = _filter_to_approved_patents(
            final_df, target_project, target_dataset, args.patent_master_table, credentials_path)
    else:
        print("\n  --no-patent-filter given — skipping patent_master approval filter.")

    # ── 6. Deduplicate patents ────────────────────────────────────────────────
    final_df = _dedupe_duplicate_patents(final_df)

    # ── 7. Reset always-recalculated fields ───────────────────────────────────
    final_df = _reset_always_recalculated_fields(final_df)

    # ── 8. Source Controlling Patent Expiry Year ──────────────────────────────
    patent_discovery_df = _load_table_from_bq(
        target_project, target_dataset, args.patent_discovery_table,
        "patent_discovery", credentials_path)
    final_df = _apply_patent_discovery_expiry(final_df, patent_discovery_df)

    # ── 9. Calculate base Years to Entry ─────────────────────────────────────
    current_year = datetime.now().year
    yte_col    = "Years to Entry"
    expiry_col = "Controlling Patent Expiry Year"
    excl_col   = "Exclusivity Year"

    if yte_col in final_df.columns and expiry_col in final_df.columns:
        expiry_num = pd.to_numeric(final_df[expiry_col], errors="coerce")
        excl_num   = pd.to_numeric(
            final_df.get(excl_col, pd.Series(index=final_df.index, dtype=float)),
            errors="coerce").reindex(final_df.index)
        effective_expiry = np.maximum(expiry_num.fillna(0), excl_num.fillna(0))
        valid_expiry     = effective_expiry >= (current_year - 50)
        derived_yte      = effective_expiry - current_year
        n_calculated     = int(valid_expiry.sum())
        if n_calculated:
            final_df.loc[valid_expiry, yte_col] = derived_yte[valid_expiry].round(1)
            print(f"  Calculated {yte_col!r} for {n_calculated} row(s) "
                  f"(max({expiry_col}, {excl_col}) - {current_year}); "
                  f"{len(final_df) - n_calculated} row(s) had no valid expiry.")
        else:
            print(f"  [WARN] No row had a valid {expiry_col!r} — {yte_col!r} left as-is.")

    # ── 10. Clinical efficacy / vw_drug_details phase overwrite ──────────────
    if not args.no_clinical:
        clinical_df = _load_table_from_bq(
            args.clinical_project, args.clinical_dataset,
            args.clinical_table, "clinical_efficacy", credentials_path)
        if drug_details_df is None:
            drug_details_df = _load_table_from_bq(
                args.clinical_project, args.clinical_dataset,
                args.drug_details_table, "vw_drug_details_full", credentials_path)
        if not clinical_df.empty or (drug_details_df is not None and not drug_details_df.empty):
            final_df = _apply_clinical_efficacy(final_df, clinical_df, drug_details_df)
        else:
            print("  [WARN] clinical_efficacy and vw_drug_details both empty — skipping.")
    else:
        print("\n  --no-clinical given — skipping clinical_efficacy/vw_drug_details step.")

    # ── 11. Marketed / Pre-registration real-approval-date overrides ─────────
    if not args.no_marketed_approval:
        final_df = _apply_marketed_approval_dates(final_df, drug_details_df=drug_details_df)
        final_df = _override_avg_years_from_loe(final_df, drug_details_df=drug_details_df)
    else:
        print("\n  --no-marketed-approval given — skipping real approval-date lookup.")

    # ── 12. Fill missing Avg Years to Entry / IP Dimension Score ─────────────
    final_df = _fill_missing_avg_score(final_df)

    # ── 13. Export to Excel ───────────────────────────────────────────────────
    out_path = Path(output)
    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Combined")
        ws = writer.sheets["Combined"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col_cells in ws.columns:
            length = max(len(str(cell.value)) if cell.value is not None else 0
                         for cell in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(length + 4, 60)

    print(f"\n  Saved: {out_path.resolve()}")
    print(f"  Total rows: {len(final_df)} | Total columns: {len(final_df.columns)}")

    # ── 14. Upload to BigQuery (d1) ───────────────────────────────────────────
    if args.no_upload:
        print("\n  [BQ] --no-upload given — skipping BigQuery upload.")
    else:
        write_disposition = {"truncate": "WRITE_TRUNCATE", "append": "WRITE_APPEND"}[args.write_mode]
        _upload_to_bigquery(final_df, target_project, target_dataset, args.target_table,
                             credentials_path, write_disposition)

        # ── 15. Build d2 ──────────────────────────────────────────────────────
        if not args.no_d2:
            _build_d2(
                project=target_project,
                dataset=target_dataset,
                d1_table=args.target_table,
                patent_discovery_table=args.patent_discovery_table,
                d2_table=args.d2_table,
                credentials_path=credentials_path,
            )
