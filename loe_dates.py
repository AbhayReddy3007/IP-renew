"""
loe_dates.py — Real-approval-date overrides for Marketed/Pre-registration rows,
and LOE_Date-based recomputation of Avg Years to Entry / IP Dimension 1 Score.

Depends on vw_drug_details_full (LOE_Date, First_Approval_Date columns).
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from utils import (
    _find_col, _normalize_drug_str, _jurisdiction_token,
    _vwd_geo_tokens, _vwd_geo_tokens_combined,
    _approval_year_from_value, _ip_dimension_score, _norm_ws,
)


# ── Column discovery ──────────────────────────────────────────────────────────

def _find_loe_columns(drug_details_df: pd.DataFrame, verbose: bool = False):
    """Locate name_col, geo_col, geo_fallback_col, loe_col, approval_col in
    vw_drug_details_full.  Returns a 5-tuple (any may be None if absent)."""
    if drug_details_df is None or drug_details_df.empty:
        return None, None, None, None, None
    name_col = (_find_col(drug_details_df, ["cleaned", "generic", "name"])
                or _find_col(drug_details_df, ["generic", "name"])
                or _find_col(drug_details_df, ["drug", "name"])
                or _find_col(drug_details_df, ["drugname"]))
    geo_col = (_find_col(drug_details_df, ["drug", "geo"], prefer_tokens=["new"])
               or _find_col(drug_details_df, ["geo"], prefer_tokens=["new"]))
    geo_fallback_col = _find_col(drug_details_df, ["drug", "geography"])
    loe_col  = _find_col(drug_details_df, ["loe", "date"])
    approval_col = _find_col(drug_details_df, ["first", "approval", "date"])
    if verbose:
        print(f"  [LOE columns] name_col={name_col!r}  geo_col={geo_col!r}  "
              f"geo_fallback_col={geo_fallback_col!r}  loe_col={loe_col!r}  "
              f"approval_col={approval_col!r}  (all: {list(drug_details_df.columns)})")
    return name_col, geo_col, geo_fallback_col, loe_col, approval_col


def _lookup_loe_and_first_approval(drug_details_df, name_col, geo_col, loe_col,
                                    approval_col, drug, token, debug=False,
                                    geo_fallback_col=None):
    """Return (loe_year, first_approval_year) for a given drug + jurisdiction
    token ('US' or 'EU') from vw_drug_details_full.  Either value is None
    if the column is blank or there's no matching row."""
    if not (name_col and geo_col and loe_col) or token not in ("US", "EU"):
        return None, None

    name_norm = drug_details_df[name_col].astype(str).map(_normalize_drug_str)
    rows = drug_details_df[name_norm == _normalize_drug_str(drug)]
    if rows.empty:
        if debug:
            print(f"    [LOE debug] {drug!r} ({token}): no rows matched by name.")
        return None, None

    if geo_fallback_col:
        geo_match = rows.apply(
            lambda r: token in _vwd_geo_tokens_combined(r[geo_col], r[geo_fallback_col]), axis=1)
    else:
        geo_match = rows[geo_col].map(lambda g: token in _vwd_geo_tokens(g))
    matched = rows[geo_match]

    if debug:
        raw_geo = list(rows[geo_col])
        raw_fallback = list(rows[geo_fallback_col]) if geo_fallback_col else None
        print(f"    [LOE debug] {drug!r} ({token}): {len(rows)} name-matched, "
              f"{len(matched)} jurisdiction-matched. "
              f"Raw {geo_col!r}: {raw_geo}"
              + (f"; fallback {geo_fallback_col!r}: {raw_fallback}" if geo_fallback_col else ""))

    if matched.empty:
        return None, None

    loe_year = None
    for v in matched[loe_col]:
        y = _approval_year_from_value(v)
        if debug:
            print(f"    [LOE debug] {drug!r} ({token}): raw {loe_col!r}={v!r} -> year={y}")
        if y is not None:
            loe_year = y
            break

    approval_year = None
    if approval_col:
        for v in matched[approval_col]:
            y = _approval_year_from_value(v)
            if y is not None:
                approval_year = y
                break

    return loe_year, approval_year


# ── Write helpers ─────────────────────────────────────────────────────────────

def _write_loe_based_result(final_df, token_mask, loe_year, approval_year, excl_years, current_year):
    """Write LOE_Date-based Controlling Patent Expiry Year, Exclusivity Year,
    Est. Approval Year, and Years to Entry for rows in token_mask."""
    final_df.loc[token_mask, "Controlling Patent Expiry Year"] = loe_year
    if approval_year is not None:
        exclusivity_year = approval_year + excl_years
        effective_expiry = max(loe_year, exclusivity_year)
        final_df.loc[token_mask, "Est. Approval Year"] = approval_year
        final_df.loc[token_mask, "Exclusivity Year"]   = exclusivity_year
    else:
        effective_expiry = loe_year
    final_df.loc[token_mask, "Years to Entry"] = round(effective_expiry - current_year, 1)


def _write_marketed_approval_result(final_df, token_mask, approval_year, excl_years, current_year):
    """Write First_Approval_Date-based Exclusivity Year / Years to Entry for
    rows in token_mask (no LOE_Date path — approval year known, LOE year not)."""
    exclusivity_year = approval_year + excl_years
    expiry_num = pd.to_numeric(final_df.loc[token_mask, "Controlling Patent Expiry Year"], errors="coerce")
    effective_expiry = expiry_num.where(
        expiry_num.isna() | (expiry_num >= exclusivity_year), exclusivity_year
    ).fillna(exclusivity_year)
    final_df.loc[token_mask, "Est. Approval Year"] = approval_year
    final_df.loc[token_mask, "Exclusivity Year"]   = exclusivity_year
    final_df.loc[token_mask, "Years to Entry"]     = (effective_expiry - current_year).round(1)


# ── Public entry points ───────────────────────────────────────────────────────

def _apply_marketed_approval_dates(final_df: pd.DataFrame,
                                    drug_details_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Override Controlling Patent Expiry Year / Exclusivity Year / Years to Entry
    for Marketed and Pre-registration rows, using LOE_Date first, then
    First_Approval_Date, from vw_drug_details_full."""
    current_year = datetime.now().year
    required = ("Phase", "Drug Name", "Jurisdiction",
                "Est. Approval Year", "Exclusivity Year", "Years to Entry")
    missing = [c for c in required if c not in final_df.columns]
    if missing:
        print(f"  [WARN] Cannot apply marketed-approval logic — missing: {missing}")
        return final_df

    is_marketed = final_df["Phase"].astype(str).str.strip().str.lower() == "approved/marketed"
    is_prereg   = final_df["Phase"].astype(str).str.strip().str.lower() == "pre-registration"
    is_target   = is_marketed | is_prereg
    if not is_target.any():
        print("\n  [Marketed approval dates] No Marketed/Pre-registration rows — nothing to do.")
        return final_df

    loe_name_col, loe_geo_col, loe_geo_fallback_col, loe_col, loe_approval_col = _find_loe_columns(
        drug_details_df, verbose=True)
    if not (loe_name_col and loe_geo_col and loe_col):
        print(f"\n  [WARN] vw_drug_details_full missing usable columns — LOE_Date path skipped.")
    if not loe_approval_col:
        print(f"\n  [WARN] vw_drug_details_full missing First_Approval_Date — fallback unavailable.")

    n_loe_based = n_from_first_approval = n_unresolved = 0
    target_drugs = final_df.loc[is_target, "Drug Name"].dropna().astype(str).str.strip().unique()

    for drug in target_drugs:
        drug_mask = is_target & (final_df["Drug Name"].astype(str).str.strip() == drug)
        jt = final_df.loc[drug_mask, "Jurisdiction"].map(_jurisdiction_token)

        for token, excl_years in (("US", 5), ("EU", 10)):
            token_mask = drug_mask & (jt == (token, excl_years))
            if not token_mask.any():
                continue

            loe_year = loe_approval_year = None
            if loe_name_col and loe_geo_col and loe_col:
                loe_year, loe_approval_year = _lookup_loe_and_first_approval(
                    drug_details_df, loe_name_col, loe_geo_col, loe_col, loe_approval_col,
                    drug, token, geo_fallback_col=loe_geo_fallback_col)

            if loe_year is not None:
                approval_year_for_excl = loe_approval_year
                if approval_year_for_excl is None and "Est. Approval Year" in final_df.columns:
                    existing_est = pd.to_numeric(
                        final_df.loc[token_mask, "Est. Approval Year"], errors="coerce").dropna()
                    if not existing_est.empty:
                        approval_year_for_excl = float(existing_est.iloc[0])
                n_loe_based += 1
                _write_loe_based_result(final_df, token_mask, loe_year, approval_year_for_excl,
                                         excl_years, current_year)
                continue

            if loe_approval_year is not None:
                n_from_first_approval += 1
                if "Approval Date" in final_df.columns:
                    final_df.loc[token_mask, "Approval Date"] = str(int(round(loe_approval_year)))
                _write_marketed_approval_result(final_df, token_mask, loe_approval_year,
                                                 excl_years, current_year)
            else:
                n_unresolved += 1

    print(f"\n  [Marketed/Pre-reg approval dates] {n_loe_based} pair(s) used LOE_Date directly; "
          f"{n_from_first_approval} used First_Approval_Date; "
          f"{n_unresolved} kept phase-only estimate (no approval data found).")
    return final_df


def _override_avg_years_from_loe(final_df: pd.DataFrame,
                                  drug_details_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Recompute Years to Entry (per row) AND Avg Years to Entry (US & EP) (per drug)
    from vw_drug_details_full's real LOE_Date, overriding forecasted_loe's stamped value."""
    drug_col, jurisdiction_col = "Drug Name", "Jurisdiction"
    yte_col  = "Years to Entry"
    avg_col  = "Avg Years to Entry (US & EP)"
    score_col = "IP Dimension 1 Score"

    missing = [c for c in (drug_col, jurisdiction_col, yte_col, avg_col, score_col)
               if c not in final_df.columns]
    if missing:
        print(f"  [WARN] Cannot override Avg Years from LOE_Date — missing: {missing}")
        return final_df

    name_col, geo_col, geo_fallback_col, loe_col, approval_col = _find_loe_columns(
        drug_details_df, verbose=True)
    if not (name_col and geo_col and loe_col):
        print("\n  [Avg/Years override] vw_drug_details_full missing required columns — skipping.")
        return final_df

    current_year = datetime.now().year
    drugs = final_df[drug_col].dropna().astype(str).str.strip().unique()
    n_rows_overridden = n_drugs_avg_overridden = 0

    for drug in drugs:
        drug_mask = final_df[drug_col].astype(str).str.strip().str.lower() == drug.strip().lower()
        jt = final_df.loc[drug_mask, jurisdiction_col].map(_jurisdiction_token)
        effective_by_token = {}

        for token, excl_years in (("US", 5), ("EU", 10)):
            token_mask = drug_mask & (jt == (token, excl_years))
            if not token_mask.any():
                continue
            loe_year, approval_year = _lookup_loe_and_first_approval(
                drug_details_df, name_col, geo_col, loe_col, approval_col,
                drug, token, debug=True, geo_fallback_col=geo_fallback_col)
            if loe_year is None:
                continue

            approval_year_for_excl = approval_year
            if approval_year_for_excl is None and "Est. Approval Year" in final_df.columns:
                existing_est = pd.to_numeric(
                    final_df.loc[token_mask, "Est. Approval Year"], errors="coerce").dropna()
                if not existing_est.empty:
                    approval_year_for_excl = float(existing_est.iloc[0])

            _write_loe_based_result(final_df, token_mask, loe_year,
                                     approval_year_for_excl, excl_years, current_year)
            n_rows_overridden += int(token_mask.sum())
            effective_by_token[token] = (
                max(loe_year, approval_year_for_excl + excl_years)
                if approval_year_for_excl is not None else loe_year)

        if not effective_by_token:
            continue
        avg_effective_expiry = sum(effective_by_token.values()) / len(effective_by_token)
        avg_years = round(avg_effective_expiry - current_year, 1)
        final_df.loc[drug_mask, avg_col]   = avg_years
        final_df.loc[drug_mask, score_col] = _ip_dimension_score(avg_years)
        n_drugs_avg_overridden += 1

    print(f"\n  [Years to Entry / Avg override] Recomputed YTE for {n_rows_overridden} row(s) "
          f"and Avg YTE for {n_drugs_avg_overridden} drug(s) from LOE_Date.")
    return final_df
