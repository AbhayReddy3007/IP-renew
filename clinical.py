"""
clinical.py — Phase resolution from clinical_efficacy + vw_drug_details_full.

Overwrites Phase / Est. Approval Year / Exclusivity Year / Years to Entry
for every row where real trial data exists.  Runs in two explicit passes so
the Phase column is fully settled before any derived calculations are written.
"""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from config import (
    PHASE_STATUS_TABLE, PHASE_DISPLAY,
    _PHASE_PRIORITY_ORDER, _STATUS_PRIORITY_ORDER,
)
from utils import (
    _find_col, _norm, _norm_ws, _norm_phase,
    _normalize_drug_str, _jurisdiction_token,
    _location_tokens, _vwd_geo_tokens, _vwd_geo_tokens_combined,
)


def _fallback_x_for_phase(phase: str) -> Optional[float]:
    return {
        "approved/marketed": 0.0,
        "pre-registration":  1.0,
        "phase 3":           3.0,
        "phase 2":           5.0,
    }.get(phase)


def _pick_best_phase_multi(clinical_rows, clin_phase_col, vwd_rows, vwd_phase_col):
    clin_phases: set = set()
    if clinical_rows is not None and not clinical_rows.empty and clin_phase_col:
        clin_phases = {_norm_phase(p) for p in clinical_rows[clin_phase_col].dropna().unique()}
        clin_phases = {p for p in clin_phases if p in _PHASE_PRIORITY_ORDER}

    vwd_phases: set = set()
    if vwd_rows is not None and not vwd_rows.empty and vwd_phase_col:
        vwd_phases = {_norm_phase(p) for p in vwd_rows[vwd_phase_col].dropna().unique()}
        vwd_phases = {p for p in vwd_phases if p in _PHASE_PRIORITY_ORDER}

    all_phases = clin_phases | vwd_phases
    if not all_phases:
        return None, None

    for phase in _PHASE_PRIORITY_ORDER:
        if phase in all_phases:
            source = "clinical" if phase in clin_phases else "vwd"
            return phase, source
    return None, None


def _pick_best_status_and_x(rows: pd.DataFrame, phase_col: str, status_col: str, phase: str):
    for tbl_phase, tbl_status, x in PHASE_STATUS_TABLE:
        if tbl_phase == phase and tbl_status == "-":
            return x

    present_statuses = {_norm_ws(s) for s in rows[status_col].dropna().unique()} if status_col in rows else set()
    if not present_statuses:
        return None

    best_status = None
    for status in _STATUS_PRIORITY_ORDER:
        if status in present_statuses:
            best_status = status
            break
    if best_status is None:
        return None

    for tbl_phase, tbl_status, x in PHASE_STATUS_TABLE:
        if tbl_phase == phase and tbl_status == best_status:
            return x
    return None


def _apply_clinical_efficacy(final_df: pd.DataFrame, clinical_df: pd.DataFrame,
                              drug_details_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Overwrite Phase / Est. Approval Year / Exclusivity Year / Years to Entry
    on every row backed by real data from clinical_efficacy or vw_drug_details."""
    current_year = datetime.now().year

    molecule_col = _find_col(clinical_df, ["molecule", "name"]) or _find_col(clinical_df, ["molecule"])
    location_col = _find_col(clinical_df, ["trial", "location"]) or _find_col(clinical_df, ["location"])
    status_col   = _find_col(clinical_df, ["phase", "status"]) or _find_col(clinical_df, ["status"])
    phase_candidates = [c for c in clinical_df.columns if "phase" in _norm(c) and "status" not in _norm(c)]
    phase_col = phase_candidates[0] if phase_candidates else None

    missing = [n for n, c in (("molecule_name", molecule_col), ("trial_location", location_col),
                               ("phase", phase_col), ("phase_status", status_col)) if not c]
    if missing:
        print(f"  [WARN] clinical_efficacy missing expected column(s): {missing} — skipping.")
        return final_df

    clinical_df = clinical_df.copy()
    clinical_df["_molecule_norm"] = clinical_df[molecule_col].astype(str).map(_normalize_drug_str)

    vwd_name_col = vwd_geo_col = vwd_geo_fallback_col = vwd_phase_col = None
    if drug_details_df is not None and not drug_details_df.empty:
        vwd_name_col = (_find_col(drug_details_df, ["cleaned", "generic", "name"])
                         or _find_col(drug_details_df, ["generic", "name"])
                         or _find_col(drug_details_df, ["drug", "name"])
                         or _find_col(drug_details_df, ["drugname"]))
        vwd_geo_col = None
        for _c in drug_details_df.columns:
            if _c.strip().lower() == "drug_geo_new":
                vwd_geo_col = _c
                break
        if not vwd_geo_col:
            vwd_geo_col = (_find_col(drug_details_df, ["drug", "geo", "new"])
                           or _find_col(drug_details_df, ["drug", "geo"])
                           or _find_col(drug_details_df, ["geo"]))
        vwd_geo_fallback_col = _find_col(drug_details_df, ["drug", "geography"])
        vwd_phase_col = (_find_col(drug_details_df, ["highest", "development", "stage"])
                          or _find_col(drug_details_df, ["development", "stage"]))
        vwd_missing = [n for n, c in (("Cleaned_Generic_Name", vwd_name_col),
                                       ("Drug_Geo_New", vwd_geo_col),
                                       ("Highest_Development_Stage", vwd_phase_col)) if not c]
        if vwd_missing:
            print(f"  [WARN] vw_drug_details missing expected column(s): {vwd_missing} — clinical only.")
            drug_details_df = None
        else:
            print(f"  vw_drug_details columns: name={vwd_name_col!r}, geo={vwd_geo_col!r} "
                  f"(fallback={vwd_geo_fallback_col!r}), phase={vwd_phase_col!r}")
            drug_details_df = drug_details_df.copy()
            drug_details_df["_drug_norm"] = drug_details_df[vwd_name_col].astype(str).map(_normalize_drug_str)
            if vwd_geo_fallback_col:
                drug_details_df["_geo_tokens"] = drug_details_df.apply(
                    lambda r: _vwd_geo_tokens_combined(r[vwd_geo_col], r[vwd_geo_fallback_col]), axis=1)
            else:
                drug_details_df["_geo_tokens"] = drug_details_df[vwd_geo_col].map(_vwd_geo_tokens)

    drug_col, jurisdiction_col = "Drug Name", "Jurisdiction"
    if drug_col not in final_df.columns or jurisdiction_col not in final_df.columns:
        print("  [WARN] final_df missing Drug Name / Jurisdiction — skipping clinical_efficacy step.")
        return final_df

    # ── Pass 1: resolve Phase ─────────────────────────────────────────────────
    print("\n  [clinical_efficacy + vw_drug_details] Pass 1: overwriting Phase …")
    _resolved: Dict[int, tuple] = {}
    n_no_drug_data = n_no_phase = n_from_vwd = 0

    for idx, row in final_df.iterrows():
        drug_norm = _normalize_drug_str(str(row[drug_col])) if pd.notna(row[drug_col]) else ""
        if not drug_norm:
            continue
        jt = _jurisdiction_token(row[jurisdiction_col])
        if jt is None:
            continue
        token, excl_years = jt

        clin_drug_rows = clinical_df[clinical_df["_molecule_norm"] == drug_norm]
        clin_matched = (clin_drug_rows[clin_drug_rows[location_col].apply(
            lambda v: token in _location_tokens(v))]
            if not clin_drug_rows.empty else clin_drug_rows)

        if drug_details_df is not None:
            vwd_drug_rows = drug_details_df[drug_details_df["_drug_norm"] == drug_norm]
            vwd_matched = (vwd_drug_rows[vwd_drug_rows["_geo_tokens"].apply(lambda toks: token in toks)]
                           if not vwd_drug_rows.empty else vwd_drug_rows)
        else:
            vwd_matched = None

        no_trial_data = clin_matched.empty and (vwd_matched is None or vwd_matched.empty)
        if no_trial_data:
            n_no_drug_data += 1
            best_phase, source = None, None
        else:
            best_phase, source = _pick_best_phase_multi(clin_matched, phase_col, vwd_matched, vwd_phase_col)
            if source == "vwd":
                n_from_vwd += 1

        if best_phase is None:
            n_no_phase += 1
            continue

        final_df.at[idx, "Phase"] = PHASE_DISPLAY.get(best_phase, best_phase)
        _resolved[idx] = (best_phase, source, clin_matched, excl_years)

    print(f"    Phase overwritten for {len(_resolved)} row(s). "
          f"No trial data: {n_no_drug_data}. No recognised phase: {n_no_phase}.")

    # ── Pass 2: compute derived fields ────────────────────────────────────────
    print("  [clinical_efficacy + vw_drug_details] Pass 2: computing derived fields …")
    n_matched = n_no_status = 0

    for idx, (best_phase, source, clin_matched, excl_years) in _resolved.items():
        row = final_df.loc[idx]
        x = None
        if not clin_matched.empty:
            phase_rows = clin_matched[clin_matched[phase_col].map(_norm_phase) == best_phase]
            if not phase_rows.empty:
                x = _pick_best_status_and_x(phase_rows, phase_col, status_col, best_phase)
        if x is None:
            x = _fallback_x_for_phase(best_phase)
        if x is None:
            n_no_status += 1
            continue

        est_approval_year = current_year + x
        exclusivity_year  = est_approval_year + excl_years
        final_df.at[idx, "Est. Approval Year"] = est_approval_year
        final_df.at[idx, "Exclusivity Year"]   = exclusivity_year

        expiry_val = pd.to_numeric(pd.Series([row.get("Controlling Patent Expiry Year")]), errors="coerce").iloc[0]
        effective_expiry = max(expiry_val, exclusivity_year) if pd.notna(expiry_val) else exclusivity_year
        final_df.at[idx, "Years to Entry"] = round(effective_expiry - current_year, 1)
        n_matched += 1

    print(f"\n  [clinical_efficacy + vw_drug_details] Recomputed {n_matched} row(s) "
          f"({n_from_vwd} winning phase from vw_drug_details). "
          f"No trial data — untouched: {n_no_drug_data}. "
          f"No recognised phase — untouched: {n_no_phase}. "
          f"No resolvable status/fallback: {n_no_status}.")
    return final_df
