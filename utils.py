"""
utils.py — Shared utility functions: string normalisation, column finding,
BigQuery-safe naming, and type-coercion helpers.
"""

import re
from typing import Optional

import pandas as pd
import numpy as np

from config import MANUAL_COLUMN_ALIASES, FINAL_COLUMNS, _COUNTRY_CODE_ALIASES


# ── String normalisation ──────────────────────────────────────────────────────

def _norm(col: str) -> str:
    """Collapse a column name to just letters/digits, lowercased."""
    return re.sub(r'[^a-z0-9]', '', col.lower())


def _norm_ws(s) -> str:
    """Lowercase, strip, and collapse whitespace for loose text matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


_ROMAN_PHASE_NUMERALS = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def _norm_phase(s) -> str:
    """Normalise any phase spelling to a canonical form (e.g. 'Phase 3',
    'Phase III', '3b', '2/3' -> 'phase 3'; 'Approved/Marketed' ->
    'approved/marketed'). Returns '' if blank/unrecognisable."""
    if isinstance(s, float):
        if pd.isna(s):
            return ""
        s = str(int(s)) if s == int(s) else str(s)
    elif isinstance(s, int):
        s = str(s)

    s = _norm_ws(s)
    if not s:
        return ""
    if "approv" in s or "market" in s:
        return "approved/marketed"
    if "pre" in s and "regist" in s:
        return "pre-registration"

    digit_parts = re.findall(r"\d+", s)
    if digit_parts:
        return f"phase {max(int(p) for p in digit_parts)}"

    for token in re.split(r"[^a-z0-9]+", s):
        if token in _ROMAN_PHASE_NUMERALS:
            return f"phase {_ROMAN_PHASE_NUMERALS[token]}"

    compact  = re.sub(r"[^a-z0-9]", "", s)
    stripped = re.sub(r"^(phase|ph|p)", "", compact)
    if stripped in _ROMAN_PHASE_NUMERALS:
        return f"phase {_ROMAN_PHASE_NUMERALS[stripped]}"
    return s


def _normalize_drug_str(s) -> str:
    """Lowercase, strip, and collapse whitespace around + / & so combination
    drug names normalise consistently regardless of spacing."""
    if s is None or s is pd.NA or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s*([+/&])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_blank(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip().upper() in ("", "N/A", "NULL", "NONE", "NAN")


# ── Column finding ────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, required_tokens, prefer_tokens=None) -> Optional[str]:
    """Find a column whose normalised name contains all required_tokens.
    If prefer_tokens is given, prefer a match that also contains those."""
    candidates = [c for c in df.columns if all(t in _norm(c) for t in required_tokens)]
    if not candidates:
        return None
    if prefer_tokens:
        preferred = [c for c in candidates if all(t in _norm(c) for t in prefer_tokens)]
        if preferred:
            return preferred[0]
    return candidates[0]


def _ensure_column(df: pd.DataFrame, canonical_name: str, required_tokens,
                    exclude_tokens=None, label: str = "") -> pd.DataFrame:
    """Guarantee canonical_name exists in df, renaming from a fuzzy match
    or adding a blank column (with a warning) if nothing matches."""
    exclude_tokens = exclude_tokens or []
    alternatives = (required_tokens
                    if (required_tokens and isinstance(required_tokens[0], (list, tuple)))
                    else [required_tokens])

    match = None
    for alt in alternatives:
        candidates = [
            c for c in df.columns
            if all(t in _norm(c) for t in alt) and not any(t in _norm(c) for t in exclude_tokens)
        ]
        if candidates:
            match = candidates[0]
            break

    if match:
        if match != canonical_name:
            df = df.rename(columns={match: canonical_name})
            print(f"  Mapped {match!r} -> {canonical_name!r}{f' ({label})' if label else ''}")
    else:
        print(f"  [WARN] No column resembling {canonical_name!r} found"
              f"{f' in {label}' if label else ''} — adding blank column.")
        df[canonical_name] = pd.NA
    return df


# ── Column alignment ──────────────────────────────────────────────────────────

def _apply_manual_aliases(df_source: pd.DataFrame, df_reference: pd.DataFrame) -> pd.DataFrame:
    """Rename df_source columns per MANUAL_COLUMN_ALIASES wherever the
    target normalised name exists in df_reference."""
    ref_by_norm = {_norm(c): c for c in df_reference.columns}
    rename_map = {}
    for c in df_source.columns:
        target_norm = MANUAL_COLUMN_ALIASES.get(_norm(c))
        if target_norm and target_norm in ref_by_norm:
            rename_map[c] = ref_by_norm[target_norm]
    if rename_map:
        print(f"\n  Applied {len(rename_map)} manual column alias(es):")
        for old, new in rename_map.items():
            print(f"    {old!r}  ->  {new!r}")
    return df_source.rename(columns=rename_map)


def _align_columns(df_source: pd.DataFrame, df_reference: pd.DataFrame) -> pd.DataFrame:
    """Rename columns in df_source to match df_reference's spelling wherever
    they represent the same field. Columns unique to df_source are left as-is."""
    df_source = _apply_manual_aliases(df_source, df_reference)
    ref_by_norm = {_norm(c): c for c in df_reference.columns}
    rename_map = {
        c: ref_by_norm[_norm(c)]
        for c in df_source.columns
        if _norm(c) in ref_by_norm and c != ref_by_norm[_norm(c)]
    }
    if rename_map:
        print(f"\n  Aligned {len(rename_map)} column name(s) from file 1 to match file 2:")
        for old, new in rename_map.items():
            print(f"    {old!r}  ->  {new!r}")
    return df_source.rename(columns=rename_map)


def _select_final_columns(df: pd.DataFrame):
    """Build the final output dataframe with exactly the friendly-named
    columns in FINAL_COLUMNS, in that order. Returns (result_df, used_set)."""
    used = set()
    data = {}
    for friendly, alternatives in FINAL_COLUMNS:
        match = None
        for alt in alternatives:
            if isinstance(alt, tuple):
                required, exclude = alt
            else:
                required, exclude = alt, []
            candidates = [c for c in df.columns if c not in used
                          and all(t in _norm(c) for t in required)
                          and not any(t in _norm(c) for t in exclude)]
            if candidates:
                match = candidates[0]
                break
        if match:
            data[friendly] = df[match].values
            used.add(match)
            print(f"  {match!r:35s} -> {friendly!r}")
        else:
            print(f"  {'(none found)':35s} -> {friendly!r}  [WARN] filling blank")
            data[friendly] = [pd.NA] * len(df)
    return pd.DataFrame(data, index=df.index), used


# ── Jurisdiction helpers ──────────────────────────────────────────────────────

def _jurisdiction_token(jurisdiction: str):
    """Map a Jurisdiction value to ('US', 5), ('EU', 10), ('<CODE>', 0), or None."""
    j = _norm_ws(jurisdiction).upper().strip()
    if not j:
        return None
    if j in ("US", "USA", "U.S.", "U.S.A.", "UNITED STATES", "UNITED STATES OF AMERICA"):
        return ("US", 5)
    if j in ("EU", "EP", "EUROPE", "EUROPEAN UNION"):
        return ("EU", 10)
    if j in _COUNTRY_CODE_ALIASES:
        return (_COUNTRY_CODE_ALIASES[j], 0)
    return None


def _vwd_geo_tokens(geo) -> set:
    """Split a Drug_Geo_New value into a set of jurisdiction tokens."""
    if geo is None or (isinstance(geo, float) and pd.isna(geo)):
        return set()
    raw = str(geo).strip().strip("'\"").strip()
    if raw.upper() == "GLOBAL":
        return {"US", "EU"} | set(_COUNTRY_CODE_ALIASES.values())
    tokens = set()
    for part in re.split(r"[,;]+", raw):
        p = re.sub(r"\s+", " ", part.strip().strip("'\"").strip()).upper()
        if not p:
            continue
        if p in ("US", "USA", "U.S.", "U.S.A.", "UNITED STATES", "UNITED STATES OF AMERICA",
                 "UNITES STATES"):
            tokens.add("US")
        elif p in ("EU", "EP", "EUROPE", "EUROPEAN UNION"):
            tokens.add("EU")
        elif p in _COUNTRY_CODE_ALIASES:
            tokens.add(_COUNTRY_CODE_ALIASES[p])
        elif p == "GLOBAL":
            tokens |= {"US", "EU"} | set(_COUNTRY_CODE_ALIASES.values())
        else:
            tokens.add(p)
    return tokens


def _vwd_geo_tokens_combined(geo_new, geo_fallback) -> set:
    """Prefer Drug_Geo_New tokens; fall back to Drug_Geography if geo_new is blank."""
    tokens = _vwd_geo_tokens(geo_new)
    return tokens if tokens else _vwd_geo_tokens(geo_fallback)


def _location_tokens(location) -> set:
    """Split a clinical_efficacy trial_location value into jurisdiction tokens."""
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return set()
    raw = str(location).strip().strip("'\"").strip()
    if raw.upper() == "GLOBAL":
        return {"US", "EU"} | set(_COUNTRY_CODE_ALIASES.values())
    tokens = set()
    for part in re.split(r"[/,;]+", raw):
        p = re.sub(r"\s+", " ", part.strip().strip("'\"").strip()).upper()
        if not p:
            continue
        if p in ("US", "USA", "U.S.", "U.S.A.", "UNITED STATES", "UNITED STATES OF AMERICA",
                 "UNITES STATES"):
            tokens.add("US")
        elif p in ("EU", "EP", "EUROPE", "EUROPEAN UNION"):
            tokens.add("EU")
        elif p in _COUNTRY_CODE_ALIASES:
            tokens.add(_COUNTRY_CODE_ALIASES[p])
        elif p == "GLOBAL":
            tokens |= {"US", "EU"} | set(_COUNTRY_CODE_ALIASES.values())
        else:
            tokens.add(p)
    return tokens


# ── Patent-number normalisation ───────────────────────────────────────────────

def _normalize_patent_number(pn) -> str:
    """Normalise a patent number: strip non-alphanumeric chars, uppercase,
    strip trailing kind-code suffix (e.g. A1, B2, E, E1)."""
    if pn is None or (isinstance(pn, float) and pd.isna(pn)):
        return ""
    s = re.sub(r'[^A-Za-z0-9]', '', str(pn)).upper()
    s = re.sub(r'[A-Z]\d{0,2}$', '', s)
    return s


# ── Date helpers ──────────────────────────────────────────────────────────────

def _filing_year(val) -> Optional[float]:
    """Extract a 4-digit filing year from a Filing Date value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return float(val.year)
    m = re.search(r'(\d{4})', str(val))
    return float(m.group(1)) if m else None


def _approval_year_from_value(val) -> Optional[float]:
    """Extract a calendar year from an Approval Date / First_Approval_Date cell."""
    from datetime import datetime
    if _is_blank(val):
        return None
    if isinstance(val, pd.Timestamp):
        return float(val.year)
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y",
                "%B %d, %Y", "%B %d %Y", "%d %B %Y"):
        try:
            return float(datetime.strptime(s, fmt).year)
        except ValueError:
            continue
    m = re.search(r"(\d{4})", s)
    return float(m.group(1)) if m else None


# ── BQ naming / type helpers ──────────────────────────────────────────────────

def _bq_safe_name(name: str) -> str:
    """'PTE (months)' -> 'PTE_months', 'Est. Approval Year' -> 'Est_Approval_Year'."""
    s = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"_{s}"
    return s


def _stringify_cell(v):
    """Convert a single cell to a plain string (or None) for pyarrow upload."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    s = str(v).strip()
    return None if s == "" or s.lower() in ("nan", "nat", "none") else s


def _sanitize_mixed_columns_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    """Force mixed-type columns (date-ish or genuinely mixed) down to a
    consistent string type so pyarrow can infer one type for upload."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        looks_like_date = "date" in col.lower()
        non_null = df[col].dropna()
        mixed_types = non_null.map(type).nunique() > 1 if not non_null.empty else False
        if looks_like_date or mixed_types:
            df[col] = df[col].apply(_stringify_cell)
    return df


def _ip_dimension_score(avg_years: float) -> int:
    """Bucket avg_years into an IP Dimension 1 Score (1–5)."""
    if avg_years <= 6:
        return 5
    if avg_years <= 8:
        return 4
    if avg_years <= 11:
        return 3
    if avg_years <= 13:
        return 2
    return 1


def _filter_to_drug(df: pd.DataFrame, drug: str, label: str) -> pd.DataFrame:
    """Keep only rows whose drug-name column exactly matches `drug` (case-insensitive)."""
    drug_col = _find_col(df, ["drug"])
    if not drug_col:
        print(f"  [WARN] No drug column found in {label}; cannot filter.")
        return df
    normalized_col = df[drug_col].astype(str).map(_normalize_drug_str)
    target = _normalize_drug_str(drug)
    mask = normalized_col == target
    filtered = df[mask]
    print(f"  {label}: {len(filtered)}/{len(df)} row(s) match drug={drug!r} (column {drug_col!r})")
    return filtered
