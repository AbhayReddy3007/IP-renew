"""
config.py — All constants and schema definitions for the combine_master_loe pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── BigQuery defaults ─────────────────────────────────────────────────────────
# BQ_PROJECT / BQ_DATASET / D1_TABLE / D2_TABLE can be set in .env so you don't
# have to pass --project/--dataset/--target-table/--d2-table on every run.
# CLI flags, when given, still take priority over these (and over .env).
DEFAULT_BQ_PROJECT      = os.getenv("BQ_PROJECT", "cognito-prod-394707")
DEFAULT_BQ_DATASET      = os.getenv("BQ_DATASET", "cognito_prod_datamart")
DEFAULT_FORECAST_TABLE  = "forecasted_loe"
DEFAULT_MASTER_TABLE    = "loe_table_2"
DEFAULT_TARGET_TABLE    = os.getenv("D1_TABLE", "d1")
DEFAULT_PATENT_DISCOVERY_TABLE = "patent_discovery"
DEFAULT_D2_TABLE        = os.getenv("D2_TABLE", "d2")

# patent_master — the approved-patent shortlist. A patent row is only used
# for downstream computation (dedup, expiry sourcing, Years to Entry, d2)
# when patent_master marks it patent_status = 'PDF Downloaded' and
# confidence > 0.3 for that drug/molecule. Mirrors
# IP_refactored.chunking.patent_filter's approval criteria.
DEFAULT_PATENT_MASTER_TABLE = "patent_master"

# Service account JSON path — loaded from .env; --credentials on the CLI overrides this.
GCS_CREDENTIALS = os.getenv("GCS_CREDENTIALS")

# ── Clinical / drug-details source ───────────────────────────────────────────
DEFAULT_CLINICAL_PROJECT    = "cognito-dev-380506"
DEFAULT_CLINICAL_DATASET    = "data_mart"
DEFAULT_CLINICAL_TABLE      = "clinical_efficacy"
DEFAULT_DRUG_DETAILS_TABLE  = "vw_drug_details_full"

# ── Columns to drop from the final combined output ───────────────────────────
# "rationale" stays dropped because report1.py separately generates its own
# "Rationale" column into this table; keeping both would create a confusing
# naming collision.
DROP_COLUMNS = ["rationale"]

# ── Fields always recalculated every run (never carried over from source tables)
_ALWAYS_RECALCULATED_FIELDS = [
    "Phase",
    "Years to Entry",
    "Avg Years to Entry (US & EP)",
    "Controlling Patent Expiry Year",
    "IP Dimension 1 Score",
]
_TEXT_ALWAYS_RECALCULATED_FIELDS = {"Phase"}

# ── Final output schema ───────────────────────────────────────────────────────
# (friendly display name, list-of-alternative token-sets to match against
# normalised column names).  Order sets column order in the Excel/BQ output.
FINAL_COLUMNS = [
    ("Drug Name",                       [["drug"]]),
    ("Jurisdiction",                    [["jurisdiction"]]),
    ("Tag",                             [["tag"]]),
    ("Type",                            [["type"]]),
    ("Filing Date",                     [["filing", "date"]]),
    ("Patent Number",                   [["patent", "number"]]),
    ("No Of Forecasted Patents",        [["noofforecastedpatents"],
                                          ["numberofforecastedpatents"],
                                          ["forecastedpatents"],
                                          ["noofforecasted"]]),
    ("Years to Entry",                  [(["years", "entry"], ["avg"])]),
    ("Avg Years to Entry (US & EP)",    [["avgyears", "usep"], ["avgyears"]]),
    ("IP Dimension 1 Score",            [["ipdimension"]]),
    ("Step 1 Claim Category",           [["step1claimcategory"], ["step", "claim", "category"]]),
    ("Grant Date",                      [["grant", "date"]]),
    ("PTE (months)",                    [["pte"]]),
    ("Pediatric Exclusivity",           [["pediatric"]]),
    ("Controlling Patent Expiry Year",  [["controlling", "expiry"]]),
    ("Exclusivity Year",                [["exclusivity", "year"]]),
    ("Approval Date",                   [["approval", "date"]]),
    ("Est. Approval Year",              [["est", "approval", "year"]]),
    ("Phase",                           [["phase"]]),
    ("Strategic Purpose",               [["strategic", "purpose"]]),
    ("Risk Assessment",                 [["risk", "assessment"]]),
    ("Overall Forecast",                [["overall", "forecast"]]),
    ("Portfolio Gaps",                  [["portfolio", "gaps"]]),
    ("Existing Patent Summary",         [["existing", "patent", "summary"]]),
    ("Blocking Category",               [["blocking", "category"]]),
    ("Reason",                          [["reason"]]),
]

# ── Phase / clinical constants ────────────────────────────────────────────────
# (phase, phase_status) -> years to add to current_year for Est. Approval Year
PHASE_STATUS_TABLE = [
    ("approved/marketed", "-",                           0.0),
    ("pre-registration",  "-",                           1.0),
    ("phase 3",            "recruiting",                 3.0),
    ("phase 3",            "active, not recruiting",     2.0),
    ("phase 3",            "primary completion reached", 1.0),
    ("phase 3",            "completed",                  0.75),
    ("phase 2",            "recruiting",                 5.0),
    ("phase 2",            "active, not recruiting",     4.0),
    ("phase 2",            "primary completion reached", 3.0),
    ("phase 2",            "completed",                  2.5),
]

PHASE_DISPLAY = {
    "approved/marketed": "Approved/Marketed",
    "pre-registration":  "Pre-registration",
    "phase 3":           "Phase 3",
    "phase 2":           "Phase 2",
    "phase 1":           "Phase 1",
}

_PHASE_PRIORITY_ORDER  = ["approved/marketed", "pre-registration", "phase 3", "phase 2"]
_STATUS_PRIORITY_ORDER = [
    "completed",
    "primary completion reached",
    "active, not recruiting",
    "recruiting",
]

# ── Country-code aliases ──────────────────────────────────────────────────────
_COUNTRY_CODE_ALIASES = {
    "CN": "CN", "CHINA": "CN",
    "AU": "AU", "AUSTRALIA": "AU",
    "BR": "BR", "BRAZIL": "BR",
    "CA": "CA", "CANADA": "CA",
    "ES": "ES", "SPAIN": "ES",
    "IN": "IN", "INDIA": "IN",
    "JP": "JP", "JAPAN": "JP",
    "KR": "KR", "SOUTH KOREA": "KR",
    "MX": "MX", "MEXICO": "MX",
    "PL": "PL", "POLAND": "PL",
    "RU": "RU", "RUSSIA": "RU",
    "TW": "TW", "TAIWAN": "TW",
    "NL": "NL", "NETHERLANDS": "NL",
}

# ── Manual column aliases (forecasted_loe -> loe_table_2) ────────────────────
MANUAL_COLUMN_ALIASES = {
    "filingdatelower": "filingdate",
    "globalphase":     "phase",
}
