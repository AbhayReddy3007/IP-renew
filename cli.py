"""
cli.py — Argument parser for the combine_master_loe pipeline.
"""

import argparse
from config import (
    DEFAULT_BQ_PROJECT, DEFAULT_BQ_DATASET,
    DEFAULT_FORECAST_TABLE, DEFAULT_MASTER_TABLE,
    DEFAULT_TARGET_PROJECT, DEFAULT_TARGET_DATASET, DEFAULT_TARGET_TABLE,
    DEFAULT_CLINICAL_PROJECT, DEFAULT_CLINICAL_DATASET,
    DEFAULT_CLINICAL_TABLE, DEFAULT_DRUG_DETAILS_TABLE,
    DEFAULT_PATENT_PROJECT, DEFAULT_PATENT_DATASET,
    DEFAULT_D2_TABLE, DEFAULT_PATENT_MASTER_TABLE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download forecasted_loe + Master_LOE from BigQuery, combine, and export to Excel."
    )
    parser.add_argument("--project", default=DEFAULT_BQ_PROJECT,
                        help="BigQuery project ID")
    parser.add_argument("--dataset", default=DEFAULT_BQ_DATASET,
                        help="BigQuery dataset ID")
    parser.add_argument("--forecast-table", default=DEFAULT_FORECAST_TABLE,
                        help=f"forecasted_loe table name (default: {DEFAULT_FORECAST_TABLE})")
    parser.add_argument("--master-table", default=DEFAULT_MASTER_TABLE,
                        help=f"loe_table_2 table name (default: {DEFAULT_MASTER_TABLE})")
    parser.add_argument("--drug", default=None,
                        help="Only combine rows for this drug (case-insensitive). "
                             "You can also pass it as its own flag, e.g. `python main.py --Semaglutide`.")
    parser.add_argument("--file1", default=None,
                        help="Use a local forecasted_loe .xlsx instead of querying BigQuery")
    parser.add_argument("--file2", default=None,
                        help="Use a local Master_LOE .xlsx instead of querying BigQuery")
    parser.add_argument("--credentials", "--service-account", dest="credentials", default=None,
                        help="Path to a GCP service account JSON key file. If omitted, falls back "
                             "to GCS_CREDENTIALS from .env, then to Application Default Credentials.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file name (default: combined_master_loe.xlsx, or "
                             "combined_<drug>.xlsx when --drug is used)")
    parser.add_argument("--target-project", default=DEFAULT_TARGET_PROJECT,
                        help="BigQuery project for the uploaded d1/d2 output tables "
                             "(default: TARGET_PROJECT from .env if set, else same as --project)")
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET,
                        help="BigQuery dataset for the uploaded d1/d2 output tables "
                             "(default: TARGET_DATASET from .env if set, else same as --dataset)")
    parser.add_argument("--target-table", default=DEFAULT_TARGET_TABLE,
                        help=f"BigQuery table to upload the combined output to "
                             f"(default: {DEFAULT_TARGET_TABLE!r}). Created automatically if needed.")
    parser.add_argument("--write-mode", choices=["truncate", "append"], default="truncate",
                        help="'truncate' (default): replace all rows. 'append': add rows.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading to BigQuery; only write the local Excel file.")
    parser.add_argument("--clinical-project", default=DEFAULT_CLINICAL_PROJECT,
                        help=f"GCP project for clinical_efficacy (default: {DEFAULT_CLINICAL_PROJECT})")
    parser.add_argument("--clinical-dataset", default=DEFAULT_CLINICAL_DATASET,
                        help=f"Dataset for clinical_efficacy (default: {DEFAULT_CLINICAL_DATASET})")
    parser.add_argument("--clinical-table", default=DEFAULT_CLINICAL_TABLE,
                        help=f"Table name for clinical_efficacy (default: {DEFAULT_CLINICAL_TABLE})")
    parser.add_argument("--drug-details-table", default=DEFAULT_DRUG_DETAILS_TABLE,
                        help=f"Table name for vw_drug_details_full (default: {DEFAULT_DRUG_DETAILS_TABLE})")
    parser.add_argument("--no-clinical", action="store_true",
                        help="Skip the clinical_efficacy/vw_drug_details phase-overwrite step.")
    parser.add_argument("--no-marketed-approval", action="store_true",
                        help="Skip overriding Exclusivity Year/Years to Entry for Marketed/Pre-registration "
                             "rows with a real approval year.")
    parser.add_argument("--no-marketed-skip", action="store_true",
                        help="Disable the Table-1 (forecasted_loe) skip for Marketed/Pre-registration drugs.")
    parser.add_argument("--patent-project", default=DEFAULT_PATENT_PROJECT,
                        help=f"BigQuery project for patent_master — independent of both --project "
                             f"(source) and --target-project (d1/d2) (default: {DEFAULT_PATENT_PROJECT!r}).")
    parser.add_argument("--patent-dataset", default=DEFAULT_PATENT_DATASET,
                        help=f"BigQuery dataset for patent_master (default: {DEFAULT_PATENT_DATASET!r}).")
    parser.add_argument("--patent-master-table", default=DEFAULT_PATENT_MASTER_TABLE,
                        help=f"BigQuery table name for patent_master. This is the SOLE source for both "
                             f"the approved-patent shortlist (patent_status='PDF Downloaded', "
                             f"confidence>0.3) and Controlling Patent Expiry Year (via its expiry_date "
                             f"column) — patent_discovery is not used (default: {DEFAULT_PATENT_MASTER_TABLE!r}).")
    parser.add_argument("--no-patent-filter", action="store_true",
                        help="Skip the patent_master approval filter (step 5); use all patent rows "
                             "as-is instead of restricting to the approved shortlist. patent_master "
                             "is still used afterward to source Controlling Patent Expiry Year.")
    parser.add_argument("--d2-table", default=DEFAULT_D2_TABLE,
                        help=f"BigQuery table name to write the d2 output to (default: {DEFAULT_D2_TABLE!r}).")
    parser.add_argument("--no-d2", action="store_true",
                        help="Skip building the d2 table.")
    return parser
