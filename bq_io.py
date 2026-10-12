"""
bq_io.py — BigQuery and local-file I/O helpers.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import _bq_safe_name, _sanitize_mixed_columns_for_bq


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_table_from_bq(project: str, dataset: str, table: str, label: str,
                         credentials_path: Optional[str] = None) -> pd.DataFrame:
    from google.cloud import bigquery

    client = _bq_client(project, credentials_path)
    fq_table = f"{client.project}.{dataset}.{table}"
    print(f"\n  [BQ] Downloading {label}: {fq_table} ...")
    try:
        df = client.query(f"SELECT * FROM `{fq_table}`").to_dataframe()
    except Exception as e:
        print(f"[ERROR] Failed to query {fq_table}: {e}")
        sys.exit(1)
    print(f"    Rows: {len(df)} | Columns: {len(df.columns)}")
    return df


def _load_table_from_file(path: str, label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p}"); sys.exit(1)
    print(f"\n  Loading {label}: {p}")
    df = pd.read_excel(p)
    print(f"    Rows: {len(df)} | Columns: {len(df.columns)}")
    return df


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload_to_bigquery(df: pd.DataFrame, project: str, dataset: str, table: str,
                         credentials_path: Optional[str],
                         write_disposition: str = "WRITE_TRUNCATE") -> None:
    from google.cloud import bigquery

    client = _bq_client(project, credentials_path)
    dataset_ref = bigquery.DatasetReference(client.project, dataset)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        print(f"  [BQ] Dataset {dataset!r} not found — creating it ...")
        client.create_dataset(bigquery.Dataset(dataset_ref))
        print(f"  [BQ] Dataset {dataset!r} created.")

    table_id = f"{client.project}.{dataset}.{table}"
    table_ref = bigquery.TableReference(dataset_ref, table)
    try:
        client.get_table(table_ref)
        print(f"  [BQ] Table {table_id} exists — loading per write_disposition={write_disposition!r}.")
    except Exception:
        print(f"  [BQ] Table {table_id} not found — it will be created automatically by this load.")

    bq_df = _prepare_for_bigquery(df)

    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=write_disposition,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )
    print(f"\n  [BQ] Uploading {len(bq_df)} row(s) / {len(bq_df.columns)} col(s) to {table_id} "
          f"(write_disposition={write_disposition}) ...")
    try:
        job = client.load_table_from_dataframe(bq_df, table_id, job_config=job_config)
        job.result()
    except Exception as e:
        print(f"[ERROR] Failed to upload to {table_id}: {e}")
        sys.exit(1)

    table_ref = client.get_table(table_id)
    print(f"  [BQ] Done — {table_id} now has {table_ref.num_rows} row(s) and "
          f"{len(table_ref.schema)} column(s).")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bq_client(project: str, credentials_path: Optional[str]):
    from google.cloud import bigquery
    if credentials_path:
        from google.oauth2 import service_account
        cred_path = Path(credentials_path)
        if not cred_path.exists():
            print(f"[ERROR] Service account JSON not found: {cred_path}"); sys.exit(1)
        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(cred_path),
                scopes=["https://www.googleapis.com/auth/bigquery",
                        "https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as e:
            print(f"[ERROR] Failed to load service account JSON {cred_path}: {e}")
            sys.exit(1)
        return bigquery.Client(project=project or credentials.project_id, credentials=credentials)
    return bigquery.Client(project=project)


def _prepare_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    """Rename friendly display columns to BQ-safe names and sanitise mixed types."""
    df = _sanitize_mixed_columns_for_bq(df)
    rename_map = {}
    seen = {}
    for c in df.columns:
        safe = _bq_safe_name(c)
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        rename_map[c] = safe
    renamed = df.rename(columns=rename_map)
    print("\n  Column names prepared for BigQuery:")
    for old, new in rename_map.items():
        marker = "" if old == new else "  <-- renamed"
        print(f"    {new!r:35s}{marker}")
    return renamed
