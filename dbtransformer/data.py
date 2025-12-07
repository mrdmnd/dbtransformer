"""
Data loading utilities for DuckDB datasets hosted on S3-compatible storage.

Supports AWS S3, Cloudflare R2, MinIO, etc. R2 is recommended for
multi-cloud GPU training due to zero egress fees.

Environment variables:
    DATA_BUCKET: S3 bucket name (required)
    DATA_PREFIX: Prefix/folder in bucket (default: "")
    DATA_CACHE: Local cache directory (default: ~/.cache/dbtransformer)
    AWS_ENDPOINT_URL: S3-compatible endpoint (required for R2/MinIO)
    AWS_ACCESS_KEY_ID: Access key
    AWS_SECRET_ACCESS_KEY: Secret key

Usage:
    from dbtransformer.data import ensure_data, list_tables, load_table

    # Download dataset to local cache (skips if already cached)
    data_path = ensure_data(version="v1")

    # List available DuckDB files
    tables = list_tables(data_path)

    # Load a specific table
    df = load_table(data_path / "customers.duckdb", "customers")
"""

import os
from pathlib import Path

import boto3
import duckdb
from botocore.config import Config
from loguru import logger


def _get_s3_client() -> boto3.client:
    """Create S3 client, using custom endpoint if specified (for R2/MinIO)."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def ensure_data(
    version: str = "v1",
    bucket: str | None = None,
    prefix: str | None = None,
    cache_dir: str | None = None,
    force: bool = False,
) -> Path:
    """
    Download dataset from S3/R2 if not cached, return local path.

    Args:
        version: Dataset version (used as subfolder in bucket and cache)
        bucket: S3 bucket name (defaults to DATA_BUCKET env var)
        prefix: Prefix in bucket (defaults to DATA_PREFIX env var)
        cache_dir: Local cache directory (defaults to DATA_CACHE env var)
        force: Force re-download even if cached

    Returns:
        Path to the local dataset directory
    """
    bucket = bucket or os.environ.get("DATA_BUCKET")
    if not bucket:
        raise ValueError(
            "No bucket specified. Set DATA_BUCKET env var or pass bucket arg."
        )

    prefix = prefix or os.environ.get("DATA_PREFIX", "")
    cache_dir = cache_dir or os.environ.get(
        "DATA_CACHE",
        os.path.expanduser("~/.cache/dbtransformer"),
    )

    # Build paths
    s3_prefix = f"{prefix}/{version}".strip("/")
    local_path = Path(cache_dir) / version
    marker = local_path / ".downloaded"

    # Skip if already cached
    if marker.exists() and not force:
        logger.info(f"Using cached dataset at {local_path}")
        return local_path

    logger.info(f"Downloading s3://{bucket}/{s3_prefix} to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)

    # Download all files from S3
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    file_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Get relative path from the version prefix
            rel_path = key[len(s3_prefix):].lstrip("/")
            if not rel_path:
                continue

            local_file = local_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Downloading {key} -> {local_file}")
            s3.download_file(bucket, key, str(local_file))
            file_count += 1

    marker.touch()
    logger.info(f"Downloaded {file_count} files to {local_path}")
    return local_path


def list_tables(data_path: Path) -> list[Path]:
    """
    List all DuckDB files in the dataset directory.

    Args:
        data_path: Path to the dataset directory

    Returns:
        List of paths to DuckDB files
    """
    duckdb_files = list(data_path.rglob("*.duckdb"))
    logger.info(f"Found {len(duckdb_files)} DuckDB files in {data_path}")
    return sorted(duckdb_files)


def load_table(
    db_path: Path,
    table_name: str,
    query: str | None = None,
) -> duckdb.DuckDBPyRelation:
    """
    Load a table from a DuckDB file.

    Args:
        db_path: Path to the DuckDB file
        table_name: Name of the table to load
        query: Optional SQL query (defaults to SELECT * FROM table_name)

    Returns:
        DuckDB relation (lazy, can be converted to pandas/arrow/etc)
    """
    conn = duckdb.connect(str(db_path), read_only=True)

    if query is None:
        query = f"SELECT * FROM {table_name}"

    return conn.sql(query)


def get_table_info(db_path: Path) -> list[dict[str, str | int]]:
    """
    Get information about all tables in a DuckDB file.

    Args:
        db_path: Path to the DuckDB file

    Returns:
        List of dicts with table name and row count
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    tables = conn.sql("SHOW TABLES").fetchall()

    info = []
    for (table_name,) in tables:
        count = conn.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        info.append({"name": table_name, "rows": count})

    return info
