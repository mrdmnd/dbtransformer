"""
Download CTU Relational and RelBench datasets into RelBench format and cache them.

Usage:
    uv run python scripts/download_relbench.py

Note: RelBench datasets (rel-*) are downloaded from relbench.stanford.edu.
      CTU datasets (ctu-*) are fetched directly from the CTU database server
      at relational.fel.cvut.cz.
"""

# suppress import ordering here; we need to import redelex after relbench.datasets
# ruff: noqa: I001
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

# importing redelex calls register_dataset() on the ctu datasets.
from relbench.datasets import get_dataset, get_dataset_names
from relbench.base import Database, Table
import redelex  # noqa: F401
from redelex.datasets.db_dataset import (
    DBDataset,
    SQL_TO_PANDAS,
    DATE_TYPES,
    DATE_MAP,
)
from redelex.db import DBInspector
from tqdm import tqdm as tqdm_bar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Monkey patch for redelex.datasets.db_dataset.DBDataset
# Adds retry logic at the per-table level to handle connection timeouts
# ---------------------------------------------------------------------------

def _download_table_with_retry(
    remote_url: str,
    sql_table: sa.Table,
    dtypes: dict[str, str],
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> pd.DataFrame:
    """Download a single table with retry logic and fresh connections.

    Creates a fresh connection for each attempt, which helps when the
    connection has been closed by the server due to timeout.

    Uses NullPool to avoid connection pooling issues - each connection
    is fully closed when done, preventing "MySQL server has gone away"
    errors during cleanup.
    """
    delay = base_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        engine = None
        con = None
        try:
            # Create fresh engine with NullPool - no connection pooling
            # This ensures each connection is fully closed when done,
            # avoiding cleanup errors on stale pooled connections
            engine = sa.create_engine(
                remote_url,
                poolclass=NullPool,  # Don't pool connections
            )
            con = engine.connect()

            statement = sa.select(sql_table.columns)
            query = statement.compile(engine)
            df = pd.read_sql_query(str(query), con=con, dtype=dtypes)
            return df

        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            is_connection_error = any(
                x in err_str for x in [
                    "connection", "lost", "timeout", "gone away",
                    "broken pipe", "reset by peer", "operationalerror",
                ]
            )

            if attempt < max_retries and is_connection_error:
                print(
                    f"\n    Connection error downloading table "
                    f"'{sql_table.name}' (attempt {attempt + 1}/"
                    f"{max_retries + 1}): {e}"
                )
                print(f"    Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
            else:
                raise

        finally:
            # Clean up connection and engine, ignoring errors on dead connections
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass

    raise last_exception  # type: ignore[misc]


def _create_connection_with_null_pool(remote_url: str) -> sa.Connection:
    """Create a connection with NullPool to avoid cleanup issues."""
    engine = sa.create_engine(remote_url, poolclass=NullPool)
    return engine.connect()


def _patched_make_db(self) -> Database:
    """
    Patched version of DBDataset.make_db with per-table retry logic.

    This replaces the original make_db to handle connection timeouts
    that occur when downloading large databases from remote servers.
    """
    # Create initial connection for metadata inspection
    # Use NullPool to avoid "MySQL server has gone away" errors during cleanup
    remote_con = _create_connection_with_null_pool(self.remote_url)

    inspector = DBInspector(remote_con)

    remote_md = sa.MetaData()
    remote_md.reflect(bind=inspector.engine)

    table_names = inspector.get_tables()

    df_dict: dict[str, pd.DataFrame] = {}
    fk_dict: dict[str, list] = {}

    for t_name in tqdm_bar(table_names, desc="Downloading tables"):
        sql_table = sa.Table(t_name, remote_md)

        dtypes: dict[str, str] = {}
        sql_types_dict: dict[str, sa.types.TypeEngine] = {}

        for c in sql_table.columns:
            try:
                sql_type = type(c.type.as_generic())
            except NotImplementedError:
                sql_type = None

            dtype = SQL_TO_PANDAS.get(sql_type, None)

            if dtype is None:
                # Special case for YEAR type
                if c.type.__str__() == "YEAR":
                    dtype = pd.Int32Dtype()
                    sql_type = sa.types.Integer

            if dtype is not None:
                dtypes[c.name] = dtype
                sql_types_dict[c.name] = sql_type
            else:
                print(f"Unknown data type {c.type} in {t_name}.{c.name}")

        # Use retry-enabled table download instead of single query
        df = _download_table_with_retry(
            self.remote_url,
            sql_table,
            dtypes,
            max_retries=3,
            base_delay=5.0,
        )

        for col, sql_type in sql_types_dict.items():
            time_col = self.time_col_dict.get(t_name, None)
            if sql_type in DATE_TYPES or time_col == col:
                try:
                    df[col] = pd.to_datetime(df[col])
                except pd.errors.OutOfBoundsDatetime:
                    print(f"Out of bounds datetime in {t_name}.{col}")
                except Exception as e:
                    print(
                        f"Error converting {t_name}.{col} to datetime: {e}"
                    )

            if DATE_MAP.get(sql_type, None) is not None:
                try:
                    df[col] = df[col].astype(
                        DATE_MAP[sql_type], errors="raise"
                    )
                except pd.errors.OutOfBoundsDatetime:
                    print(f"Out of bounds datetime in {t_name}.{col}")
                except Exception as e:
                    print(
                        f"Error converting {t_name}.{col} to datetime: {e}"
                    )

        # Create index column used as artificial primary key
        df.index.name = "__PK__"
        df.reset_index(inplace=True)

        df_dict[t_name] = df

        # Refresh connection for foreign key lookup if needed
        try:
            fk_dict[t_name] = inspector.get_foreign_keys(t_name)
        except Exception:
            # Reconnect and retry once - close old connection gracefully
            try:
                remote_con.close()
            except Exception:
                pass
            remote_con = _create_connection_with_null_pool(self.remote_url)
            inspector = DBInspector(remote_con)
            fk_dict[t_name] = inspector.get_foreign_keys(t_name)

    table_dict: dict[str, Table] = {}

    # Re-index keys as RelBench do not support composite keys.
    for t_name in table_names:
        fkey_col_to_pkey_table: dict[str, str] = {}

        for fk in fk_dict[t_name]:
            fk_col, fk_name = self._reindex_fk(
                df_dict, t_name, fk.src_columns, fk.ref_table, fk.ref_columns
            )

            fkey_col_to_pkey_table[fk_name] = fk.ref_table
            df_dict[t_name][fk_name] = fk_col

        table_dict[t_name] = Table(
            df=df_dict[t_name],
            fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            pkey_col="__PK__",
            time_col=self.time_col_dict.get(t_name, None),
        )

    db = Database(table_dict)

    # Custom modifications hook
    try:
        db = self.customize_db(db)
    except NotImplementedError:
        pass

    # Remove original primary and foreign keys if configured
    if not self.keep_original_keys:
        for t_name in table_names:
            if t_name not in db.table_dict:
                continue

            sql_table = sa.Table(t_name, remote_md)
            table = db.table_dict[t_name]
            drop_cols = set()

            # Drop primary key columns
            keep_compound = self.keep_original_compound_keys
            if not keep_compound or len(sql_table.primary_key.columns) == 1:
                drop_cols |= {c.name for c in sql_table.primary_key.columns}

            for fk in sql_table.foreign_key_constraints:
                if fk.referred_table not in db.table_dict:
                    continue

                if not keep_compound or len(fk.columns) == 1:
                    drop_cols |= {c.name for c in fk.columns}

            table.df.drop(columns=drop_cols, inplace=True)

    # Close connection gracefully - ignore errors on dead connections
    try:
        remote_con.close()
    except Exception:
        pass

    return db


def apply_db_dataset_patch():
    """Apply the monkey patch to DBDataset.make_db for retry support."""
    DBDataset.make_db = _patched_make_db
    print("Applied DBDataset monkey patch for connection retry support.")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for the delay after each retry.
        exceptions: Tuple of exception types to catch and retry.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Check for connection-related errors
                        err_str = str(e).lower()
                        is_connection_error = any(
                            x in err_str for x in [
                                "connection",
                                "lost",
                                "timeout",
                                "gone away",
                                "broken pipe",
                                "reset by peer",
                            ]
                        )
                        if is_connection_error:
                            print(
                                f"\n  Connection error on attempt {attempt + 1}/"
                                f"{max_retries + 1}: {e}"
                            )
                            print(f"  Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            delay = min(delay * backoff_factor, max_delay)
                        else:
                            # Non-connection error, re-raise immediately
                            raise

            # All retries exhausted
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


def load_dataset(dataset_name: str) -> str:
    """Load a dataset and return its name on completion."""
    dataset = get_dataset(dataset_name, download=False)
    _ = dataset.get_db()
    return dataset_name


@retry_with_backoff(
    max_retries=3,
    base_delay=10.0,
    max_delay=120.0,
    backoff_factor=2.0,
)
def download_dataset(dataset_name: str) -> str:
    """Download a single dataset and return its name on completion.

    RelBench datasets (rel-*) use download=True to fetch pre-packaged files.
    CTU datasets (ctu-*) fetch directly from the CTU database server, so
    download=False is used (they handle their own data fetching).

    This function includes retry logic for transient connection failures
    that can occur with large CTU databases.
    """
    # CTU datasets connect directly to the CTU MariaDB server and don't
    # have pre-packaged downloads on the relbench server.
    should_download = not dataset_name.startswith("ctu-")
    dataset = get_dataset(dataset_name, download=should_download)
    # Force the database to be created/cached by accessing it
    _ = dataset.get_db()
    return dataset_name


def download_datasets_parallel(
    dataset_names: list,
    max_workers: int,
    desc: str = "Downloading",
) -> list:
    """Download datasets in parallel with a thread pool.

    Returns list of (name, error) tuples for failed downloads.
    """
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_dataset, name): name
            for name in dataset_names
        }
        pbar = tqdm_bar(total=len(dataset_names), desc=desc, colour="green")
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\nError downloading {name}: {e}")
                failures.append((name, e))
            pbar.update(1)
        pbar.close()
    return failures


def download_datasets_sequential(
    dataset_names: list,
    desc: str = "Downloading",
) -> list:
    """Download datasets sequentially (for CTU datasets).

    Returns list of (name, error) tuples for failed downloads.
    """
    failures = []
    for name in tqdm_bar(dataset_names, desc=desc, colour="cyan"):
        try:
            download_dataset(name)
        except Exception as e:
            print(f"\nError downloading {name}: {e}")
            failures.append((name, e))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to download (default: all available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel downloads (for rel-* datasets)",
    )
    parser.add_argument(
        "--ctu-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel downloads for CTU datasets. "
            "Use 1 (sequential) for best reliability."
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available datasets, don't download",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Disable the DBDataset monkey patch (not recommended)",
    )
    args = parser.parse_args()

    # Apply monkey patch for connection retry support (unless disabled)
    if not args.no_patch:
        apply_db_dataset_patch()

    # Get dataset names
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = get_dataset_names()

    print(f"Found {len(dataset_names)} datasets: {dataset_names}")

    if args.list_only:
        print("\nAvailable datasets:")
        for name in dataset_names:
            print(f"  - {name}")
        return

    # Separate CTU and RelBench datasets
    ctu_datasets = [n for n in dataset_names if n.startswith("ctu-")]
    rel_datasets = [n for n in dataset_names if not n.startswith("ctu-")]

    all_failures = []

    # Download RelBench datasets in parallel (more reliable, from relbench server)
    if rel_datasets:
        print(f"\nDownloading {len(rel_datasets)} RelBench datasets "
              f"(parallel, {args.workers} workers)...")
        failures = download_datasets_parallel(
            rel_datasets,
            max_workers=args.workers,
            desc="RelBench datasets",
        )
        all_failures.extend(failures)

    # Download CTU datasets with reduced parallelism
    # These hit the same remote MariaDB server and are prone to timeouts
    if ctu_datasets:
        if args.ctu_workers == 1:
            print(f"\nDownloading {len(ctu_datasets)} CTU datasets "
                  "(sequential for reliability)...")
            failures = download_datasets_sequential(
                ctu_datasets,
                desc="CTU datasets",
            )
        else:
            print(f"\nDownloading {len(ctu_datasets)} CTU datasets "
                  f"(parallel, {args.ctu_workers} workers)...")
            failures = download_datasets_parallel(
                ctu_datasets,
                max_workers=args.ctu_workers,
                desc="CTU datasets",
            )
        all_failures.extend(failures)

    # Summary
    total = len(dataset_names)
    success = total - len(all_failures)
    print(f"\nDownload complete: {success}/{total} datasets succeeded")

    if all_failures:
        print("\nFailed datasets:")
        for name, err in all_failures:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
