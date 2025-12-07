"""
Sanity check script for RelBench datasets.

Loops through all registered datasets, loads them, and verifies they
contain data.

Usage:
    uv run python scripts/sanity_check.py
    uv run python scripts/sanity_check.py --datasets rel-f1 rel-trial
    uv run python scripts/sanity_check.py --verbose
"""

import argparse
import sys
from dataclasses import dataclass

import redelex  # noqa: F401  # importing redelex registers CTU datasets
from relbench.datasets import get_dataset, get_dataset_names


@dataclass
class DatasetCheckResult:
    """Result of checking a single dataset."""

    name: str
    success: bool
    num_tables: int = 0
    total_rows: int = 0
    table_info: dict[str, int] | None = None
    error: str | None = None


def check_dataset(
    dataset_name: str,
    verbose: bool = False,
) -> DatasetCheckResult:
    """Load a dataset and verify it has data.

    Args:
        dataset_name: Name of the dataset to check.
        verbose: If True, collect detailed table info.

    Returns:
        DatasetCheckResult with check outcome.
    """
    try:
        # Load dataset without downloading (assume already cached)
        dataset = get_dataset(dataset_name, download=False)
        db = dataset.get_db()

        # Count tables and rows
        table_dict = db.table_dict
        num_tables = len(table_dict)
        table_info = {}
        total_rows = 0

        for table_name, table in table_dict.items():
            row_count = len(table.df)
            table_info[table_name] = row_count
            total_rows += row_count

        # Check that we have at least some data
        if num_tables == 0:
            return DatasetCheckResult(
                name=dataset_name,
                success=False,
                error="No tables found in database",
            )

        if total_rows == 0:
            return DatasetCheckResult(
                name=dataset_name,
                success=False,
                num_tables=num_tables,
                error="All tables are empty",
            )

        return DatasetCheckResult(
            name=dataset_name,
            success=True,
            num_tables=num_tables,
            total_rows=total_rows,
            table_info=table_info if verbose else None,
        )

    except FileNotFoundError as e:
        return DatasetCheckResult(
            name=dataset_name,
            success=False,
            error=f"Dataset not cached/downloaded: {e}",
        )
    except Exception as e:
        return DatasetCheckResult(
            name=dataset_name,
            success=False,
            error=str(e),
        )


def print_result(result: DatasetCheckResult, verbose: bool = False) -> None:
    """Print the result of a dataset check."""
    if result.success:
        status = "✓"
        details = (
            f"{result.num_tables} tables, "
            f"{result.total_rows:,} total rows"
        )
        print(f"  {status} {result.name}: {details}")

        if verbose and result.table_info:
            for table_name, row_count in result.table_info.items():
                print(f"      - {table_name}: {row_count:,} rows")
    else:
        status = "✗"
        print(f"  {status} {result.name}: FAILED - {result.error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check RelBench datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to check (default: all available)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed table information",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available datasets, don't check them",
    )
    args = parser.parse_args()

    # Get dataset names
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = get_dataset_names()

    print(f"Found {len(dataset_names)} datasets")

    if args.list_only:
        print("\nAvailable datasets:")
        for name in dataset_names:
            print(f"  - {name}")
        return 0

    # Check each dataset
    print("\nChecking datasets...")
    results: list[DatasetCheckResult] = []

    for name in dataset_names:
        result = check_dataset(name, verbose=args.verbose)
        results.append(result)
        print_result(result, verbose=args.verbose)

    # Summary
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    print(f"\nSummary: {success_count}/{len(results)} datasets passed")

    if fail_count > 0:
        print(f"\nFailed datasets ({fail_count}):")
        for r in results:
            if not r.success:
                print(f"  - {r.name}: {r.error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
