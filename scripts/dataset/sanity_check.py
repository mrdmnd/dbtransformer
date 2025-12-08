"""
Sanity check script that loads all RelBench datasets
and counts tables and rows.
"""

from loguru import logger
from relbench.datasets import get_dataset, get_dataset_names


def format_number(n: int) -> str:
    """Format large numbers with commas for readability."""
    return f"{n:,}"


def main() -> None:
    dataset_names = get_dataset_names()
    logger.info(f"Found {len(dataset_names)} datasets: {dataset_names}")

    total_tables = 0
    total_rows = 0

    for dataset_name in dataset_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading dataset: {dataset_name}")
        logger.info(f"{'=' * 60}")

        try:
            dataset = get_dataset(dataset_name, download=True)
            db = dataset.get_db()

            table_dict = db.table_dict
            num_tables = len(table_dict)
            total_tables += num_tables

            dataset_rows = 0
            logger.info(f"  Tables ({num_tables}):")

            for table_name, table in sorted(table_dict.items()):
                row_count = len(table)
                dataset_rows += row_count
                pkey = table.pkey_col or "None"
                time_col = table.time_col or "None"
                fkeys = list(table.fkey_col_to_pkey_table.keys())
                fkeys_str = ", ".join(fkeys) if fkeys else "None"

                logger.info(
                    f"    - {table_name}: "
                    f"{format_number(row_count)} rows | "
                    f"pkey={pkey} | time={time_col} | fkeys=[{fkeys_str}]"
                )

            total_rows += dataset_rows
            logger.info(f"  Dataset total: {num_tables} tables, {format_number(dataset_rows)} rows")

        except Exception as e:
            logger.error(f"  Failed to load dataset {dataset_name}: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total datasets: {len(dataset_names)}")
    logger.info(f"Total tables:   {total_tables}")
    logger.info(f"Total rows:     {format_number(total_rows)}")


if __name__ == "__main__":
    main()
