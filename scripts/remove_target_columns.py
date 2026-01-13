#!/usr/bin/env python3
"""Remove target_columns from all metadata.json files."""

import argparse
import json
from pathlib import Path


def remove_target_columns(metadata_path: Path, dry_run: bool = False) -> bool:
    """Remove target_columns from a metadata.json file.
    
    Returns True if the file was modified.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    modified = False
    for table_name, table_meta in metadata.items():
        if "target_columns" in table_meta:
            if dry_run:
                print(f"  Would remove target_columns from {table_name}: {table_meta['target_columns']}")
            else:
                del table_meta["target_columns"]
            modified = True
    
    if modified and not dry_run:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Updated {metadata_path}")
    
    return modified


def main():
    parser = argparse.ArgumentParser(description="Remove target_columns from metadata.json files")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Paths to metadata.json files or directories to search recursively",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    metadata_files = []
    for path in args.paths:
        if path.is_file() and path.name == "metadata.json":
            metadata_files.append(path)
        elif path.is_dir():
            metadata_files.extend(path.rglob("metadata.json"))
        else:
            print(f"Warning: {path} is not a metadata.json file or directory")

    if not metadata_files:
        print("No metadata.json files found")
        return

    print(f"Found {len(metadata_files)} metadata.json file(s)")
    
    modified_count = 0
    for metadata_path in metadata_files:
        print(f"Processing {metadata_path}...")
        if remove_target_columns(metadata_path, args.dry_run):
            modified_count += 1

    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {modified_count} file(s)")


if __name__ == "__main__":
    main()
