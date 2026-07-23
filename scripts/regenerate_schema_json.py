import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, Set

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the parent directory is in sys.path so we can import workspace modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sql_schema_guard


def load_json_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_schema(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate actual_schema.json from the live database schema."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes to actual_schema.json and save a backup.",
    )
    args = parser.parse_args()

    schema_json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../actual_schema.json")
    )

    if not os.path.exists(schema_json_path):
        print(f"Error: actual_schema.json not found at {schema_json_path}")
        sys.exit(1)

    print("Fetching live schema from database...")
    live_schema = sql_schema_guard.fetch_live_schema()
    if live_schema is None:
        print("Error: Could not connect to the database to fetch live schema.")
        sys.exit(1)

    print("Loading original actual_schema.json...")
    original_schema = load_json_schema(schema_json_path)

    # Prepare new schema
    new_schema = {}

    # 1. Identify table changes
    original_tables = set(original_schema.keys())
    live_tables = set(live_schema.keys())

    # Exclude __EFMigrationsHistory and other denied tables
    denied_tables = getattr(sql_schema_guard, "DENIED_TABLES", set())

    tables_to_remove = sorted(list(original_tables - live_tables))
    tables_to_add = sorted(
        list(live_tables - original_tables - {"__EFMigrationsHistory"} - denied_tables)
    )

    print("\n--- TABLE LEVEL CHANGES ---")
    print(f"Tables to remove ({len(tables_to_remove)}): {tables_to_remove}")
    print(f"Tables to add ({len(tables_to_add)}): {tables_to_add}")

    # Process all tables present in live DB
    column_diffs = {}

    for table_name in sorted(live_tables):
        if table_name == "__EFMigrationsHistory" or table_name.lower() in denied_tables:
            continue

        live_cols = live_schema[table_name]

        # If it was an original table, diff its columns
        if table_name in original_schema:
            original_cols = set(original_schema[table_name].keys())

            cols_removed = sorted(list(original_cols - live_cols))
            cols_added = sorted(list(live_cols - original_cols))

            if cols_removed or cols_added:
                column_diffs[table_name] = {
                    "removed": cols_removed,
                    "added": cols_added,
                }

            # Build the new table columns dict
            new_table_cols = {}
            for col in sorted(live_cols):
                if col in original_schema[table_name]:
                    # Preserve original metadata (type, nullable, etc.)
                    new_table_cols[col] = original_schema[table_name][col]
                else:
                    # New column, add empty defaults
                    new_table_cols[col] = {"type": "", "nullable": True}
            new_schema[table_name] = new_table_cols
        else:
            # Entirely new table
            new_table_cols = {}
            for col in sorted(live_cols):
                new_table_cols[col] = {"type": "", "nullable": True}
            new_schema[table_name] = new_table_cols

    # Report column changes
    print("\n--- COLUMN LEVEL CHANGES ---")
    if not column_diffs:
        print("No column-level drift detected for existing tables.")
    else:
        for table, diff in sorted(column_diffs.items()):
            rem = diff["removed"]
            add = diff["added"]
            print(f"Table '{table}':")
            if rem:
                print(f"  - Removed columns: {rem}")
            if add:
                print(f"  - Added columns: {add}")

    # Write changes if requested
    if args.write:
        # Create backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{schema_json_path}.{timestamp}.bak"
        print(
            f"\nWriting backup of actual_schema.json to {os.path.basename(backup_path)}..."
        )
        save_json_schema(backup_path, original_schema)

        print(f"Writing updated schema to {os.path.basename(schema_json_path)}...")
        save_json_schema(schema_json_path, new_schema)
        print("Regeneration complete!")
    else:
        print("\n[DRY RUN] No files were written. Run with --write to apply changes.")


if __name__ == "__main__":
    main()
