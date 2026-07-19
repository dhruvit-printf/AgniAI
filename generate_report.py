import json
import re


def parse_md_schema(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    schema = {}
    current_table = None

    for line in lines:
        line = line.strip()
        # Find tables: * **TableName**
        table_match = re.match(r"^\*\s*\*\*(.*?)\*\*", line)
        if table_match:
            current_table = table_match.group(1).strip()
            schema[current_table] = set()
            continue

        # Find columns: - `Col1`, `Col2`
        if current_table and line.startswith("-"):
            # extract anything in backticks
            cols = re.findall(r"`(.*?)`", line)
            for c in cols:
                # clean up PK/FK annotations inside backticks if any, usually they are outside
                c = c.replace("[", "").replace("]", "").replace("'", "")
                if c:
                    schema[current_table].add(c)

    return schema


def generate_report():
    md_schema = parse_md_schema(
        r"C:\Users\dhruv\.gemini\antigravity-ide\brain\d6a094fd-1341-4992-9ec1-f64684561bbb\database_schema.md"
    )
    with open("actual_schema.json", "r", encoding="utf-8") as f:
        actual_schema = json.load(f)

    report = ["# Schema Audit Report\n"]

    # Check each table in MD against Actual
    for table, md_cols in md_schema.items():
        if table not in actual_schema:
            report.append(f"### Table: {table}")
            report.append(f"**Current:** Present in SCHEMA_CARD")
            report.append(f"**Actual:** Missing in Database")
            report.append(f"**Problem:** Hallucinated table in schema card.")
            report.append(f"**Fix:** Remove {table} from SCHEMA_CARD.\n")
            continue

        actual_cols_info = actual_schema[table]
        # remove brackets and quotes from actual columns to match MD
        actual_cols = {
            c.replace("[", "").replace("]", "").replace("'", "")
            for c in actual_cols_info.keys()
        }
        # but also keep a mapping to the raw name
        raw_col_map = {
            c.replace("[", "").replace("]", "").replace("'", ""): c
            for c in actual_cols_info.keys()
        }

        # Missing columns in MD (present in DB, but omitted in MD)
        missing_in_md = actual_cols - md_cols
        # Extra columns in MD (hallucinated in MD, not in DB)
        extra_in_md = md_cols - actual_cols

        if missing_in_md or extra_in_md:
            report.append(f"### Table: {table}")

        for col in extra_in_md:
            report.append(f"**Current:** Column `{col}` in SCHEMA_CARD")
            report.append(f"**Actual:** Does not exist in Database")
            report.append(f"**Problem:** Hallucinated column.")
            report.append(f"**Fix:** Remove `{col}` from SCHEMA_CARD.\n")

        for col in missing_in_md:
            # We don't report Id if the schema card has it, wait, MD has Id for AgniveerMaster but actual is 'Id'.
            # Also MD might miss audit columns.
            report.append(
                f"**Current:** Missing column `{raw_col_map[col]}` in SCHEMA_CARD"
            )
            report.append(f"**Actual:** Present in Database")
            report.append(f"**Problem:** Incomplete schema definition.")
            report.append(f"**Fix:** Add `{raw_col_map[col]}` to SCHEMA_CARD.\n")

    # Check tables in Actual that are not in MD (ignoring __EFMigrationsHistory)
    for table in actual_schema.keys():
        if table not in md_schema and table != "__EFMigrationsHistory":
            report.append(f"### Table: {table}")
            report.append(f"**Current:** Missing in SCHEMA_CARD")
            report.append(f"**Actual:** Present in Database")
            report.append(f"**Problem:** Undocumented table.")
            report.append(f"**Fix:** Add `{table}` to SCHEMA_CARD.\n")

    with open("schema_audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    generate_report()
