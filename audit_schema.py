import re

def parse_sql_schema(file_path):
    tables = {}
    current_table = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Look for CREATE TABLE [dbo].[TableName](
        table_match = re.match(r"CREATE TABLE \[dbo\]\.\[(.*?)\]\s*\(", line)
        if table_match:
            current_table = table_match.group(1)
            tables[current_table] = {}
            continue
            
        if current_table:
            # We are inside a table definition
            # End of table definition check (a line starting with ) ON [PRIMARY])
            if line.startswith(") ON"):
                current_table = None
                continue
                
            # Skip constraints and PKs
            if line.startswith("CONSTRAINT") or line.startswith("PRIMARY KEY") or line.startswith("UNIQUE"):
                continue
            
            # Match a column definition: [ColName] [type](size) NULL/NOT NULL,
            col_match = re.match(r"^\[(.*?)\]\s+(.*)", line)
            if col_match:
                col_name = col_match.group(1)
                col_type_info = col_match.group(2)
                
                is_nullable = "NOT NULL" not in col_type_info.upper()
                type_match = re.match(r"\[?(.*?)\]?(?:\(.*?\))?", col_type_info.split()[0])
                col_type = type_match.group(1) if type_match else "UNKNOWN"
                
                tables[current_table][col_name] = {
                    "type": col_type,
                    "nullable": is_nullable
                }
                
    return tables

if __name__ == "__main__":
    db_schema = parse_sql_schema("extracted_schema.sql")
    
    import json
    with open("actual_schema.json", "w", encoding="utf-8") as f:
        json.dump(db_schema, f, indent=2)
