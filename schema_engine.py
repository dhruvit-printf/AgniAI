import json
import logging
import os
import functools
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SchemaEngine:
    """
    Schema Intelligence Engine for AgniAI.
    Parses actual_schema.json and business_ontology.json to provide
    a dynamic, semantic understanding of the database structure.
    """

    def __init__(self, schema_path: str = "actual_schema.json", ontology_path: str = "business_ontology.json"):
        self.schema_path = schema_path
        self.ontology_path = ontology_path
        self.schema: Dict[str, Any] = {}
        self.ontology: Dict[str, Any] = {}
        self.load()

    def load(self):
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema from {self.schema_path}: {e}")
            self.schema = {}

        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ontology from {self.ontology_path}: {e}")
            self.ontology = {"concepts": {}}

    def get_tables(self) -> List[str]:
        """Return all tables defined in the schema."""
        return [k for k in self.schema.keys() if k != "__EFMigrationsHistory"]

    @functools.lru_cache(maxsize=1024)
    def get_columns(self, table_name: str) -> List[str]:
        """Return all columns for a given table."""
        if table_name not in self.schema:
            return []
        return list(self.schema[table_name].keys())

    def get_column_type(self, table_name: str, column_name: str) -> str:
        """Heuristically infers column type since actual_schema.json types might be empty."""
        col = self.schema.get(table_name, {}).get(column_name, {})
        explicit_type = col.get("type")
        if explicit_type:
            return explicit_type.lower()
            
        if column_name.endswith("Id") or column_name == "Id" or "No" in column_name:
            return "integer"
        if "Date" in column_name or "Time" in column_name:
            return "datetime"
        if "Is" in column_name or "On" in column_name:
            return "boolean"
        if "Marks" in column_name or "Weight" in column_name or "Height" in column_name:
            return "float"
        return "string"

    def get_primary_key(self, table_name: str) -> Optional[str]:
        """Infers the primary key. In this EF schema, it is typically 'Id'."""
        columns = self.get_columns(table_name)
        if "Id" in columns:
            return "Id"
        return None

    @functools.lru_cache(maxsize=1024)
    def get_table_for_concept(self, concept_name: str) -> Optional[str]:
        """Resolves a business concept to its physical table."""
        concepts = self.ontology.get("concepts", {})
        concept = concepts.get(concept_name)
        if concept:
            return concept.get("table")
        return None

    def get_implicit_filters(self, concept_name: str) -> Dict[str, Any]:
        """Returns implicit filters for a concept (e.g., IsPresent=1 for Attendance)."""
        concepts = self.ontology.get("concepts", {})
        concept = concepts.get(concept_name)
        if concept:
            return concept.get("implicit_filters", {})
        return {}
        
    def get_date_column(self, concept_name: str) -> Optional[str]:
        """Returns the default date column for temporal filtering."""
        concepts = self.ontology.get("concepts", {})
        concept = concepts.get(concept_name)
        if concept:
            return concept.get("default_date_column")
        return None

schema_engine = SchemaEngine()
