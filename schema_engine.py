import json
import logging
import os
import functools
from typing import Dict, Any, List, Optional, Tuple

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
            
        if column_name in ["AgniveerNo", "PlatoonNo"]:
            return "string"
        if column_name in ["IsPresent", "IsBestAttempt", "IsActive", "IsDisqualified", "IsHospitalized", "IsAbscondedLeave"]:
            return "boolean"
        if column_name in ["MarksObtained", "MaxMarks", "Height", "Weight", "SubItemTotalMarks", "ExceptionalMarks"]:
            return "float"
        if "Date" in column_name or "Time" in column_name:
            return "datetime"
        if column_name.endswith("Id") or column_name == "Id" or "No" in column_name or column_name.endswith("Total"):
            return "integer"
        if "Is" in column_name or "On" in column_name:
            return "boolean"
        return "string"

    def get_column(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Return comprehensive metadata for a single column."""
        col_data = self.schema.get(table_name, {}).get(column_name, {})
        return {
            "name": column_name,
            "type": self.get_column_type(table_name, column_name),
            "nullable": col_data.get("nullable", True),
            "default_value": col_data.get("default", None),
            "is_identity": column_name == "Id",
            "is_primary_key": column_name == self.get_primary_key(table_name)
        }

    def get_table(self, table_name: str) -> Dict[str, Any]:
        """Return high-level metadata for a single table."""
        if table_name not in self.schema:
            return {}
        return {
            "name": table_name,
            "columns": self.get_columns(table_name),
            "primary_key": self.get_primary_key(table_name)
        }

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

    @functools.lru_cache(maxsize=1024)
    def get_concept_for_table(self, table_name: str) -> Optional[str]:
        """Resolves a physical table back to its business concept."""
        concepts = self.ontology.get("concepts", {})
        for concept_name, concept_data in concepts.items():
            if concept_data.get("table") == table_name:
                return concept_name
        return None

    def get_implicit_filters(self, concept_name: str) -> Dict[str, Any]:
        """Returns implicit filters for a concept (e.g., IsPresent=1 for Attendance)."""
        concepts = self.ontology.get("concepts", {})
        concept = concepts.get(concept_name)
        if concept:
            return concept.get("implicit_filters", {})
        return {}
        
    @functools.lru_cache(maxsize=1024)
    def get_date_column(self, concept_name: str) -> Optional[str]:
        """Returns the default date column for temporal filtering."""
        concepts = self.ontology.get("concepts", {})
        concept = concepts.get(concept_name)
        if concept:
            return concept.get("default_date_column")
        return None

    # --- New Relational APIs ---

    # Real foreign keys the "{Prefix}Id" -> "{Prefix}Master" naming heuristic
    # below cannot infer, because either the referenced table doesn't follow
    # that naming convention (Doctor/Team/SponserUnit -> no such *Master
    # table exists) or the column's casing doesn't match the target table's
    # (AgniVeerId vs. AgniveerMaster). Sourced from the actual SQL FK
    # constraints in the DB_Agni schema dump — verified against
    # sys.foreign_keys, not guessed. Keyed by (table, column); checked before
    # the naming heuristic so a real relationship always wins over a wrong
    # heuristic guess for the same column.
    _FK_OVERRIDES: Dict[Tuple[str, str], Dict[str, str]] = {
        ("MedicalRecordMaster", "DoctorId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("AgniveerScoreAttempt", "EvaluatedBy"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("AgniveerAttendanceMaster", "MarkedBy"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("AgniveerLeaveMaster", "MarkedBy"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("UserMaster", "AgniVeerId"): {"referenced_table": "AgniveerMaster", "referenced_column": "Id"},
        ("AgniveerMaster", "SponserUnitId"): {"referenced_table": "DistributionMaster", "referenced_column": "Id"},
        ("DistributionHistoryMaster", "TeamId"): {"referenced_table": "DistributionMaster", "referenced_column": "Id"},
        ("CompanyMaster", "CompanyCommanderId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("CompanyMaster", "CommandingOfficerId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("PlatoonMaster", "PlatoonCommanderId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("CompanyCommanderHistory", "CommanderId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("CompanyCommandingOfficerHistory", "CommandingOfficerId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
        ("PlatoonCommanderHistory", "CommanderId"): {"referenced_table": "UserMaster", "referenced_column": "Id"},
    }

    @functools.lru_cache(maxsize=1024)
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """Infers outgoing foreign keys.

        Checks the curated `_FK_OVERRIDES` table first (real relationships
        the naming heuristic below cannot express), then falls back to
        analyzing "*Id" columns via the "{Prefix}Id" -> "{Prefix}Master"
        convention most of this schema follows.
        """
        fks = []
        tables = set(self.get_tables())
        seen_columns = set()

        for col in self.get_columns(table_name):
            override = self._FK_OVERRIDES.get((table_name, col))
            if override:
                fks.append({"column": col, **override})
                seen_columns.add(col)

        for col in self.get_columns(table_name):
            if col in seen_columns:
                continue
            if col.endswith("Id") and col != "Id":
                base = col[:-2]
                if f"{base}Master" in tables:
                    fks.append({
                        "column": col,
                        "referenced_table": f"{base}Master",
                        "referenced_column": "Id"
                    })
                # Add special cases for relationship graph
                elif base == "SubItem" and "ScoreSubItemMaster" in tables:
                    fks.append({"column": col, "referenced_table": "ScoreSubItemMaster", "referenced_column": "Id"})
                elif base == "Section" and "ScoreSectionMaster" in tables:
                    fks.append({"column": col, "referenced_table": "ScoreSectionMaster", "referenced_column": "Id"})
        return fks

    @functools.lru_cache(maxsize=1024)
    def get_reverse_relationships(self, table_name: str) -> List[Dict[str, str]]:
        """Infers incoming foreign keys (which tables point to this one)."""
        rev = []
        for t in self.get_tables():
            if t == table_name:
                continue
            for fk in self.get_foreign_keys(t):
                if fk["referenced_table"] == table_name:
                    rev.append({
                        "table": t,
                        "column": fk["column"]
                    })
        return rev

    def get_relationships(self, table_name: str) -> Dict[str, List]:
        """Returns both outgoing and incoming relationships."""
        return {
            "foreign_keys": self.get_foreign_keys(table_name),
            "reverse_relationships": self.get_reverse_relationships(table_name)
        }

    # --- New Semantic APIs ---

    @functools.lru_cache(maxsize=1024)
    def get_display_columns(self, table_name: str) -> List[str]:
        """Returns columns that are suitable for human-readable display titles."""
        cols = self.get_columns(table_name)
        candidates = ["FullName", "Name", "AgniveerNo", "Title", "SectionName", "BatchName"]
        found = [c for c in cols if c in candidates]
        return found if found else ["Id"]

    @functools.lru_cache(maxsize=1024)
    def get_searchable_columns(self, table_name: str) -> List[str]:
        """Returns columns suitable for text search."""
        return [c for c in self.get_columns(table_name) if self.get_column_type(table_name, c) == "string"]

    @functools.lru_cache(maxsize=1024)
    def get_aggregation_columns(self, table_name: str) -> List[str]:
        """Returns numeric columns suitable for SUM, AVG, etc."""
        return [c for c in self.get_columns(table_name) if self.get_column_type(table_name, c) in ["integer", "float"] and not c.endswith("Id") and c != "Id" and "No" not in c]

    @functools.lru_cache(maxsize=1024)
    def get_ranking_columns(self, table_name: str) -> List[str]:
        """Returns columns suitable for ranking or sorting."""
        cols = self.get_columns(table_name)
        candidates = ["MarksObtained", "Score", "Percentage", "Rating", "Rank", "TotalMarks", "Height", "Weight"]
        return [c for c in cols if any(cand in c for cand in candidates)]

    @functools.lru_cache(maxsize=1024)
    def get_status_columns(self, table_name: str) -> List[str]:
        """Returns columns representing state or status."""
        return [c for c in self.get_columns(table_name) if "Status" in c or c.startswith("Is") or c.startswith("On")]

    @functools.lru_cache(maxsize=1024)
    def get_date_columns(self, table_name: str) -> List[str]:
        """Returns all datetime columns for a table."""
        return [c for c in self.get_columns(table_name) if self.get_column_type(table_name, c) == "datetime"]

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Returns the fully aggregated metadata object for a table."""
        return {
            "table": self.get_table(table_name),
            "concept": self.get_concept_for_table(table_name),
            "relationships": self.get_relationships(table_name),
            "display_columns": self.get_display_columns(table_name),
            "searchable_columns": self.get_searchable_columns(table_name),
            "aggregation_columns": self.get_aggregation_columns(table_name),
            "ranking_columns": self.get_ranking_columns(table_name),
            "date_columns": self.get_date_columns(table_name),
            "status_columns": self.get_status_columns(table_name)
        }

schema_engine = SchemaEngine()
