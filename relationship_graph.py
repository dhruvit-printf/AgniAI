import logging
from typing import Dict, List, Optional, Tuple
import collections
import itertools
import heapq
import functools

from schema_engine import schema_engine

logger = logging.getLogger(__name__)

class RelationshipGraph:
    """
    Automated Join Path Discovery.
    Models tables as Nodes, Foreign Keys as Edges.
    Uses BFS to find the minimum joins between tables.
    """

    def __init__(self, engine=None):
        self.engine = engine or schema_engine
        self.adj: Dict[str, set[Tuple[str, str, str, int]]] = collections.defaultdict(set)
        self.build_graph()

    def _infer_target_table(self, col: str, all_tables: List[str]) -> Optional[str]:
        if not col.endswith("Id"):
            return None
        if col == "Id":
            return None
            
        base_name = col[:-2] # e.g. "Agniveer" from "AgniveerId"
        
        # Exact match to Master
        if f"{base_name}Master" in all_tables:
            return f"{base_name}Master"
            
        # Special cases for score tracking hierarchy
        if base_name == "SubItem" and "ScoreSubItemMaster" in all_tables:
            return "ScoreSubItemMaster"
        if base_name == "Section" and "ScoreSectionMaster" in all_tables:
            return "ScoreSectionMaster"
            
        return None

    def build_graph(self):
        """Builds an adjacency list representing undirected edges between tables."""
        tables = self.engine.get_tables()
        for t1 in tables:
            columns = self.engine.get_columns(t1)
            for col in columns:
                target_table = self._infer_target_table(col, tables)
                if target_table:
                    # Edge from t1 to target_table via col, weight=1 for direct FK
                    self.adj[t1].add((target_table, col, "Id", 1))
                    # Reverse edge
                    self.adj[target_table].add((t1, "Id", col, 1))

    @functools.lru_cache(maxsize=1024)
    def find_shortest_path(self, start_table: str, end_table: str) -> Optional[List[Dict[str, str]]]:
        """
        Uses Dijkstra's algorithm to find the lowest-cost join path from start_table to end_table.
        Returns a list of join steps.
        """
        if start_table == end_table:
            return []

        # Priority queue stores: (cumulative_cost, tie_breaker, node, path_of_joins)
        # The tie-breaker keeps heap operations deterministic even when two
        # candidate paths have the same cost and path lengths.
        counter = itertools.count()
        queue = [(0, next(counter), start_table, [])]
        visited = set()

        while queue:
            cost, _, current, path = heapq.heappop(queue)

            if current == end_table:
                return path

            if current in visited:
                continue
            visited.add(current)

            for neighbor, col1, col2, weight in sorted(
                self.adj[current], key=lambda edge: (edge[0], edge[1], edge[2], edge[3])
            ):
                if neighbor not in visited:
                    new_path = path + [{
                        'left': current,
                        'right': neighbor,
                        'left_col': col1,
                        'right_col': col2
                    }]
                    heapq.heappush(queue, (cost + weight, next(counter), neighbor, new_path))

        return None

relationship_graph = RelationshipGraph()
