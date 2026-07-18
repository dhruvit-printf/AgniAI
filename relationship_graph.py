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
    Uses Dijkstra to find the minimum-cost join path between tables.
    """

    # Tables that represent an actor/account (who did/owns something) rather
    # than a business entity. UserMaster is pointed to by many unrelated
    # columns (DoctorId, MarkedBy, EvaluatedBy, CompanyCommanderId,
    # CommandingOfficerId, PlatoonCommanderId, ...) — in an unweighted graph,
    # Dijkstra will happily "shortcut" an unrelated query (e.g. "which
    # company is this medical record's company") through UserMaster,
    # coincidentally matching a doctor's UserId against some company's
    # CommandingOfficerId, which is a syntactically valid but semantically
    # meaningless join (a doctor is not that company's commanding officer).
    # Weighting hops through these tables higher keeps the real business
    # hierarchy (Agniveer -> Platoon -> Company) preferred, while still
    # allowing a UserMaster hop when it's genuinely the only path, or the
    # actual destination (e.g. "who is Agniveer X's doctor").
    _HUB_TABLES = frozenset({"UserMaster"})
    _HUB_HOP_WEIGHT = 5

    def __init__(self, engine=None):
        self.engine = engine or schema_engine
        self.adj: Dict[str, set[Tuple[str, str, str, int]]] = collections.defaultdict(set)
        self.build_graph()

    def build_graph(self):
        """Builds an adjacency list representing undirected edges between tables.

        Foreign keys are sourced from schema_engine.get_foreign_keys() — the
        single source of truth for FK inference (curated overrides + naming
        heuristic) — rather than re-implementing the same heuristic here.
        Previously this duplicated schema_engine's logic verbatim, which
        meant a fix to one copy (e.g. adding a curated override for
        Doctor/Team/AgniVeerId) silently didn't apply to the other, and this
        graph is what query_planner_v2._add_joins actually walks to build
        live joins — so a gap here silently drops joins, not just metadata.
        """
        tables = self.engine.get_tables()
        for t1 in tables:
            for fk in self.engine.get_foreign_keys(t1):
                col = fk["column"]
                target_table = fk["referenced_table"]
                referenced_column = fk.get("referenced_column", "Id")
                weight = (
                    self._HUB_HOP_WEIGHT
                    if target_table in self._HUB_TABLES or t1 in self._HUB_TABLES
                    else 1
                )
                # Edge from t1 to target_table via col
                self.adj[t1].add((target_table, col, referenced_column, weight))
                # Reverse edge
                self.adj[target_table].add((t1, referenced_column, col, weight))

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
