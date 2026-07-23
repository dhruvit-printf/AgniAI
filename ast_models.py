from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


class ConditionNode:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class WhereNode(ConditionNode):
    column: str
    operator: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "operator": self.operator, "value": self.value}


@dataclass
class ConditionGroupNode(ConditionNode):
    operator: str  # 'AND' or 'OR'
    conditions: List[ConditionNode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_operator": self.operator,
            "conditions": [c.to_dict() for c in self.conditions],
        }


@dataclass
class JoinNode:
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    join_type: str = "INNER"


@dataclass
class AggregateNode:
    column: str
    function: str  # SUM, AVG, COUNT, MIN, MAX
    alias: Optional[str] = None


@dataclass
class OrderByNode:
    column: str
    descending: bool = True


@dataclass
class ASTNode:
    """Base logical query plan"""

    type: str = "Select"
    base_table: str = ""
    columns: List[str] = field(default_factory=list)
    joins: List[JoinNode] = field(default_factory=list)
    where: List[ConditionNode] = field(default_factory=list)
    having: List[ConditionNode] = field(default_factory=list)
    aggregates: List[AggregateNode] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: List[OrderByNode] = field(default_factory=list)
    limit: Optional[int] = None
    is_distinct: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "base_table": self.base_table,
            "columns": self.columns,
            "joins": [j.__dict__ for j in self.joins],
            "where": [w.to_dict() for w in self.where],
            "having": [h.to_dict() for h in self.having],
            "aggregates": [a.__dict__ for a in self.aggregates],
            "group_by": self.group_by,
            "order_by": [o.__dict__ for o in self.order_by],
            "limit": self.limit,
            "is_distinct": self.is_distinct,
            "metadata": self.metadata,
        }
