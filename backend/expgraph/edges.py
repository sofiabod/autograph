from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from expgraph.types import EdgeType


class BaseEdge(BaseModel):
    """shared fields for all edges."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str = "default"
    edge_type: EdgeType
    source: UUID
    target: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Tried(BaseEdge):
    """experiment used a technique."""

    edge_type: EdgeType = EdgeType.tried
    was_novel: bool = False
    fact: str = ""


class ImprovedFrom(BaseEdge):
    """experiment improved on a parent experiment."""

    edge_type: EdgeType = EdgeType.improved_from
    delta_val_bpb: float = 0.0


class FailedFrom(BaseEdge):
    """experiment failed to improve on a parent experiment."""

    edge_type: EdgeType = EdgeType.failed_from
    delta_val_bpb: float = 0.0


class MotivatedBy(BaseEdge):
    """experiment was motivated by a hypothesis."""

    edge_type: EdgeType = EdgeType.motivated_by


class Invalidates(BaseEdge):
    """one technique invalidates another under a condition."""

    edge_type: EdgeType = EdgeType.invalidates
    condition: str = ""


class CorrelatedWith(BaseEdge):
    """two techniques tend to succeed together."""

    edge_type: EdgeType = EdgeType.correlated_with
    joint_success_rate: float = 0.0


class LedTo(BaseEdge):
    """captures why one experiment led to the next."""

    edge_type: EdgeType = EdgeType.led_to
    rationale: str = ""
    next_idea: str = ""
