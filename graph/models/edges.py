from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graph.models.types import EdgeType


class BaseEdge(BaseModel):
    # every edge has a source, target, and type


    id: UUID = Field(default_factory=uuid4)
    edge_type: EdgeType
    source: UUID
    target: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# how experiments relate to each other. graph only grows, never shrinks.

class Tried(BaseEdge):
    # experiment used a technique

    edge_type: EdgeType = EdgeType.tried
    was_novel: bool = False



class ImprovedFrom(BaseEdge):
    
    # beat the parent it branched from

    edge_type: EdgeType = EdgeType.improved_from
    delta_val_bpb: float = 0.0


class FailedFrom(BaseEdge):
    # tried to improve on parent but it was worse

    edge_type: EdgeType = EdgeType.failed_from
    delta_val_bpb: float = 0.0


class LedTo(BaseEdge):
    # why one experiment led to the next

    edge_type: EdgeType = EdgeType.led_to
    rationale: str = ""
    next_idea: str = ""


# what we learned, not just what we did

class MotivatedBy(BaseEdge):
    # why we ran this

    edge_type: EdgeType = EdgeType.motivated_by


class Tested(BaseEdge):
    # this run was testing that hypothesis

    edge_type: EdgeType = EdgeType.tested


class Produced(BaseEdge):
    # what we learned from running it

    edge_type: EdgeType = EdgeType.produced


class Contradicts(BaseEdge):
    # two results that disagree

    edge_type: EdgeType = EdgeType.contradicts
    explanation: str = ""


# debate — proposer vs challenger

class Challenged(BaseEdge):
    # challenger's argument against running this hypothesis

    edge_type: EdgeType = EdgeType.challenged
    reason: str = ""
    round: int = 1


class Refines(BaseEdge):
    # updated hypothesis after seeing new evidence

    edge_type: EdgeType = EdgeType.refines
    based_on_result: str = ""


# infra — who ran what, where

class RanBy(BaseEdge):
   # which agent ...

    edge_type: EdgeType = EdgeType.ran_by


class RanOn(BaseEdge):
    # gpu

    edge_type: EdgeType = EdgeType.ran_on


class PartOf(BaseEdge):
    # groups experiments into overnight batch sessions

    edge_type: EdgeType = EdgeType.part_of


class BeatGlobal(BaseEdge):
    # new global best across all agents

    edge_type: EdgeType = EdgeType.beat_global
    improvement: float = 0.0


class BeatOwn(BaseEdge):
    # new personal best for this agent

    edge_type: EdgeType = EdgeType.beat_own
    improvement: float = 0.0


# technique interactions

class Invalidates(BaseEdge):
    # one technique makes the other worse

    edge_type: EdgeType = EdgeType.invalidates
    condition: str = ""


class CorrelatedWith(BaseEdge):
    # these techniques tend to work well together

    edge_type: EdgeType = EdgeType.correlated_with
    joint_success_rate: float = 0.0
