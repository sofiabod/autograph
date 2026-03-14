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
    valid_from: datetime | None = None  # when this became true
    valid_until: datetime | None = None  # when this stopped being true (None = still valid)


# how experiments relate to each other. graph only grows, never shrinks.

class Tried(BaseEdge):
    # experiment used a technique

    edge_type: EdgeType = EdgeType.tried
    was_novel: bool = False
    parameters: dict = Field(default_factory=dict)  # what values were used
    outcome_summary: str = ""  # one-line what happened


class ImprovedFrom(BaseEdge):
    # beat the parent it branched from

    edge_type: EdgeType = EdgeType.improved_from
    delta_val_bpb: float = 0.0
    why_it_worked: str = ""  # reasoning about why this helped
    technique_used: str = ""  # which technique drove the improvement


class FailedFrom(BaseEdge):
    # tried to improve on parent but it was worse

    edge_type: EdgeType = EdgeType.failed_from
    delta_val_bpb: float = 0.0
    why_it_failed: str = ""  # what went wrong
    lesson_learned: str = ""  # what we should avoid next time


class LedTo(BaseEdge):
    # why one experiment led to the next

    edge_type: EdgeType = EdgeType.led_to
    rationale: str = ""
    next_idea: str = ""
    confidence: float = 0.0  # how confident are we in this reasoning
    evidence: list[int] = Field(default_factory=list)  # experiment_ids that support this


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
    significance: str = ""  # how important is this finding
    confidence: float = 0.0  # how much we trust it


class Contradicts(BaseEdge):
    # two results that disagree

    edge_type: EdgeType = EdgeType.contradicts
    explanation: str = ""
    resolution: str = ""  # how we resolved it (or empty if unresolved)
    stronger_result: str = ""  # which one we trust more


# debate — proposer vs challenger

class Challenged(BaseEdge):
    # challenger's argument against running this hypothesis

    edge_type: EdgeType = EdgeType.challenged
    reason: str = ""
    round: int = 1
    evidence_cited: list[int] = Field(default_factory=list)  # experiment_ids the challenger pointed to
    proposer_rebuttal: str = ""  # what the proposer said back


class Refines(BaseEdge):
    # updated hypothesis after seeing new evidence

    edge_type: EdgeType = EdgeType.refines
    based_on_result: str = ""
    what_changed: str = ""  # how the hypothesis evolved
    why: str = ""  # why it was refined


# infra — who ran what, where

class RanBy(BaseEdge):
    # which agent ran this

    edge_type: EdgeType = EdgeType.ran_by


class RanOn(BaseEdge):
    # which gpu

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
