from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from expgraph.types import Category, Status


class Experiment(BaseModel):
    """single training run tied to a git commit."""

    id: UUID = Field(default_factory=uuid4)
    experiment_id: int
    commit: str = Field(min_length=7, max_length=40)
    user_id: str = "default"

    val_bpb: float
    baseline_bpb: float
    delta_bpb: float
    memory_gb: float
    status: Status

    hypothesis: str = ""
    change_summary: str = ""
    category: Category
    reasoning: str = ""
    insights: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    parent_id: int | None = None
    builds_on: list[int] = Field(default_factory=list)
    contradicts: list[int] = Field(default_factory=list)
    supports: list[int] = Field(default_factory=list)

    components: list[str] = Field(default_factory=list)
    parameters_changed: dict[str, dict] = Field(default_factory=dict)

    agent_id: str = "unknown"
    repo_url: str = ""
    branch: str = ""
    commit_url: str = ""

    timestamp: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Technique(BaseModel):
    """a reusable method extracted from experiments."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    original_name: str = ""
    user_id: str = "default"
    category: Category
    success_rate: float = 0.0


class Hypothesis(BaseModel):
    """a proposed idea, optionally from a debate."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str = "default"
    text: str
    debate_rounds: int = 0
    challenger_agreed: bool = False
    winning_argument: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
