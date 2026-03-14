from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graph.models.types import Category, GPUType, HypothesisStatus, Status


class Experiment(BaseModel):
    """single training run tied to a git commit."""

    id: UUID = Field(default_factory=uuid4)
    experiment_id: int
    commit: str = Field(min_length=7, max_length=40)

    # metrics
    val_bpb: float
    baseline_bpb: float = 0.0
    delta_bpb: float = 0.0
    delta_vs_best: float = 0.0
    delta_vs_own_best: float = 0.0
    global_best_at_publish: float = 0.0
    agent_best_at_publish: float = 0.0
    memory_gb: float = 0.0
    peak_vram_mb: float = 0.0
    status: Status

    # training details
    training_steps: int = 0
    mfu_percent: float = 0.0
    tokens_trained_m: float = 0.0
    seed: int = 42
    batch: str = ""
    run_name: str = ""

    # description
    hypothesis: str = ""
    change_summary: str = ""
    category: Category = Category.hyperparameter
    reasoning: str = ""
    insights: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # lineage
    parent_id: int | None = None
    builds_on: list[int] = Field(default_factory=list)
    contradicts: list[int] = Field(default_factory=list)
    supports: list[int] = Field(default_factory=list)

    # components
    components: list[str] = Field(default_factory=list)
    parameters_changed: dict[str, dict] = Field(default_factory=dict)

    # agent + infra
    agent_id: str = "unknown"
    gpu: str = ""
    repo_url: str = ""
    branch: str = ""
    commit_url: str = ""

    # timestamps
    timestamp: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Technique(BaseModel):
    """a reusable method extracted from experiments."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    original_name: str = ""
    category: Category = Category.hyperparameter
    success_rate: float = 0.0


class Hypothesis(BaseModel):
    """a proposed idea, pre-run. status tracks if it was confirmed or rejected."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    status: HypothesisStatus = HypothesisStatus.pending
    category: Category = Category.hyperparameter
    debate_rounds: int = 0
    challenger_agreed: bool = False
    winning_argument: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None  # when this hypothesis became active
    valid_until: datetime | None = None  # when it was confirmed/rejected/superseded


class Result(BaseModel):
    """post-run finding. what we actually learned from an experiment."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    val_bpb: float
    delta: float = 0.0
    kept: bool = False
    category: Category = Category.hyperparameter
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None  # when this finding was established
    valid_until: datetime | None = None  # when it was overturned (None = still holds)
    superseded_by: UUID | None = None  # which result replaced this one


class Run(BaseModel):
    """one batch session (e.g. overnight run). groups experiments together."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    total_experiments: int = 0
    best_val_bpb: float = float("inf")
    keep_count: int = 0


class Agent(BaseModel):
    """an agent that runs experiments."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    gpu: str = ""
    total_runs: int = 0
    best_val_bpb: float = float("inf")
    keep_rate: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
