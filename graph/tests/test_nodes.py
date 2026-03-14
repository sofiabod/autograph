from uuid import UUID

import pytest

from graph.models.nodes import Experiment, Technique, Hypothesis, Agent
from graph.models.types import Status, Category


class TestExperiment:
    def test_create_minimal(self):
        e = Experiment(experiment_id=1, commit="abc1234", val_bpb=0.97, status=Status.keep)
        assert e.experiment_id == 1
        assert e.status == Status.keep
        assert isinstance(e.id, UUID)

    def test_full_fields(self):
        e = Experiment(
            experiment_id=225,
            commit="9e9c216",
            val_bpb=0.968636,
            baseline_bpb=0.996003,
            delta_bpb=-0.027367,
            delta_vs_best=-0.00129,
            global_best_at_publish=0.961639,
            agent_best_at_publish=1.046846,
            memory_gb=65.6,
            peak_vram_mb=33800.4,
            status=Status.keep,
            training_steps=2911,
            mfu_percent=37.98,
            tokens_trained_m=381.6,
            seed=42,
            batch="20260313-183842",
            run_name="a-h200-baseline-s42",
            agent_id="phoenix",
            gpu="H200",
            change_summary="token shifting (1/4 channels shifted by 1 position)",
            category=Category.architecture,
        )
        assert e.val_bpb == 0.968636
        assert e.agent_id == "phoenix"
        assert e.gpu == "H200"
        assert e.mfu_percent == 37.98

    def test_crash_status(self):
        e = Experiment(experiment_id=5, commit="abc1234", val_bpb=0.0, status=Status.crash)
        assert e.status == Status.crash

    def test_defaults(self):
        e = Experiment(experiment_id=1, commit="abc1234", val_bpb=0.97, status=Status.keep)
        assert e.seed == 42
        assert e.agent_id == "unknown"
        assert e.training_steps == 0
        assert e.tags == []

    def test_commit_min_length(self):
        with pytest.raises(Exception):
            Experiment(experiment_id=1, commit="abc", val_bpb=0.97, status=Status.keep)


class TestTechnique:
    def test_create(self):
        t = Technique(name="token shifting", category=Category.architecture)
        assert t.name == "token shifting"
        assert t.success_rate == 0.0

    def test_with_success_rate(self):
        t = Technique(name="weight decay", category=Category.regularization, success_rate=0.45)
        assert t.success_rate == 0.45


class TestHypothesis:
    def test_create(self):
        h = Hypothesis(text="shifting tokens should help because it adds local context")
        assert h.debate_rounds == 0
        assert h.challenger_agreed is False

    def test_debated(self):
        h = Hypothesis(
            text="try higher lr",
            debate_rounds=2,
            challenger_agreed=True,
            winning_argument="worked 3 times before",
        )
        assert h.debate_rounds == 2
        assert h.challenger_agreed is True


class TestAgent:
    def test_create(self):
        a = Agent(name="phoenix", gpu="H200")
        assert a.name == "phoenix"
        assert a.best_val_bpb == float("inf")
        assert a.total_runs == 0

    def test_with_stats(self):
        a = Agent(name="phoenix", gpu="H200", total_runs=50, best_val_bpb=0.9686, keep_rate=0.12)
        assert a.total_runs == 50
        assert a.keep_rate == 0.12
