from datetime import datetime
from uuid import UUID

import pytest
from expgraph.nodes import Experiment, Technique, Hypothesis
from expgraph.types import Category, Status


class TestExperiment:
    def test_create_minimal(self):
        exp = Experiment(
            experiment_id=0,
            commit="5869330",
            val_bpb=0.996,
            baseline_bpb=0.996,
            delta_bpb=0.0,
            memory_gb=44.0,
            status=Status.keep,
            category=Category.training_loop,
        )
        assert isinstance(exp.id, UUID)
        assert exp.experiment_id == 0
        assert exp.commit == "5869330"
        assert exp.user_id == "default"
        assert exp.tags == []
        assert exp.builds_on == []
        assert exp.agent_id == "unknown"
        assert exp.repo_url == ""
        assert exp.branch == ""
        assert exp.commit_url == ""
        assert exp.completed_at is None

    def test_create_full(self):
        exp = Experiment(
            experiment_id=1,
            commit="ced454d",
            val_bpb=1.002905,
            baseline_bpb=0.996003,
            delta_bpb=0.006902,
            memory_gb=66.5,
            status=Status.discard,
            category=Category.architecture,
            hypothesis="deeper model should improve quality",
            change_summary="increase depth 8 -> 10",
            reasoning="deeper model used 50% more memory",
            insights=["adding depth can hurt"],
            tags=["depth", "scaling"],
            parent_id=0,
            builds_on=[0],
            components=["model", "depth"],
            parameters_changed={"N_LAYERS": {"from": 8, "to": 10}},
            timestamp=datetime(2026, 3, 11, 20, 8),
            agent_id="claude-opus",
            repo_url="https://github.com/sofiabod/autoresearch",
            branch="autoresearch/mar11",
            commit_url="https://github.com/sofiabod/autoresearch/commit/ced454d",
            completed_at=datetime(2026, 3, 11, 20, 13),
        )
        assert exp.status == Status.discard
        assert exp.agent_id == "claude-opus"
        assert exp.branch == "autoresearch/mar11"
        assert exp.parent_id == 0
        assert exp.parameters_changed["N_LAYERS"]["to"] == 10

    def test_commit_too_short(self):
        with pytest.raises(ValueError):
            Experiment(
                experiment_id=0,
                commit="abc",
                val_bpb=1.0,
                baseline_bpb=1.0,
                delta_bpb=0.0,
                memory_gb=44.0,
                status=Status.keep,
                category=Category.training_loop,
            )

    def test_status_enum(self):
        for s in ["keep", "discard", "crash"]:
            exp = Experiment(
                experiment_id=0,
                commit="5869330",
                val_bpb=0.0,
                baseline_bpb=0.0,
                delta_bpb=0.0,
                memory_gb=0.0,
                status=s,
                category=Category.training_loop,
            )
            assert exp.status == Status(s)


class TestTechnique:
    def test_create(self):
        t = Technique(name="rope", category=Category.attention)
        assert isinstance(t.id, UUID)
        assert t.success_rate == 0.0
        assert t.user_id == "default"

    def test_with_stats(self):
        t = Technique(
            name="muon",
            original_name="Muon optimizer",
            category=Category.optimizer,
            success_rate=0.75,
        )
        assert t.success_rate == 0.75
        assert t.original_name == "Muon optimizer"


class TestHypothesis:
    def test_create(self):
        h = Hypothesis(text="higher lr improves convergence")
        assert isinstance(h.id, UUID)
        assert h.debate_rounds == 0
        assert h.challenger_agreed is False

    def test_with_debate(self):
        h = Hypothesis(
            text="remove sliding window",
            debate_rounds=3,
            challenger_agreed=True,
            winning_argument="attention patterns show full context helps",
        )
        assert h.debate_rounds == 3
        assert h.challenger_agreed is True
