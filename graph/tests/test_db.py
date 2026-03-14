"""integration tests — require running memgraph on bolt://localhost:7687."""

import pytest
from uuid import uuid4

from graph.db import MemgraphClient
from graph.models.nodes import Experiment, Technique, Hypothesis, Agent
from graph.models.edges import ImprovedFrom, FailedFrom, Tried, LedTo, RanBy
from graph.models.types import Status, Category


@pytest.fixture
def client():
    c = MemgraphClient()
    c.drop_all()
    c.create_indexes()
    yield c
    c.drop_all()
    c.close()


@pytest.fixture
def sample_experiments():
    baseline = Experiment(
        experiment_id=1, commit="5869330", val_bpb=0.996, status=Status.keep,
        change_summary="baseline", category=Category.hyperparameter,
    )
    improved = Experiment(
        experiment_id=2, commit="a34bf8c", val_bpb=0.994, status=Status.keep,
        change_summary="increase matrix lr", category=Category.schedule,
    )
    failed = Experiment(
        experiment_id=3, commit="66b06c7", val_bpb=0.997, status=Status.discard,
        change_summary="increase matrix lr too much", category=Category.schedule,
    )
    return baseline, improved, failed


class TestCreateNodes:
    def test_create_experiment(self, client, sample_experiments):
        baseline, _, _ = sample_experiments
        node_id = client.create_experiment(baseline)
        assert node_id == str(baseline.id)

    def test_create_technique(self, client):
        tech = Technique(name="weight decay", category=Category.regularization)
        node_id = client.create_technique(tech)
        assert node_id == str(tech.id)

    def test_create_technique_merge(self, client):
        """creating same technique twice should not duplicate."""
        t1 = Technique(name="rope", category=Category.attention)
        t2 = Technique(name="rope", category=Category.attention)
        client.create_technique(t1)
        client.create_technique(t2)
        techs = client.get_techniques()
        rope_count = sum(1 for t in techs if t["name"] == "rope")
        assert rope_count == 1

    def test_create_agent(self, client):
        agent = Agent(name="phoenix", gpu="H200", total_runs=50, best_val_bpb=0.968)
        node_id = client.create_agent(agent)
        assert node_id == str(agent.id)

    def test_create_hypothesis(self, client):
        hyp = Hypothesis(text="higher lr should help convergence")
        node_id = client.create_hypothesis(hyp)
        assert node_id == str(hyp.id)


class TestCreateEdges:
    def test_improved_from(self, client, sample_experiments):
        baseline, improved, _ = sample_experiments
        client.create_experiment(baseline)
        client.create_experiment(improved)
        result = client.create_edge(ImprovedFrom(
            source=improved.id, target=baseline.id, delta_val_bpb=-0.002,
        ))
        assert result[0]["edge_type"] == "IMPROVED_FROM"

    def test_failed_from(self, client, sample_experiments):
        baseline, _, failed = sample_experiments
        client.create_experiment(baseline)
        client.create_experiment(failed)
        result = client.create_edge(FailedFrom(
            source=failed.id, target=baseline.id, delta_val_bpb=0.001,
        ))
        assert result[0]["edge_type"] == "FAILED_FROM"

    def test_tried(self, client, sample_experiments):
        baseline, _, _ = sample_experiments
        tech = Technique(name="weight decay", category=Category.regularization)
        client.create_experiment(baseline)
        client.create_technique(tech)
        result = client.create_edge(Tried(source=baseline.id, target=tech.id))
        assert result[0]["edge_type"] == "TRIED"

    def test_led_to(self, client, sample_experiments):
        baseline, improved, _ = sample_experiments
        client.create_experiment(baseline)
        client.create_experiment(improved)
        result = client.create_edge(LedTo(
            source=baseline.id, target=improved.id,
            rationale="lr was too low", next_idea="try higher lr",
        ))
        assert result[0]["edge_type"] == "LED_TO"


class TestQueries:
    def test_get_experiment(self, client, sample_experiments):
        baseline, _, _ = sample_experiments
        client.create_experiment(baseline)
        result = client.get_experiment(1)
        assert result is not None
        assert result["val_bpb"] == 0.996

    def test_get_experiment_not_found(self, client):
        assert client.get_experiment(999) is None

    def test_get_all_experiments(self, client, sample_experiments):
        for exp in sample_experiments:
            client.create_experiment(exp)
        exps = client.get_all_experiments()
        assert len(exps) == 3

    def test_get_keeps(self, client, sample_experiments):
        for exp in sample_experiments:
            client.create_experiment(exp)
        keeps = client.get_keeps()
        assert len(keeps) == 2
        assert keeps[0]["val_bpb"] <= keeps[1]["val_bpb"]

    def test_get_best(self, client, sample_experiments):
        for exp in sample_experiments:
            client.create_experiment(exp)
        best = client.get_best()
        assert best["val_bpb"] == 0.994

    def test_get_neighbors(self, client, sample_experiments):
        baseline, improved, _ = sample_experiments
        client.create_experiment(baseline)
        client.create_experiment(improved)
        client.create_edge(ImprovedFrom(
            source=improved.id, target=baseline.id, delta_val_bpb=-0.002,
        ))
        neighbors = client.get_neighbors(1)
        assert len(neighbors) == 1
        assert neighbors[0]["edge_type"] == "IMPROVED_FROM"

    def test_get_technique_stats(self, client, sample_experiments):
        baseline, improved, failed = sample_experiments
        tech = Technique(name="matrix lr", category=Category.schedule)
        for exp in sample_experiments:
            client.create_experiment(exp)
        client.create_technique(tech)
        client.create_edge(Tried(source=improved.id, target=tech.id))
        client.create_edge(Tried(source=failed.id, target=tech.id))
        stats = client.get_technique_stats("matrix lr")
        assert stats["keep"] == 1
        assert stats["discard"] == 1

    def test_count_nodes(self, client, sample_experiments):
        for exp in sample_experiments:
            client.create_experiment(exp)
        counts = client.count_nodes()
        assert counts["Experiment"] == 3

    def test_experiment_chain(self, client):
        e1 = Experiment(
            experiment_id=1, commit="aaa1111", val_bpb=0.99, status=Status.keep,
            change_summary="baseline", category=Category.hyperparameter,
        )
        e2 = Experiment(
            experiment_id=2, commit="bbb2222", val_bpb=0.98, status=Status.keep,
            change_summary="step 2", category=Category.hyperparameter,
        )
        e3 = Experiment(
            experiment_id=3, commit="ccc3333", val_bpb=0.97, status=Status.keep,
            change_summary="step 3", category=Category.hyperparameter,
        )
        client.create_experiment(e1)
        client.create_experiment(e2)
        client.create_experiment(e3)
        client.create_edge(ImprovedFrom(source=e2.id, target=e1.id, delta_val_bpb=-0.01))
        client.create_edge(ImprovedFrom(source=e3.id, target=e2.id, delta_val_bpb=-0.01))
        chain = client.get_experiment_chain(3)
        assert len(chain) >= 2
