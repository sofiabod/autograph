# e2e tests — ingest results.tsv into memgraph and verify the full schema

import pytest
from pathlib import Path

from graph.db import MemgraphClient
from graph.ingest import ingest_tsv


RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "baseline" / "results.tsv"


@pytest.fixture
def client():
    c = MemgraphClient()
    c.drop_all()
    c.create_indexes()
    yield c
    c.drop_all()
    c.close()


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
class TestIngestTSV:
    def test_ingest_count(self, client):
        count = ingest_tsv(client, RESULTS_PATH)
        assert count > 200

    def test_nodes_created(self, client):
        ingest_tsv(client, RESULTS_PATH)
        counts = client.count_nodes()
        assert counts.get("Experiment", 0) > 200
        assert counts.get("Technique", 0) > 0
        assert counts.get("Result", 0) > 200  # one result per experiment
        assert counts.get("Run", 0) == 1  # one run node

    def test_best_experiment(self, client):
        ingest_tsv(client, RESULTS_PATH)
        best = client.get_best()
        assert best is not None
        assert best["val_bpb"] < 0.97

    def test_keeps_exist(self, client):
        ingest_tsv(client, RESULTS_PATH)
        keeps = client.get_keeps()
        assert len(keeps) > 10

    def test_edges_created(self, client):
        ingest_tsv(client, RESULTS_PATH)
        neighbors = client.get_neighbors(1)
        assert len(neighbors) > 0

    def test_techniques_extracted(self, client):
        ingest_tsv(client, RESULTS_PATH)
        techs = client.get_techniques()
        names = [t["name"] for t in techs]
        assert "weight decay" in names

    def test_technique_stats(self, client):
        ingest_tsv(client, RESULTS_PATH)
        stats = client.get_technique_stats("weight decay")
        assert sum(stats.values()) > 0

    def test_run_created(self, client):
        ingest_tsv(client, RESULTS_PATH, run_name="test-run")
        exps = client.get_run_experiments("test-run")
        assert len(exps) > 200

    def test_results_produced(self, client):
        ingest_tsv(client, RESULTS_PATH)
        # first experiment should have a result
        results = client.get_results_for_experiment(1)
        assert len(results) == 1

    def test_contradictions_detected(self, client):
        ingest_tsv(client, RESULTS_PATH)
        contradictions = client.get_contradictions()
        # weight decay has both keeps and discards, should produce at least one contradiction
        assert len(contradictions) > 0

    def test_category_history(self, client):
        ingest_tsv(client, RESULTS_PATH)
        history = client.get_category_history("attention")
        assert len(history) > 0

    def test_search(self, client):
        ingest_tsv(client, RESULTS_PATH)
        results = client.search_experiments("weight decay")
        assert len(results) > 5
