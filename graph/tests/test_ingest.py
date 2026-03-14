"""e2e tests — ingest results.tsv into memgraph and verify."""

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


class TestIngestTSV:
    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_ingest_count(self, client):
        count = ingest_tsv(client, RESULTS_PATH)
        assert count > 200

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_nodes_created(self, client):
        ingest_tsv(client, RESULTS_PATH)
        counts = client.count_nodes()
        assert counts.get("Experiment", 0) > 200
        assert counts.get("Technique", 0) > 0

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_best_experiment(self, client):
        ingest_tsv(client, RESULTS_PATH)
        best = client.get_best()
        assert best is not None
        assert best["val_bpb"] < 0.97

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_keeps_exist(self, client):
        ingest_tsv(client, RESULTS_PATH)
        keeps = client.get_keeps()
        assert len(keeps) > 10

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_edges_created(self, client):
        ingest_tsv(client, RESULTS_PATH)
        # baseline (exp 1) should have neighbors
        neighbors = client.get_neighbors(1)
        assert len(neighbors) > 0

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_techniques_extracted(self, client):
        ingest_tsv(client, RESULTS_PATH)
        techs = client.get_techniques()
        names = [t["name"] for t in techs]
        assert "weight decay" in names

    @pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
    def test_technique_stats(self, client):
        ingest_tsv(client, RESULTS_PATH)
        stats = client.get_technique_stats("weight decay")
        assert sum(stats.values()) > 0
