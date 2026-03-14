# integration tests for the memgraph-backed api
# requires running memgraph on bolt://localhost:7687

import pytest
from fastapi.testclient import TestClient

from graph.api import app
from graph.db import MemgraphClient
from graph.ingest import ingest_tsv
from pathlib import Path


RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "baseline" / "results.tsv"


@pytest.fixture
def api_client():
    # ingest data before testing api
    c = MemgraphClient()
    c.drop_all()
    c.create_indexes()
    if RESULTS_PATH.exists():
        ingest_tsv(c, RESULTS_PATH)
    c.close()
    yield TestClient(app)
    # cleanup
    c = MemgraphClient()
    c.drop_all()
    c.close()


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
class TestGraphEndpoint:
    def test_returns_nodes_and_links(self, api_client):
        r = api_client.get("/graph")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) > 200

    def test_has_experiment_nodes(self, api_client):
        r = api_client.get("/graph")
        data = r.json()
        experiments = [n for n in data["nodes"] if n["type"] == "experiment"]
        assert len(experiments) > 200

    def test_has_technique_nodes(self, api_client):
        r = api_client.get("/graph")
        data = r.json()
        techniques = [n for n in data["nodes"] if n["type"] == "technique"]
        assert len(techniques) > 0

    def test_has_result_nodes(self, api_client):
        r = api_client.get("/graph")
        data = r.json()
        results = [n for n in data["nodes"] if n["type"] == "result"]
        assert len(results) > 200

    def test_has_edges(self, api_client):
        r = api_client.get("/graph")
        data = r.json()
        assert len(data["links"]) > 0
        edge_types = set(l["edge_type"] for l in data["links"])
        assert "IMPROVED_FROM" in edge_types or "FAILED_FROM" in edge_types

    def test_nodes_have_required_fields(self, api_client):
        r = api_client.get("/graph")
        data = r.json()
        for node in data["nodes"][:10]:
            assert "id" in node
            assert "type" in node


class TestStatsEndpoint:
    def test_returns_stats(self, api_client):
        r = api_client.get("/graph/stats")
        assert r.status_code == 200
        data = r.json()
        assert "node_counts" in data
        assert "best" in data
        assert "contradictions" in data

    def test_best_experiment(self, api_client):
        r = api_client.get("/graph/stats")
        data = r.json()
        if data["best"]:
            assert data["best"]["val_bpb"] < 0.97


class TestLineageEndpoint:
    def test_returns_chain(self, api_client):
        r = api_client.get("/graph/lineage/224")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 10  # should have full chain


class TestDebatesEndpoint:
    def test_returns_empty_list(self, api_client):
        r = api_client.get("/graph/debates")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestHealthEndpoint:
    def test_health(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "nodes" in data
