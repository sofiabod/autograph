import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from expgraph.api import app

client = TestClient(app)

SAMPLE = [
    {
        "experiment_id": 0, "commit": "5869330",
        "timestamp": "2026-03-11T20:00:00", "parent_id": 0,
        "builds_on": [], "hypothesis": "baseline",
        "change_summary": "baseline", "category": "training_loop",
        "components": ["model"], "parameters_changed": {},
        "val_bpb": 0.996, "baseline_bpb": 0.996, "delta_bpb": 0.0,
        "memory_gb": 44.0, "status": "keep", "reasoning": "baseline",
        "insights": [], "contradicts": [], "supports": [],
        "tags": ["baseline"], "next_idea": "try lr",
    },
]


class TestHealthEndpoint:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestGraphEndpoint:
    def test_returns_nodes_and_links(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text(json.dumps(SAMPLE[0]))
        r = client.get("/graph", params={"path": str(p)})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "links" in data

    def test_experiment_in_response(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text(json.dumps(SAMPLE[0]))
        r = client.get("/graph", params={"path": str(p)})
        nodes = r.json()["nodes"]
        exp_nodes = [n for n in nodes if n["type"] == "experiment"]
        assert len(exp_nodes) == 1
        assert exp_nodes[0]["commit"] == "5869330"

    def test_default_path_if_exists(self):
        """hits default experiment_log.jsonl if it exists."""
        default = Path(__file__).parents[2] / "experiment_log.jsonl"
        if not default.exists():
            return
        r = client.get("/graph")
        assert r.status_code == 200
        assert len(r.json()["nodes"]) > 0

    def test_bad_path_returns_error(self):
        r = client.get("/graph", params={"path": "/nonexistent/file.jsonl"})
        assert r.status_code == 404
