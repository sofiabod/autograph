import json
import tempfile
from pathlib import Path

from expgraph.ingest import ingest_jsonl


SAMPLE = [
    {
        "experiment_id": 0, "commit": "5869330",
        "timestamp": "2026-03-11T20:00:00", "parent_id": 0,
        "builds_on": [], "hypothesis": "baseline",
        "change_summary": "baseline", "category": "training_loop",
        "components": ["model", "optimizer"], "parameters_changed": {},
        "val_bpb": 0.996, "baseline_bpb": 0.996, "delta_bpb": 0.0,
        "memory_gb": 44.0, "status": "keep", "reasoning": "baseline run",
        "insights": [], "contradicts": [], "supports": [],
        "tags": ["baseline"], "next_idea": "try higher lr",
    },
    {
        "experiment_id": 1, "commit": "ced454d",
        "timestamp": "2026-03-11T20:08:00", "parent_id": 0,
        "builds_on": [0], "hypothesis": "higher lr helps",
        "change_summary": "increase lr", "category": "hyperparameter",
        "components": ["optimizer", "learning_rate"], "parameters_changed": {},
        "val_bpb": 0.994, "baseline_bpb": 0.996, "delta_bpb": -0.002,
        "memory_gb": 44.0, "status": "keep", "reasoning": "lr was too low",
        "insights": [], "contradicts": [], "supports": [],
        "tags": ["learning_rate"], "next_idea": "try even higher",
    },
    {
        "experiment_id": 2, "commit": "a34bf8c",
        "timestamp": "2026-03-11T20:16:00", "parent_id": 1,
        "builds_on": [1], "hypothesis": "even higher lr",
        "change_summary": "lr too high", "category": "hyperparameter",
        "components": ["optimizer", "learning_rate"], "parameters_changed": {},
        "val_bpb": 1.001, "baseline_bpb": 0.994, "delta_bpb": 0.007,
        "memory_gb": 44.0, "status": "discard", "reasoning": "overshot",
        "insights": [], "contradicts": [], "supports": [],
        "tags": ["learning_rate"], "next_idea": "",
    },
]


def _write_sample(tmp_path: Path) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in SAMPLE))
    return p


class TestIngestJsonl:
    def test_returns_nodes_and_links(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        assert "nodes" in result
        assert "links" in result

    def test_experiment_node_count(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        exp_nodes = [n for n in result["nodes"] if n["type"] == "experiment"]
        assert len(exp_nodes) == 3

    def test_technique_nodes_deduplicated(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        tech_nodes = [n for n in result["nodes"] if n["type"] == "technique"]
        names = {t["name"] for t in tech_nodes}
        # model, optimizer, baseline, learning_rate from components + tags
        assert "optimizer" in names
        assert "learning_rate" in names

    def test_improved_from_edge(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        improved = [l for l in result["links"] if l["edge_type"] == "IMPROVED_FROM"]
        assert len(improved) == 1
        assert improved[0]["delta_val_bpb"] == -0.002

    def test_failed_from_edge(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        failed = [l for l in result["links"] if l["edge_type"] == "FAILED_FROM"]
        assert len(failed) == 1
        assert failed[0]["delta_val_bpb"] == 0.007

    def test_led_to_edges(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        led_to = [l for l in result["links"] if l["edge_type"] == "LED_TO"]
        # exp 0 -> exp 1 (has next_idea), exp 1 -> exp 2 (has next_idea)
        assert len(led_to) == 2
        assert led_to[0]["next_idea"] == "try higher lr"

    def test_tried_edges(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        tried = [l for l in result["links"] if l["edge_type"] == "TRIED"]
        assert len(tried) > 0

    def test_node_ids_are_strings(self, tmp_path):
        result = ingest_jsonl(_write_sample(tmp_path))
        for node in result["nodes"]:
            assert isinstance(node["id"], str)

    def test_real_file(self):
        """run against the actual experiment_log.jsonl if it exists."""
        real = Path(__file__).parents[2] / "experiment_log.jsonl"
        if not real.exists():
            return
        result = ingest_jsonl(real)
        assert len(result["nodes"]) > 0
        assert len(result["links"]) > 0
