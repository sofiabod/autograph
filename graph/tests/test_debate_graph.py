# integration tests — debate writes to memgraph correctly
# requires running memgraph on bolt://localhost:7687

import pytest
from datetime import UTC, datetime

from graph.db import MemgraphClient
from graph.models.nodes import Hypothesis, Agent
from graph.models.types import HypothesisStatus, Category
from graph.debate.debate import (
    _write_agents_to_graph,
    _write_hypothesis_to_graph,
    _write_challenge_to_graph,
    _write_refinement_to_graph,
    _update_hypothesis_status,
    _write_debate_log_to_graph,
    _guess_category,
    DebateResult,
    DebateRound,
)
from graph.debate.config import DebateConfig


@pytest.fixture
def client():
    c = MemgraphClient()
    c.drop_all()
    c.create_indexes()
    yield c
    c.drop_all()
    c.close()


class TestWriteAgents:
    def test_creates_two_agents(self, client):
        config = DebateConfig(api_key="test")
        pid, cid = _write_agents_to_graph(client, config)
        assert pid != ""
        assert cid != ""
        counts = client.count_nodes()
        assert counts.get("Agent", 0) == 2

    def test_merge_existing_agents(self, client):
        config = DebateConfig(api_key="test")
        _write_agents_to_graph(client, config)
        _write_agents_to_graph(client, config)
        # should not duplicate
        counts = client.count_nodes()
        assert counts.get("Agent", 0) == 2


class TestWriteHypothesis:
    def test_creates_hypothesis_node(self, client):
        hyp = _write_hypothesis_to_graph(client, "try higher lr", "schedule")
        assert hyp.text == "try higher lr"
        assert hyp.status == HypothesisStatus.pending
        counts = client.count_nodes()
        assert counts.get("Hypothesis", 0) == 1

    def test_hypothesis_with_category(self, client):
        hyp = _write_hypothesis_to_graph(client, "add dropout", "regularization")
        assert hyp.category == Category.regularization

    def test_hypothesis_has_valid_from(self, client):
        hyp = _write_hypothesis_to_graph(client, "test idea", "hyperparameter")
        assert hyp.valid_from is not None

    def test_invalid_category_falls_back(self, client):
        hyp = _write_hypothesis_to_graph(client, "something weird", "nonexistent_category")
        assert hyp.category == Category.hyperparameter


class TestWriteChallenge:
    def test_creates_challenged_edge(self, client):
        config = DebateConfig(api_key="test")
        _, cid = _write_agents_to_graph(client, config)
        hyp = _write_hypothesis_to_graph(client, "try x", "hyperparameter")
        _write_challenge_to_graph(client, cid, hyp, "x already failed in exp #45", round_num=1)
        # check edge exists
        rows = client._run("MATCH (a:Agent)-[r:CHALLENGED]->(h:Hypothesis) RETURN r.reason AS reason, r.round AS round")
        assert len(rows) == 1
        assert "exp #45" in rows[0]["reason"]
        assert rows[0]["round"] == 1

    def test_multiple_rounds(self, client):
        config = DebateConfig(api_key="test")
        _, cid = _write_agents_to_graph(client, config)
        hyp = _write_hypothesis_to_graph(client, "try x", "hyperparameter")
        _write_challenge_to_graph(client, cid, hyp, "round 1 objection", round_num=1)
        _write_challenge_to_graph(client, cid, hyp, "round 2 still not convinced", round_num=2)
        rows = client._run("MATCH ()-[r:CHALLENGED]->() RETURN r.round AS round ORDER BY r.round")
        assert len(rows) == 2
        assert rows[0]["round"] == 1
        assert rows[1]["round"] == 2


class TestWriteRefinement:
    def test_creates_refines_edge(self, client):
        h1 = _write_hypothesis_to_graph(client, "try lr 0.07", "schedule")
        h2 = _write_hypothesis_to_graph(client, "try lr 0.065 instead", "schedule")
        _write_refinement_to_graph(client, h1, h2, "reduced lr based on challenge", "challenger cited exp #70")
        rows = client._run("MATCH (a:Hypothesis)-[r:REFINES]->(b:Hypothesis) RETURN r.what_changed AS wc, r.why AS why")
        assert len(rows) == 1
        assert "reduced lr" in rows[0]["wc"]

    def test_refinement_chain(self, client):
        h1 = _write_hypothesis_to_graph(client, "idea v1", "hyperparameter")
        h2 = _write_hypothesis_to_graph(client, "idea v2", "hyperparameter")
        h3 = _write_hypothesis_to_graph(client, "idea v3", "hyperparameter")
        _write_refinement_to_graph(client, h1, h2, "v1 to v2", "round 1")
        _write_refinement_to_graph(client, h2, h3, "v2 to v3", "round 2")
        # should be able to trace the chain
        rows = client._run("""
            MATCH path = (h:Hypothesis {text: 'idea v3'})-[:REFINES*]->(ancestor)
            RETURN length(path) AS depth
        """)
        assert any(r["depth"] >= 2 for r in rows)


class TestUpdateHypothesisStatus:
    def test_confirm(self, client):
        hyp = _write_hypothesis_to_graph(client, "good idea", "hyperparameter")
        _update_hypothesis_status(client, hyp, "confirmed")
        rows = client._run("MATCH (h:Hypothesis {id: $id}) RETURN h.status AS status", {"id": str(hyp.id)})
        assert rows[0]["status"] == "confirmed"

    def test_reject(self, client):
        hyp = _write_hypothesis_to_graph(client, "bad idea", "hyperparameter")
        _update_hypothesis_status(client, hyp, "rejected")
        rows = client._run("MATCH (h:Hypothesis {id: $id}) RETURN h.status AS status", {"id": str(hyp.id)})
        assert rows[0]["status"] == "rejected"

    def test_sets_valid_until(self, client):
        hyp = _write_hypothesis_to_graph(client, "idea", "hyperparameter")
        _update_hypothesis_status(client, hyp, "confirmed")
        rows = client._run("MATCH (h:Hypothesis {id: $id}) RETURN h.valid_until AS vu", {"id": str(hyp.id)})
        assert rows[0]["vu"] is not None


class TestWriteDebateLog:
    def test_creates_debate_log_node(self, client):
        result = DebateResult()
        result.hypothesis_ids = ["abc-123"]
        result.rounds = [DebateRound(
            round_num=1,
            proposal="try lr 0.07",
            challenge="already tried",
            rebuttal="conditions changed",
            assessment="APPROVE",
        )]
        result.decision = {
            "run": True,
            "hypothesis": "try lr 0.07",
            "change_summary": "increase matrix lr 0.06 -> 0.07",
            "confidence": 0.7,
            "reasoning": "evidence supports this",
        }
        _write_debate_log_to_graph(client, result)
        rows = client._run("MATCH (d:DebateLog) RETURN d")
        assert len(rows) == 1
        log = dict(rows[0]["d"])
        assert log["run_decision"] is True
        assert log["confidence"] == 0.7


class TestFullDebatePipeline:
    def test_full_write_pipeline(self, client):
        # simulate a full debate write without api calls
        config = DebateConfig(api_key="test")
        pid, cid = _write_agents_to_graph(client, config)

        # round 1
        h1 = _write_hypothesis_to_graph(client, "increase depth to 10", "architecture")
        _write_challenge_to_graph(client, cid, h1, "depth 10 OOMed in exp #26", round_num=1)

        # proposer modifies
        h2 = _write_hypothesis_to_graph(client, "increase depth to 9 with smaller batch", "architecture")
        _write_refinement_to_graph(client, h1, h2, "reduced depth target", "OOM risk at depth 10")

        # round 2
        _write_challenge_to_graph(client, cid, h2, "APPROVE — depth 9 worked before in exp #16", round_num=2)
        _update_hypothesis_status(client, h1, "rejected")
        _update_hypothesis_status(client, h2, "confirmed")

        # write debate log
        result = DebateResult()
        result.hypothesis_ids = [str(h1.id), str(h2.id)]
        result.rounds = [
            DebateRound(round_num=1, proposal="depth 10", challenge="OOM risk", rebuttal="depth 9 instead", assessment="APPROVE"),
        ]
        result.decision = {"run": True, "hypothesis": "depth 9 with smaller batch", "change_summary": "depth 8 -> 9, batch 2^18 -> 2^17", "confidence": 0.8, "reasoning": "depth 9 worked before"}
        _write_debate_log_to_graph(client, result)

        # verify full graph state
        counts = client.count_nodes()
        assert counts.get("Agent", 0) == 2
        assert counts.get("Hypothesis", 0) == 2
        assert counts.get("DebateLog", 0) == 1

        # verify edges
        challenged = client._run("MATCH ()-[r:CHALLENGED]->() RETURN count(r) AS cnt")
        assert challenged[0]["cnt"] == 2

        refines = client._run("MATCH ()-[r:REFINES]->() RETURN count(r) AS cnt")
        assert refines[0]["cnt"] == 1

        # verify statuses
        h1_status = client._run("MATCH (h:Hypothesis {id: $id}) RETURN h.status AS s", {"id": str(h1.id)})
        assert h1_status[0]["s"] == "rejected"

        h2_status = client._run("MATCH (h:Hypothesis {id: $id}) RETURN h.status AS s", {"id": str(h2.id)})
        assert h2_status[0]["s"] == "confirmed"
