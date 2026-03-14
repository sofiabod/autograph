# unit tests for debate models and helpers (no API calls)

from graph.debate.config import DebateConfig
from graph.debate.prompts import PROPOSER_SYSTEM, CHALLENGER_SYSTEM, PROPOSER_INITIAL, DECISION_PROMPT
from graph.debate.agents import Proposer, Challenger
from graph.debate.debate import _guess_category, DebateRound, DebateResult


class TestDebateConfig:
    def test_defaults(self):
        c = DebateConfig(api_key="test")
        assert c.proposer_model == "anthropic/claude-sonnet-4-20250514"
        assert c.challenger_model == "openai/gpt-4o"
        assert c.max_rounds == 2
        assert c.proposer_temperature == 0.7
        assert c.challenger_temperature == 0.3

    def test_custom_models(self):
        c = DebateConfig(
            proposer_model="anthropic/claude-3-haiku",
            challenger_model="google/gemini-pro",
            api_key="test",
        )
        assert c.proposer_model == "anthropic/claude-3-haiku"
        assert c.challenger_model == "google/gemini-pro"


class TestGuessCategory:
    def test_attention(self):
        assert _guess_category("increase number of attention heads") == "attention"

    def test_optimizer(self):
        assert _guess_category("change muon momentum") == "optimizer"

    def test_schedule(self):
        assert _guess_category("increase warmdown ratio") == "schedule"

    def test_architecture(self):
        assert _guess_category("add more depth layers") == "architecture"

    def test_regularization(self):
        assert _guess_category("increase weight decay") == "regularization"

    def test_default(self):
        assert _guess_category("try something random") == "hyperparameter"

    def test_token_shift(self):
        assert _guess_category("token shifting 1/4 channels") == "architecture"


class TestPrompts:
    def test_proposer_system_has_key_instructions(self):
        assert "hypothesis" in PROPOSER_SYSTEM
        assert "change_summary" in PROPOSER_SYSTEM
        assert "evidence" in PROPOSER_SYSTEM

    def test_challenger_system_has_key_instructions(self):
        assert "APPROVE" in CHALLENGER_SYSTEM
        assert "MODIFY" in CHALLENGER_SYSTEM
        assert "REJECT" in CHALLENGER_SYSTEM
        assert "stubborn" in CHALLENGER_SYSTEM

    def test_proposer_initial_has_placeholders(self):
        assert "{best}" in PROPOSER_INITIAL
        assert "{recent_keeps}" in PROPOSER_INITIAL
        assert "{contradictions}" in PROPOSER_INITIAL

    def test_decision_prompt_has_json_template(self):
        assert "run" in DECISION_PROMPT
        assert "hypothesis" in DECISION_PROMPT
        assert "confidence" in DECISION_PROMPT


class TestDebateRound:
    def test_create(self):
        r = DebateRound(round_num=1, proposal="try x", challenge="x already failed")
        assert r.round_num == 1
        assert r.rebuttal == ""

    def test_with_rebuttal(self):
        r = DebateRound(
            round_num=1,
            proposal="try x",
            challenge="x already failed",
            rebuttal="but conditions changed",
            assessment="APPROVE",
        )
        assert r.assessment == "APPROVE"


class TestDebateResult:
    def test_create(self):
        r = DebateResult()
        assert r.rounds == []
        assert r.decision == {}
        assert r.hypothesis_ids == []

    def test_with_rounds(self):
        r = DebateResult()
        r.rounds.append(DebateRound(round_num=1, proposal="x", challenge="y"))
        r.hypothesis_ids.append("abc-123")
        assert len(r.rounds) == 1
        assert r.hypothesis_ids[0] == "abc-123"


class TestAgentInit:
    def test_proposer_init(self):
        c = DebateConfig(api_key="test")
        p = Proposer(c)
        assert p.model == "anthropic/claude-sonnet-4-20250514"
        assert p.history == []

    def test_challenger_init(self):
        c = DebateConfig(api_key="test")
        ch = Challenger(c)
        assert ch.model == "openai/gpt-4o"
        assert ch.history == []
