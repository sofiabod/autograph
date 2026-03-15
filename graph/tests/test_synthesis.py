# tests for synthesis nodes and auto-detection

from uuid import UUID

import pytest

from graph.models.nodes import Synthesis
from graph.models.edges import SynthesizedFrom, SupportsSynthesis, ContradictsSynthesis
from graph.models.types import Category, EdgeType
from graph.db import MemgraphClient
from graph.ingest import ingest_tsv
from graph.synthesize import run_synthesis
from pathlib import Path


RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "baseline" / "results.tsv"


# unit tests

class TestSynthesisModel:
    def test_create(self):
        s = Synthesis(text="weight decay has a sweet spot", confidence=0.7)
        assert isinstance(s.id, UUID)
        assert s.auto_generated is False
        assert s.domain == "ml"

    def test_auto_generated(self):
        s = Synthesis(text="test", confidence=0.5, auto_generated=True, source_count=10)
        assert s.auto_generated is True
        assert s.source_count == 10

    def test_with_category(self):
        s = Synthesis(text="test", confidence=0.5, category=Category.regularization)
        assert s.category == Category.regularization


class TestSynthesisEdges:
    def test_synthesized_from(self):
        from uuid import uuid4
        e = SynthesizedFrom(source=uuid4(), target=uuid4())
        assert e.edge_type == EdgeType.synthesized_from

    def test_supports_synthesis(self):
        from uuid import uuid4
        e = SupportsSynthesis(source=uuid4(), target=uuid4())
        assert e.edge_type == EdgeType.supports_synthesis

    def test_contradicts_synthesis(self):
        from uuid import uuid4
        e = ContradictsSynthesis(source=uuid4(), target=uuid4(), explanation="doesn't hold for depth > 9")
        assert e.edge_type == EdgeType.contradicts_synthesis
        assert "depth" in e.explanation


# integration tests

@pytest.fixture
def client():
    c = MemgraphClient()
    c.drop_all()
    c.create_indexes()
    yield c
    c.drop_all()
    c.close()


class TestSynthesisDB:
    def test_create_synthesis(self, client):
        s = Synthesis(text="test synthesis", confidence=0.8, domain="ml")
        sid = client.create_synthesis(s)
        assert sid == str(s.id)

    def test_get_syntheses(self, client):
        client.create_synthesis(Synthesis(text="synth 1", confidence=0.9))
        client.create_synthesis(Synthesis(text="synth 2", confidence=0.5))
        results = client.get_syntheses()
        assert len(results) == 2
        assert results[0]["confidence"] >= results[1]["confidence"]

    def test_get_syntheses_by_domain(self, client):
        client.create_synthesis(Synthesis(text="ml thing", confidence=0.5, domain="ml"))
        client.create_synthesis(Synthesis(text="finance thing", confidence=0.5, domain="finance"))
        ml = client.get_syntheses(domain="ml")
        assert len(ml) == 1
        assert ml[0]["domain"] == "ml"


# e2e synthesis detection

@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="results.tsv not found")
class TestAutoSynthesis:
    def test_detects_patterns(self, client):
        ingest_tsv(client, RESULTS_PATH)
        count = run_synthesis(client)
        assert count > 0

    def test_creates_synthesis_nodes(self, client):
        ingest_tsv(client, RESULTS_PATH)
        run_synthesis(client)
        syntheses = client.get_syntheses()
        assert len(syntheses) > 0

    def test_no_duplicates_on_rerun(self, client):
        ingest_tsv(client, RESULTS_PATH)
        count1 = run_synthesis(client)
        count2 = run_synthesis(client)
        assert count2 == 0  # no new ones on second run

    def test_syntheses_have_evidence(self, client):
        ingest_tsv(client, RESULTS_PATH)
        run_synthesis(client)
        syntheses = client.get_syntheses()
        # at least one synthesis should have supporting experiments
        has_evidence = False
        for s in syntheses:
            evidence = client.get_synthesis_evidence(s["id"])
            if evidence:
                has_evidence = True
                break
        assert has_evidence

    def test_detects_productive_directions(self, client):
        ingest_tsv(client, RESULTS_PATH)
        run_synthesis(client)
        syntheses = client.get_syntheses()
        productive = [s for s in syntheses if "productive" in s["text"]]
        assert len(productive) > 0

    def test_detects_crashes(self, client):
        ingest_tsv(client, RESULTS_PATH)
        run_synthesis(client)
        syntheses = client.get_syntheses()
        crash_syntheses = [s for s in syntheses if "crashed" in s["text"]]
        assert len(crash_syntheses) > 0
