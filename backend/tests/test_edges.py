from uuid import uuid4

import pytest
from expgraph.edges import (
    Tried,
    ImprovedFrom,
    FailedFrom,
    MotivatedBy,
    Invalidates,
    CorrelatedWith,
    LedTo,
)
from expgraph.types import EdgeType


SRC = uuid4()
TGT = uuid4()


class TestTried:
    def test_create(self):
        e = Tried(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.tried
        assert e.was_novel is False

    def test_novel(self):
        e = Tried(source=SRC, target=TGT, was_novel=True, fact="first use of rope")
        assert e.was_novel is True
        assert e.fact == "first use of rope"


class TestImprovedFrom:
    def test_create(self):
        e = ImprovedFrom(source=SRC, target=TGT, delta_val_bpb=-0.002)
        assert e.edge_type == EdgeType.improved_from
        assert e.delta_val_bpb == -0.002


class TestFailedFrom:
    def test_create(self):
        e = FailedFrom(source=SRC, target=TGT, delta_val_bpb=0.005)
        assert e.edge_type == EdgeType.failed_from
        assert e.delta_val_bpb == 0.005


class TestMotivatedBy:
    def test_create(self):
        e = MotivatedBy(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.motivated_by


class TestInvalidates:
    def test_create(self):
        e = Invalidates(
            source=SRC, target=TGT, condition="only works when lr < 1e-3"
        )
        assert e.edge_type == EdgeType.invalidates
        assert e.condition == "only works when lr < 1e-3"


class TestCorrelatedWith:
    def test_create(self):
        e = CorrelatedWith(source=SRC, target=TGT, joint_success_rate=0.8)
        assert e.edge_type == EdgeType.correlated_with
        assert e.joint_success_rate == 0.8


class TestLedTo:
    def test_create(self):
        e = LedTo(
            source=SRC,
            target=TGT,
            rationale="higher lr worked so try pushing further",
            next_idea="try matrix lr 0.08",
        )
        assert e.edge_type == EdgeType.led_to
        assert e.rationale == "higher lr worked so try pushing further"
        assert e.next_idea == "try matrix lr 0.08"


class TestBaseEdgeFields:
    def test_all_edges_have_source_target(self):
        """every edge type stores source and target uuids."""
        for cls in [Tried, ImprovedFrom, FailedFrom, MotivatedBy, Invalidates, CorrelatedWith, LedTo]:
            e = cls(source=SRC, target=TGT)
            assert e.source == SRC
            assert e.target == TGT
            assert e.user_id == "default"
