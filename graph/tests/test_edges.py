from uuid import uuid4

import pytest

from graph.models.edges import (
    Tried,
    ImprovedFrom,
    FailedFrom,
    MotivatedBy,
    Invalidates,
    CorrelatedWith,
    LedTo,
    RanBy,
    RanOn,
    BeatGlobal,
    BeatOwn,
)
from graph.models.types import EdgeType


SRC = uuid4()
TGT = uuid4()


class TestTried:
    def test_create(self):
        e = Tried(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.tried
        assert e.was_novel is False

    def test_novel(self):
        e = Tried(source=SRC, target=TGT, was_novel=True)
        assert e.was_novel is True


class TestImprovedFrom:
    def test_create(self):
        e = ImprovedFrom(source=SRC, target=TGT, delta_val_bpb=-0.002)
        assert e.edge_type == EdgeType.improved_from
        assert e.delta_val_bpb == -0.002


class TestFailedFrom:
    def test_create(self):
        e = FailedFrom(source=SRC, target=TGT, delta_val_bpb=0.005)
        assert e.edge_type == EdgeType.failed_from


class TestMotivatedBy:
    def test_create(self):
        e = MotivatedBy(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.motivated_by


class TestInvalidates:
    def test_create(self):
        e = Invalidates(source=SRC, target=TGT, condition="only below 64gb vram")
        assert e.condition == "only below 64gb vram"


class TestCorrelatedWith:
    def test_create(self):
        e = CorrelatedWith(source=SRC, target=TGT, joint_success_rate=0.8)
        assert e.joint_success_rate == 0.8


class TestLedTo:
    def test_create(self):
        e = LedTo(
            source=SRC,
            target=TGT,
            rationale="higher lr worked so try pushing further",
            next_idea="try matrix lr 0.08",
        )
        assert e.rationale == "higher lr worked so try pushing further"
        assert e.next_idea == "try matrix lr 0.08"


class TestRanBy:
    def test_create(self):
        e = RanBy(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.ran_by


class TestRanOn:
    def test_create(self):
        e = RanOn(source=SRC, target=TGT)
        assert e.edge_type == EdgeType.ran_on


class TestBeatGlobal:
    def test_create(self):
        e = BeatGlobal(source=SRC, target=TGT, improvement=0.0013)
        assert e.edge_type == EdgeType.beat_global
        assert e.improvement == 0.0013


class TestBeatOwn:
    def test_create(self):
        e = BeatOwn(source=SRC, target=TGT, improvement=0.086)
        assert e.edge_type == EdgeType.beat_own
        assert e.improvement == 0.086


class TestAllEdgesHaveSourceTarget:
    def test_shared_fields(self):
        for cls in [Tried, ImprovedFrom, FailedFrom, MotivatedBy, Invalidates, CorrelatedWith, LedTo, RanBy, RanOn, BeatGlobal, BeatOwn]:
            e = cls(source=SRC, target=TGT)
            assert e.source == SRC
            assert e.target == TGT
