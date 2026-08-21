#!/usr/bin/env python

"""Tests for task-specific semantic asset selection."""

import sys
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).parent / "quadloco_rsl_rl"
sys.path.insert(0, str(MODULE_ROOT))

from evaluation_assets import semantic_candidate_assets  # noqa: E402


def test_direct_and_occluded_use_goal_candidates():
    candidates = [object(), object()]
    term = SimpleNamespace(candidate_assets=candidates)
    assert semantic_candidate_assets(term, "direct") == candidates
    assert semantic_candidate_assets(term, "occluded") == candidates


def test_relational_flattens_target_slots():
    targets = [[object(), object()], [object()]]
    term = SimpleNamespace(relational_target_assets=targets)
    assert semantic_candidate_assets(term, "relational") == [
        targets[0][0],
        targets[0][1],
        targets[1][0],
    ]


def test_near_far_flattens_distance_slots():
    targets = [[object()], [object()], [object()]]
    term = SimpleNamespace(distance_assets=targets)
    assert semantic_candidate_assets(term, "near_far") == [slot[0] for slot in targets]


def test_object_relative_has_no_competing_target_asset():
    assert semantic_candidate_assets(SimpleNamespace(), "object_relative") == []


def test_missing_task_collection_fails_loudly():
    try:
        semantic_candidate_assets(SimpleNamespace(), "relational")
    except AttributeError as exc:
        assert "relational_target_assets" in str(exc)
    else:
        raise AssertionError("Missing task assets should not silently pass evaluation.")
