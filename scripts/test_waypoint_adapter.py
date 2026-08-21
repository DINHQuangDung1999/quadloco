#!/usr/bin/env python

"""Tests for deterministic deployment waypoint conversion."""

import sys
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).parent / "quadloco_rsl_rl"
sys.path.insert(0, str(MODULE_ROOT))

from waypoint_adapter import waypoint_to_velocity  # noqa: E402


PARAMS = {
    "forward_velocity": 1.0,
    "yaw_gain": 1.5,
    "max_yaw_rate": 0.8,
    "goal_tolerance": 0.25,
    "slowdown_distance": 1.0,
}


def test_forward_waypoint_produces_forward_velocity():
    command = waypoint_to_velocity(np.array([2.0, 0.0]), **PARAMS)
    np.testing.assert_allclose(command, [1.0, 0.0, 0.0])


def test_lateral_waypoint_turns_before_advancing():
    command = waypoint_to_velocity(np.array([0.0, 2.0]), **PARAMS)
    np.testing.assert_allclose(command, [0.0, 0.0, 0.8], atol=1e-6)


def test_near_waypoint_stops_and_intermediate_distance_slows_down():
    stopped = waypoint_to_velocity(np.array([0.1, 0.0]), **PARAMS)
    slowed = waypoint_to_velocity(np.array([0.625, 0.0]), **PARAMS)
    np.testing.assert_allclose(stopped, np.zeros(3))
    np.testing.assert_allclose(slowed, [0.5, 0.0, 0.0], atol=1e-6)


def test_minimum_approach_velocity_overcomes_policy_dead_zone():
    command = waypoint_to_velocity(
        np.array([0.26, 0.0]), minimum_approach_velocity=0.15, **PARAMS
    )
    np.testing.assert_allclose(command, [0.15, 0.0, 0.0], atol=1e-6)

    stopped = waypoint_to_velocity(
        np.array([0.24, 0.0]), minimum_approach_velocity=0.15, **PARAMS
    )
    np.testing.assert_allclose(stopped, np.zeros(3))
