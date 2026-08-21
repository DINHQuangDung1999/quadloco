"""Task-specific asset selection helpers for navigation evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _flatten_assets(assets: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in assets:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_assets(item))
        else:
            flattened.append(item)
    return flattened


def semantic_candidate_assets(command_term: Any, mode: str) -> list[Any]:
    """Return the scene assets that can represent navigation targets."""
    attribute_by_mode = {
        "direct": "candidate_assets",
        "occluded": "candidate_assets",
        "relational": "relational_target_assets",
        "near_far": "distance_assets",
    }
    if mode == "object_relative":
        # Success is defined at a generated position relative to one object;
        # that reference object is not a competing navigation target.
        return []
    try:
        attribute = attribute_by_mode[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported navigation mode: {mode}") from exc
    if not hasattr(command_term, attribute):
        raise AttributeError(
            f"{mode} evaluation requires command_term.{attribute}."
        )
    return _flatten_assets(getattr(command_term, attribute))
