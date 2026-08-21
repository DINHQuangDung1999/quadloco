"""Small 2-D A* planner for expert waypoint generation.

The planner is deliberately independent of Isaac Lab.  Obstacles are oriented
rectangles in world coordinates and are inflated before rasterization, making
the returned path suitable for a finite-radius robot.  Raw grid paths are
greedily line-of-sight pruned before being returned as metric waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RectangleObstacle:
    """An oriented rectangle described by center, full size, and yaw."""

    center: tuple[float, float]
    size: tuple[float, float]
    yaw: float = 0.0


def _point_is_free(
    point: np.ndarray,
    obstacles: Iterable[RectangleObstacle],
    inflation_radius: float,
) -> bool:
    for obstacle in obstacles:
        delta = point - np.asarray(obstacle.center, dtype=np.float64)
        cosine = math.cos(obstacle.yaw)
        sine = math.sin(obstacle.yaw)
        local_x = cosine * delta[0] + sine * delta[1]
        local_y = -sine * delta[0] + cosine * delta[1]
        half_x = 0.5 * obstacle.size[0] + inflation_radius
        half_y = 0.5 * obstacle.size[1] + inflation_radius
        if abs(local_x) <= half_x and abs(local_y) <= half_y:
            return False
    return True


def _segment_is_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Iterable[RectangleObstacle],
    inflation_radius: float,
    sample_spacing: float,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    sample_count = max(2, int(math.ceil(distance / sample_spacing)) + 1)
    return all(
        _point_is_free(start + alpha * (end - start), obstacles, inflation_radius)
        for alpha in np.linspace(0.0, 1.0, sample_count)
    )


def _prune_path(
    path: list[np.ndarray],
    obstacles: tuple[RectangleObstacle, ...],
    inflation_radius: float,
    sample_spacing: float,
) -> list[np.ndarray]:
    """Remove grid nodes while retaining collision-free line segments."""
    if len(path) <= 2:
        return path
    pruned = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not _segment_is_free(
            path[anchor], path[candidate], obstacles, inflation_radius, sample_spacing
        ):
            candidate -= 1
        pruned.append(path[candidate])
        anchor = candidate
    return pruned


def plan_astar(
    start: tuple[float, float] | np.ndarray,
    goal: tuple[float, float] | np.ndarray,
    obstacles: Iterable[RectangleObstacle],
    *,
    resolution: float = 0.10,
    inflation_radius: float = 0.0,
    planning_margin: float = 1.0,
) -> np.ndarray:
    """Plan and simplify an 8-connected metric path from ``start`` to ``goal``.

    Returns an ``[N, 2]`` array including the exact start and goal.  Raises a
    ``RuntimeError`` when no route exists inside the generated planning bounds.
    """
    if resolution <= 0.0 or inflation_radius < 0.0 or planning_margin <= 0.0:
        raise ValueError("Invalid A* resolution, inflation radius, or planning margin")

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    obstacles = tuple(obstacles)
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("A* start and goal must both have shape (2,)")
    if not _point_is_free(start, obstacles, inflation_radius):
        raise RuntimeError("A* start lies inside an inflated obstacle")
    if not _point_is_free(goal, obstacles, inflation_radius):
        raise RuntimeError("A* goal lies inside an inflated obstacle")

    bound_points = [start, goal]
    for obstacle in obstacles:
        radius = 0.5 * math.hypot(*obstacle.size) + inflation_radius
        center = np.asarray(obstacle.center, dtype=np.float64)
        bound_points.extend((center - radius, center + radius))
    stacked = np.stack(bound_points)
    lower = stacked.min(axis=0) - planning_margin
    upper = stacked.max(axis=0) + planning_margin
    grid_shape = np.ceil((upper - lower) / resolution).astype(int) + 1

    def to_index(point: np.ndarray) -> tuple[int, int]:
        index = np.rint((point - lower) / resolution).astype(int)
        index = np.clip(index, 0, grid_shape - 1)
        return int(index[0]), int(index[1])

    def to_world(index: tuple[int, int]) -> np.ndarray:
        return lower + resolution * np.asarray(index, dtype=np.float64)

    start_index = to_index(start)
    goal_index = to_index(goal)
    moves = (
        (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),                         (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),  (1, 0, 1.0),  (1, 1, math.sqrt(2.0)),
    )
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_index)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_index: None}
    cost_so_far = {start_index: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_index:
            break
        for dx, dy, move_cost in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]):
                continue
            neighbor_world = to_world(neighbor)
            if not _point_is_free(neighbor_world, obstacles, inflation_radius):
                continue
            # Prevent diagonal moves through an occupied grid corner.
            if dx and dy:
                side_x = to_world((current[0] + dx, current[1]))
                side_y = to_world((current[0], current[1] + dy))
                if not _point_is_free(side_x, obstacles, inflation_radius) or not _point_is_free(
                    side_y, obstacles, inflation_radius
                ):
                    continue
            new_cost = cost_so_far[current] + move_cost * resolution
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = float(np.linalg.norm(to_world(neighbor) - goal))
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current

    if goal_index not in came_from:
        raise RuntimeError("A* could not find a collision-free route")

    indices = []
    current: tuple[int, int] | None = goal_index
    while current is not None:
        indices.append(current)
        current = came_from[current]
    indices.reverse()
    path = [to_world(index) for index in indices]
    path[0] = start
    path[-1] = goal
    path = _prune_path(path, obstacles, inflation_radius, 0.5 * resolution)
    return np.asarray(path, dtype=np.float32)
