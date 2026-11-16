"""Placement of static obstacle footprints (circles) in the square workspace."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

Obstacle = Tuple[float, float, float]  # (x, y, radius)


def _point_to_segment(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return np.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
    return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))


def make_scenario(num_obstacles: int, world_size: float, start: Tuple[float, float],
                  goal: Tuple[float, float], radius_range: Tuple[float, float],
                  min_sep: float, clearance: float, rng: np.random.Generator,
                  block_diagonal_prob: float = 0.7) -> List[Obstacle]:
    obstacles: List[Obstacle] = []

    # Sometimes force the first obstacle onto the start-goal corridor so the straight-line
    # policy cannot win trivially.
    if num_obstacles > 0 and rng.random() < block_diagonal_prob:
        r = rng.uniform(*radius_range)
        t = rng.uniform(0.3, 0.7)
        cx = start[0] + t * (goal[0] - start[0]) + rng.uniform(-3.0, 3.0)
        cy = start[1] + t * (goal[1] - start[1]) + rng.uniform(-3.0, 3.0)
        cx = float(np.clip(cx, r + 1, world_size - r - 1))
        cy = float(np.clip(cy, r + 1, world_size - r - 1))
        obstacles.append((cx, cy, r))

    attempts, max_attempts = 0, 5000
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        r = rng.uniform(*radius_range)
        x = rng.uniform(r + 1, world_size - r - 1)
        y = rng.uniform(r + 1, world_size - r - 1)
        if np.hypot(x - start[0], y - start[1]) < r + clearance:
            continue
        if np.hypot(x - goal[0], y - goal[1]) < r + clearance:
            continue
        ok = True
        for ox, oy, or_ in obstacles:
            if np.hypot(x - ox, y - oy) < r + or_ + min_sep:
                ok = False
                break
        if ok:
            obstacles.append((x, y, r))
    if len(obstacles) < num_obstacles:
        raise RuntimeError(f"Could not place {num_obstacles} obstacles after {max_attempts} attempts")
    return obstacles
