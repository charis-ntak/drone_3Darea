"""World generators of the 3D environment: topology families, routes and moving obstacles.

A world is (cyl, terrain, dynamic): static cylinders (x, y, r, z_lo, z_hi), an optional heightfield and
a list of moving spheres. Families differ in the number and kind of static obstacles, in the amount of
traffic and in the presence of terrain, so that a planner trained or tested over all of them meets different
topologies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .obstacles import make_scenario
from .geom3d import Terrain

# kinds = probabilities of (tower, full-height, elevated volume)
FAMILIES = {
    "open":    dict(n_static=3, n_dyn=2, kinds=(0.50, 0.25, 0.25), radius=(40.0, 80.0), hills=0),
    "mixed":   dict(n_static=6, n_dyn=3, kinds=(0.50, 0.25, 0.25), radius=(40.0, 80.0), hills=0),
    "towers":  dict(n_static=9, n_dyn=3, kinds=(0.80, 0.20, 0.00), radius=(30.0, 55.0), hills=0),
    "layered": dict(n_static=6, n_dyn=3, kinds=(0.40, 0.10, 0.50), radius=(50.0, 100.0), hills=0),
    "traffic": dict(n_static=4, n_dyn=5, kinds=(0.50, 0.25, 0.25), radius=(40.0, 80.0), hills=0),
    "hills":   dict(n_static=3, n_dyn=3, kinds=(0.50, 0.25, 0.25), radius=(40.0, 70.0), hills=4),
}
TRAIN_FAMILIES = ("open", "mixed", "towers", "layered", "traffic", "hills")
BIG = 1.0e9   # top of an elevated volume or of a full-height zone


def draw_route_v2(W: float, rng: np.random.Generator) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Start and goal in the plane (40 percent corner diagonals, 30 percent
    flights from a corner, 30 percent flights from anywhere; missions of at least 0.5 W)."""
    m = 0.1 * W
    r = rng.random()
    if r < 0.4:
        cx, cy = rng.choice([m, W - m]), rng.choice([m, W - m])
        js = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.05 * W)
        jg = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.05 * W)
        sx = cx + js if cx == m else cx - js
        sy = cy + js if cy == m else cy - js
        gx = (W - m) - jg if cx == m else m + jg
        gy = (W - m) - jg if cy == m else m + jg
        return (float(sx), float(sy)), (float(gx), float(gy))
    corner = r < 0.7
    for _ in range(1000):
        if corner:
            cx, cy = rng.choice([m, W - m]), rng.choice([m, W - m])
            jit = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.05 * W)
            sx = cx + jit if cx == m else cx - jit
            sy = cy + jit if cy == m else cy - jit
        else:
            sx, sy = rng.uniform(m, W - m, size=2)
        ang = rng.uniform(-math.pi, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        dx = (W - m - sx) / ca if ca > 1e-9 else (m - sx) / ca if ca < -1e-9 else math.inf
        dy = (W - m - sy) / sa if sa > 1e-9 else (m - sy) / sa if sa < -1e-9 else math.inf
        dmax = min(dx, dy)
        if dmax < 0.5 * W:
            continue
        d = dmax if rng.random() < 0.5 else rng.uniform(0.5 * W, dmax)
        return (float(sx), float(sy)), (float(sx + d * ca), float(sy + d * sa))
    raise RuntimeError("route draw failed")


def make_static(cfg, rng: np.random.Generator) -> np.ndarray:
    fam = FAMILIES[cfg.topology]
    n = fam["n_static"] if cfg.num_static is None else cfg.num_static
    circles = make_scenario(n, cfg.world_size, cfg.start, cfg.goal, cfg.obstacle_radius_range,
                            cfg.obstacle_min_separation, cfg.clearance_from_endpoints, rng,
                            block_diagonal_prob=cfg.block_diagonal_prob)
    hb = cfg.band
    rows = []
    for x, y, r in circles:
        kind = rng.choice(3, p=fam["kinds"])
        if kind == 0:      # tower or terrain feature: ground to a top inside the band
            rows.append((x, y, r, 0.0, cfg.z_min + rng.uniform(0.20, 0.85) * hb))
        elif kind == 1:    # full height
            rows.append((x, y, r, 0.0, BIG))
        else:              # elevated volume: a floor inside the band, open below
            rows.append((x, y, r, cfg.z_min + rng.uniform(0.35, 0.75) * hb, BIG))
    return np.array(rows, dtype=float).reshape(-1, 5)


def make_hills(cfg, rng: np.random.Generator, n: int = 41) -> Optional[Terrain]:
    k = FAMILIES[cfg.topology]["hills"]
    if not k:
        return None
    W, hb = cfg.world_size, cfg.band
    g = np.linspace(0.0, W, n)
    X, Y = np.meshgrid(g, g)
    h = np.zeros((n, n))
    for _ in range(k):
        cx, cy = rng.uniform(0.15 * W, 0.85 * W, size=2)
        s = rng.uniform(0.06, 0.12) * W
        peak = cfg.z_min + rng.uniform(0.2, 0.9) * hb
        h = np.maximum(h, peak * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)))
    for px, py in (cfg.start, cfg.goal):      # keep the endpoints over flat ground
        h *= 1.0 - np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * (0.08 * W) ** 2))
    return Terrain(h, W)


@dataclass
class Dyn3D:
    """Moving obstacle: constant speed, heading random walk, vertical speed random walk inside the band."""
    x: float
    y: float
    z: float
    r: float
    speed: float
    heading: float
    turn_std: float = 0.0
    vz: float = 0.0
    vz_std: float = 0.0
    vz_max: float = 2.0

    @property
    def vx(self) -> float:
        return self.speed * math.cos(self.heading)

    @property
    def vy(self) -> float:
        return self.speed * math.sin(self.heading)

    def step(self, dt: float, W: float, z_lo: float, z_hi: float, rng: np.random.Generator) -> None:
        if self.turn_std > 0:
            self.heading += float(rng.normal(0.0, self.turn_std))
        if self.vz_std > 0:
            self.vz = float(np.clip(self.vz + rng.normal(0.0, self.vz_std), -self.vz_max, self.vz_max))
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        if self.x < self.r:
            self.x, self.heading = self.r, math.pi - self.heading
        elif self.x > W - self.r:
            self.x, self.heading = W - self.r, math.pi - self.heading
        if self.y < self.r:
            self.y, self.heading = self.r, -self.heading
        elif self.y > W - self.r:
            self.y, self.heading = W - self.r, -self.heading
        if self.z < z_lo:
            self.z, self.vz = z_lo, abs(self.vz)
        elif self.z > z_hi:
            self.z, self.vz = z_hi, -abs(self.vz)
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi


def make_dynamic(cfg, cyl: np.ndarray, rng: np.random.Generator) -> List[Dyn3D]:
    out: List[Dyn3D] = []
    W = cfg.world_size
    attempts = 0
    while len(out) < cfg.num_dynamic_obstacles and attempts < 5000:
        attempts += 1
        r = rng.uniform(*cfg.dynamic_radius_range)
        x, y = rng.uniform(r + 1, W - r - 1, size=2)
        z = rng.uniform(cfg.z_min, cfg.z_max)
        if math.hypot(x - cfg.start[0], y - cfg.start[1]) < r + cfg.clearance_from_endpoints:
            continue
        if math.hypot(x - cfg.goal[0], y - cfg.goal[1]) < r + cfg.clearance_from_endpoints:
            continue
        if cyl.size and bool((np.hypot(x - cyl[:, 0], y - cyl[:, 1]) < r + cyl[:, 2] + 10.0).any()):
            continue
        out.append(Dyn3D(x=float(x), y=float(y), z=float(z), r=float(r),
                         speed=float(rng.uniform(*cfg.dynamic_speed_range)),
                         heading=float(rng.uniform(0, 2 * math.pi)),
                         turn_std=math.radians(cfg.dynamic_turn_std_deg),
                         vz_std=cfg.dynamic_vz_std, vz_max=cfg.dynamic_vz_max))
    if len(out) < cfg.num_dynamic_obstacles:
        raise RuntimeError("could not place the moving obstacles")
    return out
