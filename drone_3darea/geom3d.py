"""3D geometry of the drone environment.

Static obstacles are vertical cylinders with a floor and a top, rows (x, y, r, z_lo, z_hi):
  tower / terrain feature : z_lo = 0, z_hi = top       (can be overflown)
  elevated volume         : z_lo = floor, z_hi = +inf  (can be underflown)
  full-height zone        : spans the whole band       (must be flown around)
Moving obstacles are spheres, rows (x, y, z, r).  Terrain, when present, is a heightfield
sampled by bilinear interpolation.  All ray tests are vectorized over rays and obstacles.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np


def ray_dirs(heading: float, n_h: int = 8, cone_h: float = math.pi, n_e: int = 4,
             cone_e: float = math.radians(120.0), elev: float = math.radians(25.0)) -> np.ndarray:
    """Unit direction vectors of the range sensor: a horizontal fan of n_h rays, a fan of n_e rays
    pitched up by `elev` and a fan of n_e rays pitched down by `elev`. Shape (n_h + 2 n_e, 3)."""
    az_h = heading + np.linspace(-cone_h / 2, cone_h / 2, n_h)
    az_e = heading + np.linspace(-cone_e / 2, cone_e / 2, n_e)
    ce, se = math.cos(elev), math.sin(elev)
    d = np.zeros((n_h + 2 * n_e, 3))
    d[:n_h, 0], d[:n_h, 1] = np.cos(az_h), np.sin(az_h)
    d[n_h:n_h + n_e, 0], d[n_h:n_h + n_e, 1], d[n_h:n_h + n_e, 2] = ce * np.cos(az_e), ce * np.sin(az_e), se
    d[n_h + n_e:, 0], d[n_h + n_e:, 1], d[n_h + n_e:, 2] = ce * np.cos(az_e), ce * np.sin(az_e), -se
    return d


class Terrain:
    """Heightfield on a regular grid over [0, W] x [0, W]; h[iy, ix] in metres."""

    def __init__(self, h: np.ndarray, world_size: float):
        self.h = np.asarray(h, dtype=float)
        self.n = self.h.shape[0]
        self.cell = world_size / (self.n - 1)
        self.max = float(self.h.max())

    def height(self, x, y):
        gx = np.clip(np.asarray(x, dtype=float) / self.cell, 0, self.n - 1 - 1e-9)
        gy = np.clip(np.asarray(y, dtype=float) / self.cell, 0, self.n - 1 - 1e-9)
        x0, y0 = gx.astype(int), gy.astype(int)
        fx, fy = gx - x0, gy - y0
        h = self.h
        return (h[y0, x0] * (1 - fx) * (1 - fy) + h[y0, x0 + 1] * fx * (1 - fy)
                + h[y0 + 1, x0] * (1 - fx) * fy + h[y0 + 1, x0 + 1] * fx * fy)


def raycast3d(origin, dirs: np.ndarray, cyl: np.ndarray, sph: np.ndarray, world_size: float,
              max_range: float, terrain: Optional[Terrain] = None, n_march: int = 24) -> np.ndarray:
    """Distance along each ray to the nearest obstacle, wall or terrain, capped at max_range. Shape (n,)."""
    ox, oy, oz = origin
    n = dirs.shape[0]
    best = np.full(n, max_range)
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # side walls of the square workspace
    with np.errstate(divide="ignore", invalid="ignore"):
        for num, den in ((-ox, dx), (world_size - ox, dx), (-oy, dy), (world_size - oy, dy)):
            t = np.where(np.abs(den) > 1e-9, num / den, np.inf)
            t = np.where(t > 0, t, np.inf)
            best = np.minimum(best, t)

    if cyl.size:
        cx, cy, r, zlo, zhi = (cyl[:, k][None, :] for k in range(5))
        fx, fy = ox - cx, oy - cy                                    # (1, m)
        a = (dx * dx + dy * dy)[:, None]                             # (n, 1)
        b = 2.0 * (fx * dx[:, None] + fy * dy[:, None])              # (n, m)
        c = fx * fx + fy * fy - r * r                                # (1, m)
        disc = b * b - 4.0 * a * c
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = (-b - sq) / (2.0 * a)
        z1 = oz + t1 * dz[:, None]
        side = np.where(ok & (t1 >= 0) & (z1 >= zlo) & (z1 <= zhi), t1, np.inf)
        # caps: entering through the floor of the volume (from below) or through its top (from above)
        dzc = dz[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.where((dzc > 1e-9) & (oz < zlo), (zlo - oz) / dzc, np.inf)
            tt = np.where((dzc < -1e-9) & (oz > zhi), (zhi - oz) / dzc, np.inf)
        cap = np.full_like(side, np.inf)
        for tc in (tb, tt):
            fin = np.isfinite(tc)
            tcs = np.where(fin, tc, 0.0)
            px, py = fx + tcs * dx[:, None], fy + tcs * dy[:, None]
            cap = np.minimum(cap, np.where(fin & (px * px + py * py <= r * r), tc, np.inf))
        inside = (c <= 0) & (oz >= zlo) & (oz <= zhi)                # origin inside a volume
        t = np.minimum(side, cap)
        t = np.where(np.broadcast_to(inside, t.shape), 0.0, t)
        best = np.minimum(best, t.min(axis=1))

    if sph.size:
        sx, sy, sz, sr = (sph[:, k][None, :] for k in range(4))
        fx, fy, fz = ox - sx, oy - sy, oz - sz
        b = 2.0 * (fx * dx[:, None] + fy * dy[:, None] + fz * dz[:, None])
        c = fx * fx + fy * fy + fz * fz - sr * sr
        disc = b * b - 4.0 * c
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1, t2 = (-b - sq) / 2.0, (-b + sq) / 2.0
        t = np.where(ok & (t1 >= 0), t1, np.where(ok & (t2 >= 0), 0.0, np.inf))   # inside a sphere -> 0
        best = np.minimum(best, t.min(axis=1))

    if terrain is not None and terrain.max > 0:
        ts = np.linspace(max_range / n_march, max_range, n_march)    # (k,)
        px = ox + dx[:, None] * ts[None, :]
        py = oy + dy[:, None] * ts[None, :]
        pz = oz + dz[:, None] * ts[None, :]
        h = terrain.height(px, py)
        hit = (pz <= h) & (h > 1e-6)          # flat ground at the datum is not a ranged obstacle
        first = np.where(hit.any(axis=1), ts[np.argmax(hit, axis=1)], np.inf)
        best = np.minimum(best, first)
    return best


def clearance3d(p, cyl: np.ndarray, sph: np.ndarray, terrain: Optional[Terrain] = None) -> float:
    """True distance from point p to the nearest obstacle surface (0 if inside)."""
    x, y, z = p
    best = math.inf
    if cyl.size:
        dh = np.maximum(0.0, np.hypot(x - cyl[:, 0], y - cyl[:, 1]) - cyl[:, 2])
        dv = np.maximum(0.0, np.maximum(cyl[:, 3] - z, z - cyl[:, 4]))
        best = min(best, float(np.hypot(dh, dv).min()))
    if sph.size:
        d = np.sqrt((x - sph[:, 0]) ** 2 + (y - sph[:, 1]) ** 2 + (z - sph[:, 2]) ** 2) - sph[:, 3]
        best = min(best, float(max(0.0, d.min())))
    if terrain is not None and terrain.max > 0:
        h = float(terrain.height(x, y))
        if h > 1e-6:                           # clearance above terrain features only; flat ground is covered by z_min
            best = min(best, max(0.0, z - h))
    return best


def segment_hits(p0, p1, cyl: np.ndarray, sph: np.ndarray, terrain: Optional[Terrain] = None,
                 n_sub: int = 6) -> bool:
    """Collision test of the flown segment p0 -> p1: sampled against cylinders and terrain (the step is
    several times shorter than any obstacle radius), analytic against spheres."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    f = np.linspace(1.0 / n_sub, 1.0, n_sub)[:, None]
    pts = p0[None, :] + f * (p1 - p0)[None, :]
    if cyl.size:
        dh = np.hypot(pts[:, 0:1] - cyl[None, :, 0], pts[:, 1:2] - cyl[None, :, 1]) <= cyl[None, :, 2]
        dv = (pts[:, 2:3] >= cyl[None, :, 3]) & (pts[:, 2:3] <= cyl[None, :, 4])
        if bool((dh & dv).any()):
            return True
    if terrain is not None and terrain.max > 0:
        if bool((pts[:, 2] <= terrain.height(pts[:, 0], pts[:, 1])).any()):
            return True
    if sph.size:
        d = p1 - p0
        a = float(d @ d)
        for sx, sy, sz, sr in sph:
            c = np.array([sx, sy, sz])
            t = 0.0 if a == 0 else max(0.0, min(1.0, float((c - p0) @ d) / a))
            if np.linalg.norm(p0 + t * d - c) <= sr:
                return True
    return False
