"""Wind field: structured low-frequency component + zero-mean gusts.

The structured part is a sum of sinusoidal modes (deterministic given seed,
optionally time-varying), bounded by `max_speed`.  Gusts are i.i.d. Gaussian
per step with std `gust_std` and are the *aleatoric* part the UAV cannot
predict; the structured part is what a forecast would capture.

`measure()` returns the wind as seen by the onboard anemometer: the true wind
plus measurement noise.  The uncertainty estimator only ever sees measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import math
import numpy as np


@dataclass
class WindField:
    world_size: float
    max_speed: float          # magnitude bound of the structured field
    gust_std: float = 0.0     # per-step gust noise (m/s per axis)
    num_modes: int = 3
    time_freq: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._gust_rng = np.random.default_rng(self.seed + 1)
        self.kx = rng.uniform(0.5, 2.0, size=self.num_modes) * (2 * math.pi / self.world_size)
        self.ky = rng.uniform(0.5, 2.0, size=self.num_modes) * (2 * math.pi / self.world_size)
        self.phase_x = rng.uniform(0, 2 * math.pi, size=self.num_modes)
        self.phase_y = rng.uniform(0, 2 * math.pi, size=self.num_modes)
        self.amp = rng.uniform(-1, 1, size=(self.num_modes, 2))
        norm = max(1.0, np.linalg.norm(self.amp, axis=1).sum())
        self.amp *= self.max_speed / norm
        self._last_gust = (0.0, 0.0)

    def structured(self, x: float, y: float, t: float = 0.0) -> Tuple[float, float]:
        vx = vy = 0.0
        phi_t = self.time_freq * t
        for i in range(self.num_modes):
            s = math.sin(self.kx[i] * x + self.phase_x[i] + phi_t)
            c = math.cos(self.ky[i] * y + self.phase_y[i] + phi_t)
            vx += self.amp[i, 0] * s
            vy += self.amp[i, 1] * c
        return float(vx), float(vy)

    def sample(self, x: float, y: float, t: float = 0.0) -> Tuple[float, float]:
        """True wind acting on the UAV this step (structured + fresh gust)."""
        vx, vy = self.structured(x, y, t)
        if self.gust_std > 0:
            g = self._gust_rng.normal(0.0, self.gust_std, size=2)
            self._last_gust = (float(g[0]), float(g[1]))
            vx += g[0]
            vy += g[1]
        else:
            self._last_gust = (0.0, 0.0)
        return float(vx), float(vy)

    def measure(self, true_wind: Tuple[float, float],
                sigma_meas: float, rng: np.random.Generator) -> Tuple[float, float]:
        """Anemometer reading of the wind that just acted on the vehicle."""
        if sigma_meas <= 0:
            return true_wind
        n = rng.normal(0.0, sigma_meas, size=2)
        return float(true_wind[0] + n[0]), float(true_wind[1] + n[1])
