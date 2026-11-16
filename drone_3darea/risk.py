"""Operational-risk estimation of the drone environment.

Risk is deliberately decoupled from uncertainty: high uncertainty in an open
area is low risk; a dense, predictable obstacle field is high risk without any
uncertainty.  The estimator combines four bounded factors:

  proximity   : how close the nearest obstacle is (from the ray sensor),
  density     : fraction of rays that see an obstacle within the safety range,
  traffic     : dynamic-obstacle closing speed (are things converging on us?),
  energy      : ratio of estimated energy-to-go to remaining battery.

Uncertainty *modulates* the collision-related factors (uncertain proximity is
riskier than certain proximity) but does not create risk on its own:

  R_t = clip( max(prox, dens, traffic) * (1 + k_u * U_t),  energy_risk )
combined with a soft max so either collision risk or energy risk dominates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np

K_UNCERTAINTY = 0.6      # how strongly uncertainty amplifies collision risk
CLOSING_SPEED_MAX = 15.0  # m/s normalization for traffic factor


@dataclass
class RiskEstimator:
    safe_clearance: float      # m -- range below which proximity risk rises
    ray_max_range: float       # m

    def assess(self, rays_m: np.ndarray, closing_speed: float,
               energy_to_go_wh: float, battery_left_wh: float,
               U: float) -> Dict[str, float]:
        # proximity: 1 when an obstacle touches us, 0 beyond safe_clearance
        d_min = float(rays_m.min()) if rays_m.size else self.ray_max_range
        prox = float(np.clip(1.0 - d_min / self.safe_clearance, 0.0, 1.0))

        # density: fraction of rays with a return inside 2x safe clearance
        if rays_m.size:
            dens = float(np.mean(rays_m < 2.0 * self.safe_clearance))
        else:
            dens = 0.0

        # traffic: normalized closing speed of the nearest dynamic obstacle
        traffic = float(np.clip(closing_speed / CLOSING_SPEED_MAX, 0.0, 1.0))

        collision = max(prox, 0.6 * dens, traffic * max(prox, dens))
        collision = float(np.clip(collision * (1.0 + K_UNCERTAINTY * U), 0.0, 1.0))

        # energy: >1 means predicted mission energy exceeds the battery
        margin = energy_to_go_wh / max(1e-6, battery_left_wh)
        energy_risk = float(np.clip((margin - 0.5) / 0.7, 0.0, 1.0))  # 0 below 0.5, 1 at 1.2

        R = float(np.clip(max(collision, energy_risk), 0.0, 1.0))
        return {"proximity": prox, "density": dens, "traffic": traffic,
                "collision": collision, "energy_risk": energy_risk, "R": R}
