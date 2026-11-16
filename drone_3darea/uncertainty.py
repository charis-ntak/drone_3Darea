"""Online environmental-uncertainty estimation of the drone environment.

The estimator never reads the simulator's hidden noise parameters directly.
Each component uses a signal that a real avionics stack would have:

  U_loc      : reported localization std (a GPS receiver reports its own
               accuracy estimate, e.g. via HDOP) normalized by a design bound.
  U_sense    : reported obstacle-ranging noise spec, normalized likewise.
  U_wind     : sample std of the anemometer wind measurements over a sliding
               window -- captures both gusts and measurement noise, i.e. the
               unpredictable part of the wind actually experienced.
  U_obstacle : mean constant-velocity prediction innovation of tracked dynamic
               obstacles over a sliding window, normalized -- if traffic flies
               straight this is ~sensor noise; if it manoeuvres it grows.

The scalar index is a weighted mean:
  U_t = (w_l*U_loc + w_s*U_sense + w_w*U_wind + w_o*U_obstacle) / sum(w)
All components are clipped to [0, 1]; the normalization bounds are design
constants declared here (and reported in the paper), not tuned per scenario.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np

# Normalization bounds: value at which a component saturates to 1.
LOC_STD_MAX = 20.0        # m
SENSE_STD_MAX = 25.0      # m
WIND_STD_MAX = 4.0        # m/s (per-axis std of measured wind)
INNOV_MAX = 20.0          # m  (mean 1-step prediction error of traffic)

WINDOW = 20               # sliding-window length (steps)

COMPONENT_WEIGHTS = {"loc": 1.0, "sense": 1.0, "wind": 1.0, "obstacle": 1.0}


@dataclass
class UncertaintyEstimator:
    window: int = WINDOW
    loc_std_max: float = LOC_STD_MAX
    sense_std_max: float = SENSE_STD_MAX
    wind_std_max: float = WIND_STD_MAX
    innov_max: float = INNOV_MAX
    _wind_meas: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=WINDOW))
    _innovations: Deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    _last_obs_tracks: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self._wind_meas.clear()
        self._innovations.clear()
        self._last_obs_tracks.clear()

    # ---- per-step feeds -------------------------------------------------
    def feed_wind_measurement(self, wx: float, wy: float) -> None:
        self._wind_meas.append((wx, wy))

    def feed_obstacle_observations(self, obs: List[Tuple[int, float, float]],
                                   dt: float) -> None:
        """obs: list of (track_id, observed_x, observed_y) for visible traffic.

        Compares each observation against the constant-velocity prediction from
        the previous two observations and records the innovation magnitude.
        """
        new_tracks: Dict[int, Tuple[float, float, float, float]] = {}
        for tid, ox, oy in obs:
            if tid in self._last_obs_tracks:
                px, py, pvx, pvy = self._last_obs_tracks[tid]
                pred_x, pred_y = px + pvx * dt, py + pvy * dt
                self._innovations.append(float(np.hypot(ox - pred_x, oy - pred_y)))
                vx, vy = (ox - px) / dt, (oy - py) / dt
            else:
                vx = vy = 0.0
            new_tracks[tid] = (ox, oy, vx, vy)
        self._last_obs_tracks = new_tracks

    # ---- components -----------------------------------------------------
    def u_wind(self) -> float:
        if len(self._wind_meas) < 3:
            return 0.0
        arr = np.asarray(self._wind_meas)
        std = float(np.std(arr, axis=0).mean())
        return min(1.0, std / self.wind_std_max)

    def u_obstacle(self) -> float:
        if len(self._innovations) < 3:
            return 0.0
        return min(1.0, float(np.mean(self._innovations)) / self.innov_max)

    def u_loc(self, reported_std: float) -> float:
        return min(1.0, max(0.0, reported_std) / self.loc_std_max)

    def u_sense(self, reported_std: float) -> float:
        return min(1.0, max(0.0, reported_std) / self.sense_std_max)

    # ---- combined index -------------------------------------------------
    def index(self, reported_loc_std: float, reported_sense_std: float) -> Dict[str, float]:
        comps = {
            "loc": self.u_loc(reported_loc_std),
            "sense": self.u_sense(reported_sense_std),
            "wind": self.u_wind(),
            "obstacle": self.u_obstacle(),
        }
        wsum = sum(COMPONENT_WEIGHTS.values())
        comps["U"] = sum(COMPONENT_WEIGHTS[k] * v for k, v in comps.items()) / wsum
        return comps
