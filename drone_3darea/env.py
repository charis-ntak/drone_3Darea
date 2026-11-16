"""3D drone path-planning environment with explicit uncertainty injection.

  state    : position (x, y, z), heading, airspeed, vertical command, battery;
  actions  : 27 = heading change (3) x speed change (3) x vertical command (descend, level, climb);
  obstacles: static vertical cylinders with a floor and a top, optional terrain, moving spheres;
  sensing  : 16 range rays (a horizontal fan of 8, an upper fan of 4 and a lower fan of 4);
  energy   : P = P_hover + c_p v^3 + k_climb max(vz, 0) - k_descend max(-vz, 0).

Ground truth is kept internal; the planner sees observations only: own position with localization noise
(dead reckoning during a GPS outage; barometric altitude), noisy ranges, noisy traffic tracks and a
noisy anemometer. An uncertainty index U and a risk index R are estimated online from these signals.

step() returns (obs, reward_vec, shared_reward, done, info). reward_vec holds the four objectives
(distance, energy, time, safety), each <= 0 per step; the scalar reward for a preference vector omega is
shared_reward + omega . reward_vec, computed by the caller, so one transition serves any preference.
Collision is charged to the safety objective and battery exhaustion to the energy objective.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .uncertainty import UncertaintyEstimator
from .risk import RiskEstimator
from .wind import WindField
from .geom3d import Terrain, clearance3d, ray_dirs, raycast3d, segment_hits
from .topologies import Dyn3D, draw_route_v2, make_dynamic, make_hills, make_static

NUM_OBJECTIVES = 4
DYN_SLOTS = 2
EMPTY5, EMPTY4 = np.zeros((0, 5)), np.zeros((0, 4))


@dataclass
class EpisodeMetrics3D:
    path_length: float = 0.0
    energy_wh: float = 0.0
    time_s: float = 0.0
    climb_m: float = 0.0
    min_clearance: float = math.inf
    steps: int = 0
    success: bool = False
    collision: bool = False
    battery_exhausted: bool = False
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    weight_history: List[np.ndarray] = field(default_factory=list)
    u_history: List[float] = field(default_factory=list)
    r_history: List[float] = field(default_factory=list)
    battery_history: List[float] = field(default_factory=list)


class Uncertainty3D(UncertaintyEstimator):
    """Same estimator; the traffic innovation is the 3D miss of the constant-velocity prediction."""

    def feed_tracks3d(self, obs: List[Tuple[int, float, float, float]], dt: float) -> None:
        new: Dict[int, Tuple[float, ...]] = {}
        for tid, ox, oy, oz in obs:
            if tid in self._last_obs_tracks:
                px, py, pz, vx, vy, vz = self._last_obs_tracks[tid]
                self._innovations.append(float(math.sqrt((ox - px - vx * dt) ** 2 + (oy - py - vy * dt) ** 2
                                                         + (oz - pz - vz * dt) ** 2)))
                v = ((ox - px) / dt, (oy - py) / dt, (oz - pz) / dt)
            else:
                v = (0.0, 0.0, 0.0)
            new[tid] = (ox, oy, oz) + v
        self._last_obs_tracks = new


class DroneEnv3D:
    def __init__(self, cfg, seed: Optional[int] = None, fixed_world: Optional[dict] = None,
                 wind_field=None, estimator: Optional[Uncertainty3D] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._fixed_world = fixed_world        # real areas: {"cyl": array, "terrain": Terrain or None}
        self._ext_wind = wind_field
        dth = math.radians(cfg.dtheta_deg)
        self._action_table = np.array([(a, b, c) for a in (-dth, 0.0, dth) for b in (-cfg.dv, 0.0, cfg.dv)
                                       for c in (-1.0, 0.0, 1.0)], dtype=np.float32)
        self.cyl, self.terrain = EMPTY5, None
        self.dynamic_obstacles: List[Dyn3D] = []
        self.wind = None
        self.uncertainty = estimator if estimator is not None else Uncertainty3D()
        self.risk = RiskEstimator(cfg.safe_clearance, cfg.ray_max_range)
        self._vz_step = cfg.kappa_v * cfg.band / cfg.dt                       # m/s of a climb command
        self._max_step_energy = self._power(cfg.v_max, self._vz_step / cfg.z_scale) * cfg.dt / 3600.0
        self.metrics = EpisodeMetrics3D()

    # ------------------------------------------------------------------ api
    @property
    def num_rays(self) -> int:
        return self.cfg.num_rays + 2 * self.cfg.num_rays_e

    @property
    def state_dim(self) -> int:
        # 9 base + rays + 6 per traffic slot + 2 wind + U + R
        return 9 + self.num_rays + 6 * DYN_SLOTS + 2 + 2

    @property
    def num_actions(self) -> int:
        return self._action_table.shape[0]

    def _power(self, v_air: float, vz: float = 0.0) -> float:
        c = self.cfg
        p = c.p_hover_w + c.c_p * v_air ** 3 + c.k_climb_w * max(vz, 0.0) - c.k_descend_w * max(-vz, 0.0)
        return max(0.5 * c.p_hover_w, p)

    def _sph(self) -> np.ndarray:
        if not self.dynamic_obstacles:
            return EMPTY4
        return np.array([(o.x, o.y, o.z, o.r) for o in self.dynamic_obstacles], dtype=float)

    # ---------------------------------------------------------------- reset
    def reset(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.random_geometry:
            cfg.start, cfg.goal = draw_route_v2(cfg.world_size, self.rng)
            cfg.start_z = float(self.rng.uniform(cfg.z_min + 0.1 * cfg.band, cfg.z_max - 0.1 * cfg.band))
            cfg.goal_z = float(self.rng.uniform(cfg.z_min + 0.1 * cfg.band, cfg.z_max - 0.1 * cfg.band))
        if self._fixed_world is not None:
            self.cyl = np.asarray(self._fixed_world.get("cyl", EMPTY5), dtype=float).reshape(-1, 5)
            self.terrain = self._fixed_world.get("terrain")
        else:
            self.cyl = make_static(cfg, self.rng)
            self.terrain = make_hills(cfg, self.rng)
        self.dynamic_obstacles = make_dynamic(cfg, self.cyl, self.rng) if cfg.num_dynamic_obstacles > 0 else []
        if self._ext_wind is not None:
            self.wind = self._ext_wind
        elif cfg.wind_mean_max > 0 or cfg.gust_std > 0:
            self.wind = WindField(world_size=cfg.world_size, max_speed=cfg.wind_mean_max, gust_std=cfg.gust_std,
                                  num_modes=cfg.wind_modes, time_freq=cfg.wind_time_freq,
                                  seed=int(self.rng.integers(0, 2 ** 31)))
        else:
            self.wind = None

        self.x, self.y = cfg.start
        self.z = cfg.start_z if cfg.start_z is not None else cfg.z_min
        self.gz = cfg.goal_z if cfg.goal_z is not None else cfg.z_min
        gx, gy = cfg.goal
        spread = math.radians(cfg.init_heading_spread_deg)
        self.theta = _wrap(math.atan2(gy - self.y, gx - self.x) + self.rng.uniform(-spread, spread))
        self.v = 0.5 * (cfg.v_min + cfg.v_max)
        self.vz = 0.0
        self.battery_wh = cfg.battery_capacity_wh * cfg.battery_start_frac
        self.step_count = 0
        self._straight_line = math.sqrt((gx - self.x) ** 2 + (gy - self.y) ** 2 + (self.gz - self.z) ** 2)
        self.uncertainty.reset()
        self._last_uncertainty: Dict[str, float] = {"U": 0.0}
        self._last_risk: Dict[str, float] = {"R": 0.0}
        self.metrics = EpisodeMetrics3D(trajectory=[(self.x, self.y, self.z)])
        self.gps_outage, self._dr_pos, self._last_fix = False, None, None
        return self._observe()

    # ---------------------------------------------------------- observation
    def _noisy_own_position(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        oz = self.z + (float(self.rng.normal(0.0, cfg.sigma_alt_frac * cfg.sigma_loc)) * cfg.z_scale if cfg.sigma_loc > 0 else 0.0)
        if getattr(self, "gps_outage", False):      # dead reckoning in the plane; the barometer keeps working
            prev = self._dr_pos or self._last_fix or (self.x, self.y)
            self._dr_pos = (prev[0] + self.v * math.cos(self.theta) * cfg.dt,
                            prev[1] + self.v * math.sin(self.theta) * cfg.dt)
            return self._dr_pos[0], self._dr_pos[1], oz
        self._dr_pos = None
        if cfg.sigma_loc > 0:
            n = self.rng.normal(0.0, cfg.sigma_loc, size=2)
            fix = (self.x + float(n[0]), self.y + float(n[1]))
        else:
            fix = (self.x, self.y)
        self._last_fix = fix
        return fix[0], fix[1], oz

    def _sense_traffic(self):
        out, s = [], self.cfg.sigma_sense
        rng_max = self.cfg.traffic_sense_range
        for i, o in enumerate(self.dynamic_obstacles):
            d = math.sqrt((o.x - self.x) ** 2 + (o.y - self.y) ** 2 + (o.z - self.z) ** 2)
            if d > rng_max:
                continue
            sd = s * (0.5 + d / rng_max)
            if sd > 0:
                n = self.rng.normal(0.0, sd, size=3)
                out.append((i, o.x + float(n[0]), o.y + float(n[1]), o.z + 0.5 * float(n[2]), o))
            else:
                out.append((i, o.x, o.y, o.z, o))
        return out

    def _wind_at(self, x: float, y: float, z: float, t: float, gust: bool) -> Tuple[float, float]:
        if self.wind is None:
            return 0.0, 0.0
        if hasattr(self.wind, "sample3d"):
            return self.wind.sample3d(x, y, z / self.cfg.z_scale, t, gust)
        w = self.wind.sample(x, y, t) if gust else self.wind.structured(x, y, t)
        f = 1.0 + self.cfg.alt_wind_gain * ((z - self.cfg.z_min) / self.cfg.band - 0.5)
        return w[0] * f, w[1] * f

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        gx, gy = cfg.goal
        ox, oy, oz = self._noisy_own_position()

        dirs = ray_dirs(self.theta, cfg.num_rays, math.radians(cfg.ray_cone_deg), cfg.num_rays_e,
                        math.radians(cfg.ray_cone_e_deg), math.radians(cfg.ray_elev_deg))
        rays = raycast3d((self.x, self.y, self.z), dirs, self.cyl, self._sph(), cfg.world_size,
                         cfg.ray_max_range, self.terrain) / cfg.ray_max_range
        if cfg.sigma_sense > 0:
            rays = np.clip(rays + self.rng.normal(0.0, cfg.sigma_sense / cfg.ray_max_range, size=rays.shape), 0.0, 1.0)
        self._rays_obs_m = rays * cfg.ray_max_range

        sensed = self._sense_traffic()
        self.uncertainty.feed_tracks3d([(tid, sx, sy, sz) for tid, sx, sy, sz, _ in sensed], cfg.dt)

        tw = getattr(self, "_true_wind", (0.0, 0.0))
        if cfg.sigma_wind_meas > 0:
            n = self.rng.normal(0.0, cfg.sigma_wind_meas, size=2)
            wm = (tw[0] + float(n[0]), tw[1] + float(n[1]))
        else:
            wm = tw
        self.uncertainty.feed_wind_measurement(*wm)

        rep_loc = cfg.sigma_loc if cfg.reported_sigma_loc is None else cfg.reported_sigma_loc
        rep_sense = cfg.sigma_sense if cfg.reported_sigma_sense is None else cfg.reported_sigma_sense
        if getattr(self, "gps_outage", False):
            rep_loc = self.uncertainty.loc_std_max
        ucomp = self.uncertainty.index(rep_loc, rep_sense)
        closing = self._max_closing_speed(sensed)
        rcomp = self.risk.assess(self._rays_obs_m, closing, self._energy_to_go(ox, oy, oz), self.battery_wh, ucomp["U"])
        self._last_uncertainty, self._last_risk = ucomp, rcomp

        dist = math.sqrt((gx - ox) ** 2 + (gy - oy) ** 2 + (self.gz - oz) ** 2)
        base = [
            (gx - ox) / cfg.world_size,
            (gy - oy) / cfg.world_size,
            (self.gz - oz) / cfg.band,
            min(1.5, dist / self._straight_line),
            (self.v - cfg.v_min) / max(1e-6, cfg.v_max - cfg.v_min),
            self.battery_wh / cfg.battery_capacity_wh,
            _wrap(math.atan2(gy - oy, gx - ox) - self.theta) / math.pi,
            (oz - cfg.z_min) / cfg.band,
            self.vz / max(1e-6, self._vz_step),
        ]
        base.extend(rays.tolist())
        sensed.sort(key=lambda t: (t[1] - ox) ** 2 + (t[2] - oy) ** 2 + (t[3] - oz) ** 2)
        vmax = max(1e-6, cfg.dynamic_speed_range[1])
        for tid, sx, sy, sz, o in sensed[:DYN_SLOTS]:
            base.extend([(sx - ox) / cfg.world_size, (sy - oy) / cfg.world_size, (sz - oz) / cfg.band,
                         o.vx / vmax, o.vy / vmax, o.vz / max(1e-6, cfg.dynamic_vz_max)])
        base.extend([0.0] * (6 * (DYN_SLOTS - min(DYN_SLOTS, len(sensed)))))
        base.extend([wm[0] / 10.0, wm[1] / 10.0, ucomp["U"], rcomp["R"]])
        self._last_wind_meas, self._last_est_pos = wm, (ox, oy, oz)
        return np.array(base, dtype=np.float32)

    def _max_closing_speed(self, sensed) -> float:
        best = 0.0
        mv = (self.v * math.cos(self.theta), self.v * math.sin(self.theta), self.vz)
        for tid, sx, sy, sz, o in sensed:
            r = (sx - self.x, sy - self.y, sz - self.z)
            d = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            if d < 1e-6:
                continue
            rv = (o.vx - mv[0], o.vy - mv[1], o.vz - mv[2])
            best = max(best, -(r[0] * rv[0] + r[1] * rv[1] + r[2] * rv[2]) / d)
        return best

    def _energy_to_go(self, ox: float, oy: float, oz: float) -> float:
        gx, gy = self.cfg.goal
        v = 0.5 * (self.cfg.v_min + self.cfg.v_max)
        dist = math.sqrt((gx - ox) ** 2 + (gy - oy) ** 2 + (self.gz - oz) ** 2)
        climb = max(0.0, self.gz - oz) / self.cfg.z_scale
        return (self._power(v) * dist / v + self.cfg.k_climb_w * climb) / 3600.0

    # ----------------------------------------------------------------- step
    def step(self, action_idx: int):
        cfg = self.cfg
        dth, dv, cl = (float(a) for a in self._action_table[int(action_idx)])
        max_turn = math.radians(cfg.max_turn_rate_deg)
        self.theta = _wrap(self.theta + float(np.clip(dth, -max_turn, max_turn)))
        self.v = float(np.clip(self.v + dv, cfg.v_min, cfg.v_max))
        nz = float(np.clip(self.z + cl * self._vz_step * cfg.dt, cfg.z_min, cfg.z_max))
        self.vz = (nz - self.z) / cfg.dt

        self._true_wind = self._wind_at(self.x, self.y, self.z, self.step_count * cfg.dt, gust=True)
        nx = self.x + (self.v * math.cos(self.theta) + self._true_wind[0]) * cfg.dt
        ny = self.y + (self.v * math.sin(self.theta) + self._true_wind[1]) * cfg.dt

        for o in self.dynamic_obstacles:
            o.step(cfg.dt, cfg.world_size, cfg.z_min, cfg.z_max, self.rng)
        sph = self._sph()
        collision = segment_hits((self.x, self.y, self.z), (nx, ny, nz), self.cyl, sph, self.terrain)
        out_of_bounds = not (0 <= nx <= cfg.world_size and 0 <= ny <= cfg.world_size)
        step_dist = math.sqrt((nx - self.x) ** 2 + (ny - self.y) ** 2 + ((nz - self.z) / cfg.z_scale) ** 2)   # true metres
        step_energy = self._power(self.v, self.vz / cfg.z_scale) * cfg.dt / 3600.0
        self.battery_wh -= step_energy

        gx, gy = cfg.goal
        prev_goal = math.sqrt((self.x - gx) ** 2 + (self.y - gy) ** 2 + (self.z - self.gz) ** 2)
        climbed = max(0.0, nz - self.z) / cfg.z_scale
        self.x, self.y, self.z = nx, ny, nz
        new_goal = math.sqrt((self.x - gx) ** 2 + (self.y - gy) ** 2 + (self.z - self.gz) ** 2)
        clearance = clearance3d((self.x, self.y, self.z), self.cyl, sph, self.terrain)
        if clearance == math.inf:
            clearance = cfg.world_size

        r_distance = -cfg.s_distance * step_dist / (cfg.v_max * cfg.dt)
        r_energy = -cfg.s_energy * step_energy / self._max_step_energy
        r_time = -cfg.s_time
        deficit = max(0.0, 1.0 - clearance / cfg.safe_clearance)
        r_safety = -cfg.s_safety * 3.0 * deficit ** 2
        r_vec = np.array([r_distance, r_energy, r_time, r_safety], dtype=np.float32)
        shared = -cfg.r_step + cfg.progress_gain * (prev_goal - new_goal) / (cfg.v_max * cfg.dt)

        m = self.metrics
        m.path_length += step_dist
        m.energy_wh += step_energy
        m.time_s += cfg.dt
        m.climb_m += climbed
        m.min_clearance = min(m.min_clearance, clearance)
        m.steps += 1
        m.trajectory.append((self.x, self.y, self.z))
        m.u_history.append(self._last_uncertainty["U"])
        m.r_history.append(self._last_risk["R"])
        m.battery_history.append(self.battery_wh / cfg.battery_capacity_wh)

        done = False
        info = {"collision": collision, "out_of_bounds": out_of_bounds, "reached_goal": False,
                "battery_exhausted": False, "clearance": clearance,
                "uncertainty": dict(self._last_uncertainty), "risk": dict(self._last_risk),
                "battery_frac": self.battery_wh / cfg.battery_capacity_wh}
        # terminal penalties live in the objective vector, so the safety and energy weights are real risk dials
        if collision or out_of_bounds:
            r_vec[3] -= NUM_OBJECTIVES * cfg.r_collision
            m.collision = True
            done = True
        elif new_goal <= cfg.goal_radius:
            shared += cfg.r_goal
            m.success = True
            info["reached_goal"] = True
            done = True
        elif self.battery_wh <= 0.0:
            r_vec[1] -= NUM_OBJECTIVES * cfg.r_battery
            m.battery_exhausted = True
            info["battery_exhausted"] = True
            done = True
        elif self.step_count >= cfg.max_steps - 1:
            done = True
        self.step_count += 1
        return self._observe(), r_vec, float(shared), done, info


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
