"""Configuration of the 3D drone environment.

Tactical worlds are 1000 m x 1000 m with a flight band of 30 to 150 m, a small multirotor flying at
5 to 15 m/s and one decision per second. Real areas (drone_3darea.areas) reuse the same dataclass with
their own size, band and decision interval. A climb or descent command moves the vehicle by `kappa_v`
of the band per decision, so the vertical agility relative to the band is the same at every scale.

Objectives (4): distance, energy, time, safety.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

NUM_OBJECTIVES = 4
OBJECTIVE_NAMES = ("distance", "energy", "time", "safety")


@dataclass
class DronePlatform:
    """Vehicle characteristics. Power model: P = p_hover + c_p v^3 + k_climb max(vz, 0) - k_descend max(-vz, 0)."""
    name: str
    v_min: float
    v_max: float
    p_hover_w: float
    c_p: float
    k_climb_w: float
    k_descend_w: float
    max_wind_ms: float
    max_gust_ms: float


PLATFORMS = {
    # the small multirotor used for every default setting
    "quadrotor_small": DronePlatform("Small quadrotor (2.5 kg class)", 5.0, 15.0, 150.0, 0.045, 35.0, 10.0, 10.0, 14.0),
    "multirotor_delivery": DronePlatform("Delivery multirotor (5 kg payload class)", 6.0, 16.0, 420.0, 0.10, 110.0, 30.0, 10.0, 14.0),
    "vtol_longrange": DronePlatform("Fixed-wing VTOL (long-range survey or cargo)", 14.0, 25.0, 260.0, 0.030, 130.0, 35.0, 14.0, 18.0),
}


@dataclass
class DroneEnvConfig:
    # --- world and mission ---
    world_size: float = 1000.0
    start: Tuple[float, float] = (100.0, 100.0)
    goal: Tuple[float, float] = (900.0, 900.0)
    random_geometry: bool = False        # True: start, goal and their altitudes are drawn at every reset
    goal_radius: float = 30.0
    z_min: float = 30.0                  # flight band, metres (stretched units in real areas, see z_scale)
    z_max: float = 150.0
    start_z: Optional[float] = None      # None -> z_min
    goal_z: Optional[float] = None
    z_scale: float = 1.0                 # vertical stretch of the geometry: true altitude = z / z_scale

    # --- kinematics ---
    v_min: float = 5.0
    v_max: float = 15.0
    dv: float = 1.0
    dtheta_deg: float = 15.0
    max_turn_rate_deg: float = 30.0
    kappa_v: float = 0.05                # vertical step per decision, fraction of the band
    dt: float = 1.0
    max_steps: int = 400

    # --- battery and power ---
    p_hover_w: float = 150.0
    c_p: float = 0.045
    k_climb_w: float = 35.0              # W per m/s of climb (about m g / efficiency for 2.5 kg)
    k_descend_w: float = 10.0            # W recovered per m/s of descent
    battery_capacity_wh: float = 11.5
    battery_start_frac: float = 1.0
    low_battery_frac: float = 0.30

    # --- static obstacles ---
    topology: str = "mixed"              # see drone_3darea.topologies.FAMILIES
    num_static: Optional[int] = None     # None -> the family default
    obstacle_radius_range: Tuple[float, float] = (40.0, 80.0)
    obstacle_min_separation: float = 20.0
    clearance_from_endpoints: float = 60.0
    block_diagonal_prob: float = 0.7
    init_heading_spread_deg: float = 90.0

    # --- moving obstacles (other traffic) ---
    num_dynamic_obstacles: int = 3
    dynamic_radius_range: Tuple[float, float] = (25.0, 45.0)
    dynamic_speed_range: Tuple[float, float] = (3.0, 8.0)
    dynamic_turn_std_deg: float = 0.0    # heading random walk per step (unpredictability)
    dynamic_vz_std: float = 0.0          # vertical-speed random walk per step
    dynamic_vz_max: float = 2.0

    # --- sensing ---
    traffic_sense_range: float = 400.0
    num_rays: int = 8                    # horizontal fan
    ray_cone_deg: float = 180.0
    num_rays_e: int = 4                  # rays in the upper fan and in the lower fan
    ray_elev_deg: float = 25.0
    ray_cone_e_deg: float = 120.0
    ray_max_range: float = 250.0

    # --- wind ---
    wind_mean_max: float = 0.0
    wind_modes: int = 3
    wind_time_freq: float = 0.0
    gust_std: float = 0.0
    alt_wind_gain: float = 0.6           # wind factor 1 - g/2 at z_min rising to 1 + g/2 at z_max

    # --- injected noise (standard deviations of zero-mean Gaussian noise; 0 = perfect information) ---
    sigma_loc: float = 0.0               # own position, m
    sigma_alt_frac: float = 0.25         # barometric altitude noise as a fraction of sigma_loc
    sigma_sense: float = 0.0             # obstacle ranges and traffic positions, m
    sigma_wind_meas: float = 0.0         # anemometer, m/s
    reported_sigma_loc: Optional[float] = None     # accuracy the receiver reports (None = honest)
    reported_sigma_sense: Optional[float] = None

    # --- safety margin and rewards ---
    safe_clearance: float = 40.0
    r_goal: float = 200.0
    r_collision: float = 200.0
    r_battery: float = 100.0
    r_step: float = 0.05
    progress_gain: float = 0.8
    s_distance: float = 1.0
    s_energy: float = 1.0
    s_time: float = 1.0
    s_safety: float = 1.0

    @property
    def band(self) -> float:
        return self.z_max - self.z_min

    def use_platform(self, name: str) -> "DroneEnvConfig":
        p = PLATFORMS[name]
        self.v_min, self.v_max, self.p_hover_w, self.c_p = p.v_min, p.v_max, p.p_hover_w, p.c_p
        self.k_climb_w, self.k_descend_w = p.k_climb_w, p.k_descend_w
        return self


def escalation(level: int, topology: str = "mixed", num_dynamic: Optional[int] = None,
               safe_clearance: float = 40.0) -> DroneEnvConfig:
    """Tactical world at uncertainty level 0..4: every noise source scales with level / 4."""
    from .topologies import FAMILIES
    lv = max(0, min(4, int(level)))
    f = lv / 4.0
    fam = FAMILIES[topology]
    return DroneEnvConfig(
        topology=topology, random_geometry=True,
        num_dynamic_obstacles=fam["n_dyn"] if num_dynamic is None else num_dynamic,
        obstacle_radius_range=fam["radius"],
        dynamic_turn_std_deg=20.0 * f, dynamic_vz_std=0.6 * f,
        wind_mean_max=6.0 * f, gust_std=3.0 * f,
        sigma_loc=20.0 * f, sigma_sense=25.0 * f, sigma_wind_meas=1.5 * f,
        safe_clearance=safe_clearance,
    )
