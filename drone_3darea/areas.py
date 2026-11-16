"""Real areas: airspace volumes, terrain and wind over Greek low-level airspace, as 3D worlds.

An area is defined by a start and a goal in latitude and longitude. The square frame around them is
mapped to local metres, and the world is assembled from three data layers:
  * airspace zones (data/airspace_gr.json: CTR, TMA, prohibited, restricted and danger areas with their
    lower and upper limits; representative geometry) -> static cylinders with a floor and a top,
  * terrain (data/terrain/<area>.json: Copernicus DEM GLO-90 through the Open-Meteo Elevation API,
    fetched once and cached) -> heightfield,
  * wind (data/wind/<area>.json: Open-Meteo wind at 10 m and 120 m with gusts, fetched once and cached;
    a flagged synthetic field when no cache exists) -> altitude-dependent wind field.

Scale. Lateral separations in a real area are tens of times larger than vertical ones, so the vertical
axis is stretched by z_scale = 0.12 W / band: the flight band then has the same proportion to the world
as in the tactical worlds (120 m over 1000 m), every geometric test runs in the stretched space, and
energy, wind and reported altitudes use true metres (z / z_scale). Lengths that depend on scale (ranges,
margins, radii, decision interval) are the tactical values multiplied by s = 0.8 W / 1000.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DroneEnvConfig
from .env import DroneEnv3D, Uncertainty3D
from .geom3d import Terrain
from .terrain import TerrainGrid
from .topologies import BIG

DATA = Path(__file__).resolve().parent / "data"
FT_PER_M = 3.28084

AREAS: Dict[str, dict] = {
    "saronic": dict(title="Piraeus to Aegina, Saronic Gulf", start=(37.9440, 23.6520), goal=(37.7460, 23.4280),
                    margin_deg=0.10, note="medical delivery over water, between the Athens zones and Aegina"),
    "south_evia": dict(title="Marmari to Karystos, south Evia", start=(38.0480, 24.3200), goal=(38.0160, 24.4210),
                       margin_deg=0.10, note="coastal inspection sortie across hilly terrain"),
}


@dataclass
class Frame:
    lat0: float
    lon0: float
    m_per_deg_lat: float
    m_per_deg_lon: float
    world_size: float

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return (lon - self.lon0) * self.m_per_deg_lon, (lat - self.lat0) * self.m_per_deg_lat

    def to_ll(self, x, y):
        return self.lat0 + np.asarray(y) / self.m_per_deg_lat, self.lon0 + np.asarray(x) / self.m_per_deg_lon

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        la, lo = self.to_ll(self.world_size, self.world_size)
        return self.lat0, float(la), self.lon0, float(lo)


def make_frame(start: Tuple[float, float], goal: Tuple[float, float], margin_deg: float) -> Frame:
    lat_min, lat_max = min(start[0], goal[0]) - margin_deg, max(start[0], goal[0]) + margin_deg
    lon_min, lon_max = min(start[1], goal[1]) - margin_deg, max(start[1], goal[1]) + margin_deg
    m_lat = 110_540.0
    m_lon = 111_320.0 * math.cos(math.radians(0.5 * (lat_min + lat_max)))
    size = max((lat_max - lat_min) * m_lat, (lon_max - lon_min) * m_lon)
    return Frame(lat_min, lon_min, m_lat, m_lon, size)


def zone_cylinders(frame: Frame, z_max_true: float) -> List[dict]:
    """Airspace zones that intrude into the frame below z_max_true, as cylinders in true metres.
    Circular zones map directly; a polygon is covered by two circles along its longer axis."""
    out = []
    W = frame.world_size
    for z in json.loads((DATA / "airspace_gr.json").read_text(encoding="utf-8"))["zones"]:
        lo_m, hi_m = z["lower_ft"] / FT_PER_M, z["upper_ft"] / FT_PER_M
        if lo_m >= z_max_true:            # the zone starts above the flight band (for example a TMA floor)
            continue
        g = z["geometry"]
        circles = []
        if g["kind"] == "circle":
            cx, cy = frame.to_xy(*g["center"])
            circles.append((cx, cy, g["radius_km"] * 1000.0))
        else:
            pts = np.array([frame.to_xy(la, lo) for la, lo in g["vertices"]])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            span = hi - lo
            k = int(np.argmax(span))
            for f in (0.25, 0.75):
                c = lo + span * 0.5
                c[k] = lo[k] + f * span[k]
                circles.append((float(c[0]), float(c[1]), float(0.6 * span[1 - k])))
        for cx, cy, r in circles:
            if math.hypot(max(-cx, 0.0, cx - W), max(-cy, 0.0, cy - W)) < r:
                out.append(dict(id=z["id"], name=z["name"], type=z["type"], x=cx, y=cy, r=r,
                                z_lo=lo_m, z_hi=hi_m if hi_m < z_max_true else BIG))
    return out


class AreaWind:
    """Wind from a cached grid of Open-Meteo samples: bilinear in the plane, linear between the 10 m and
    the 120 m values in altitude (held above 120 m), with zero-mean Gaussian gusts whose spread follows
    the reported gust excess. `extra_gust` adds gust spread for the uncertainty levels."""

    def __init__(self, grid: dict, seed: int = 0, extra_gust: float = 0.0):
        n = self.n = grid["n"]
        self.cell = grid["world_size"] / n
        self.v10, self.v120, self.gust = np.zeros((n, n, 2)), np.zeros((n, n, 2)), np.zeros((n, n))
        for p in grid["points"]:
            ix, iy = min(n - 1, int(p["x"] / self.cell)), min(n - 1, int(p["y"] / self.cell))
            for arr, sp, di in ((self.v10, p["speed10"], p["dir10"]), (self.v120, p["speed120"], p["dir120"])):
                a = math.radians(di)                       # direction the wind blows FROM
                arr[iy, ix] = (-sp * math.sin(a), -sp * math.cos(a))
            self.gust[iy, ix] = 0.5 * max(0.0, p["gust10"] - p["speed10"])
        self.extra_gust, self.seed = extra_gust, seed
        self._rng = np.random.default_rng(seed)
        self._last_gust = (0.0, 0.0)
        self.synthetic = bool(grid.get("synthetic", False))

    def _interp(self, arr, x, y):
        gx, gy = np.clip(x / self.cell - 0.5, 0, self.n - 1), np.clip(y / self.cell - 0.5, 0, self.n - 1)
        x0, y0 = int(gx), int(gy)
        x1, y1 = min(x0 + 1, self.n - 1), min(y0 + 1, self.n - 1)
        fx, fy = gx - x0, gy - y0
        return (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
                + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)

    def sample3d(self, x, y, z_true, t=0.0, gust=True):
        f = min(1.0, max(0.0, (z_true - 10.0) / 110.0))
        w = (1 - f) * self._interp(self.v10, x, y) + f * self._interp(self.v120, x, y)
        if gust:
            sd = float(self._interp(self.gust, x, y)) + self.extra_gust
            g = self._rng.normal(0.0, max(1e-6, sd), size=2)
            self._last_gust = (float(g[0]), float(g[1]))
            return float(w[0] + g[0]), float(w[1] + g[1])
        return float(w[0]), float(w[1])

    def structured(self, x, y, t=0.0):
        return self.sample3d(x, y, 100.0, t, gust=False)

    def sample(self, x, y, t=0.0):
        return self.sample3d(x, y, 100.0, t, gust=True)


def fetch_wind_grid(frame: Frame, n: int = 4, timeout: float = 20.0) -> dict:
    """n x n Open-Meteo samples (current hour) over the frame."""
    import requests
    pts = []
    for iy in range(n):
        for ix in range(n):
            x, y = (ix + 0.5) * frame.world_size / n, (iy + 0.5) * frame.world_size / n
            lat, lon = frame.to_ll(x, y)
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": round(float(lat), 4), "longitude": round(float(lon), 4), "wind_speed_unit": "ms",
                "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,wind_speed_120m,wind_direction_120m",
                "forecast_days": 1}, timeout=timeout)
            r.raise_for_status()
            h = r.json()["hourly"]
            pts.append(dict(x=x, y=y, lat=float(lat), lon=float(lon), time=h["time"][0],
                            speed10=h["wind_speed_10m"][0], dir10=h["wind_direction_10m"][0],
                            gust10=h["wind_gusts_10m"][0], speed120=h["wind_speed_120m"][0],
                            dir120=h["wind_direction_120m"][0]))
    return dict(n=n, world_size=frame.world_size, points=pts, synthetic=False)


def load_wind_grid(name: str, frame: Frame, refresh: bool = False) -> dict:
    path = DATA / "wind" / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        grid = fetch_wind_grid(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(grid, indent=1), encoding="utf-8")
        return grid
    except Exception as e:                       # offline fallback, clearly flagged
        print(f"[wind] Open-Meteo unavailable ({type(e).__name__}); using a synthetic field")
        rng = np.random.default_rng(7)
        n, base = 4, rng.uniform(0, 360)
        pts = [dict(x=(ix + 0.5) * frame.world_size / n, y=(iy + 0.5) * frame.world_size / n, time="synthetic",
                    speed10=float(rng.uniform(3, 7)), dir10=float(base + rng.uniform(-25, 25)),
                    gust10=float(rng.uniform(6, 12)), speed120=float(rng.uniform(5, 10)),
                    dir120=float(base + rng.uniform(-25, 25))) for iy in range(n) for ix in range(n)]
        return dict(n=n, world_size=frame.world_size, points=pts, synthetic=True)


@dataclass
class Area:
    name: str
    title: str
    frame: Frame
    cfg: DroneEnvConfig
    zones: List[dict]                 # true metres, for reporting and plots
    cyl: np.ndarray                   # stretched units, for the environment
    terrain: Optional[Terrain]        # stretched units
    elevation: np.ndarray             # true metres on the terrain grid
    wind_grid: dict
    scale: float                      # s = 0.8 W / 1000
    extra_gust: float = 0.0
    meta: dict = field(default_factory=dict)

    def make_env(self, seed: Optional[int] = None) -> DroneEnv3D:
        s = self.scale
        est = Uncertainty3D(loc_std_max=20.0, sense_std_max=7.0 * s, wind_std_max=4.0, innov_max=17.0 * s)
        wind = AreaWind(self.wind_grid, seed=0 if seed is None else seed, extra_gust=self.extra_gust)
        return DroneEnv3D(self.cfg, seed=seed, fixed_world={"cyl": self.cyl, "terrain": self.terrain},
                          wind_field=wind, estimator=est)


def load_area(name: str, level: Optional[int] = None, band: Tuple[float, float] = (30.0, 450.0),
              n_terrain: int = 41, num_dynamic: int = 3, endpoint_agl: float = 60.0) -> Area:
    """Build the 3D world of a named area. level None = conditions as measured (GNSS-grade position,
    cached wind); level 0..4 = the uncertainty escalation of the tactical worlds, scaled to the area."""
    spec = AREAS[name]
    frame = make_frame(spec["start"], spec["goal"], spec["margin_deg"])
    W = frame.world_size
    s = 0.8 * W / 1000.0
    zs = 0.12 * W / (band[1] - band[0])
    tg = TerrainGrid.for_bbox(name, frame.bbox, n=n_terrain)
    g = np.linspace(0.0, W, n_terrain)
    X, Y = np.meshgrid(g, g)
    lat, lon = frame.to_ll(X, Y)
    elev = np.asarray(tg.elevation(lat, lon), dtype=float)
    zones = zone_cylinders(frame, band[1])
    cyl = np.array([(z["x"], z["y"], z["r"], z["z_lo"] * zs, z["z_hi"] if z["z_hi"] >= BIG else z["z_hi"] * zs)
                    for z in zones], dtype=float).reshape(-1, 5)

    def over(ll):   # altitude of an endpoint: endpoint_agl above the ground, inside the band
        e = float(tg.elevation(ll[0], ll[1]))
        return min(band[1], max(band[0], e + endpoint_agl))

    sxy, gxy = frame.to_xy(*spec["start"]), frame.to_xy(*spec["goal"])
    dist = math.hypot(gxy[0] - sxy[0], gxy[1] - sxy[1])
    f = 0.0 if level is None else max(0, min(4, int(level))) / 4.0
    cfg = DroneEnvConfig(
        world_size=W, start=sxy, goal=gxy, random_geometry=False, goal_radius=15.0 * s,
        z_min=band[0] * zs, z_max=band[1] * zs, start_z=over(spec["start"]) * zs, goal_z=over(spec["goal"]) * zs,
        z_scale=zs, dt=round(0.7 * s, 1), max_steps=400,
        battery_capacity_wh=round(2.0 * (150.0 + 0.045 * 12.0 ** 3) * (dist / 12.0) / 3600.0, 0),
        num_dynamic_obstacles=num_dynamic, dynamic_radius_range=(25.0 * s, 45.0 * s),
        dynamic_speed_range=(5.0, 12.0), dynamic_turn_std_deg=10.0 + 10.0 * f,
        dynamic_vz_std=(0.2 + 0.4 * f) * zs / 10.0, dynamic_vz_max=2.0 * zs / 10.0,
        traffic_sense_range=340.0 * s, ray_max_range=250.0 * s, clearance_from_endpoints=70.0 * s,
        safe_clearance=40.0 * s, init_heading_spread_deg=30.0,
        sigma_loc=5.0 + 15.0 * f, sigma_sense=(2.3 + 4.7 * f) * s, sigma_wind_meas=0.6 + 0.9 * f,
    )
    wind_grid = load_wind_grid(name, frame)
    return Area(name=name, title=spec["title"], frame=frame, cfg=cfg, zones=zones, cyl=cyl,
                terrain=Terrain(elev * zs, W), elevation=elev, wind_grid=wind_grid, scale=s, extra_gust=3.0 * f,
                meta=dict(band_m=band, z_scale=zs, straight_km=dist / 1000.0, terrain_source=tg.source,
                          terrain_synthetic=tg.synthetic, wind_synthetic=bool(wind_grid.get("synthetic", False)),
                          wind_time=wind_grid["points"][0].get("time", ""), note=spec.get("note", "")))
