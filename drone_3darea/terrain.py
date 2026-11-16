"""Terrain elevation for the simulation.

Source: Open-Meteo Elevation API (free, no API key), which serves the Copernicus DEM GLO-90
(about 90 m resolution). A regular lat/lon grid over a bounding box is fetched once (100 points per
request) and cached to data/terrain/<name>.json, so that later runs are offline and reproducible.
If the network is unavailable and no cache exists, a flat sea-level grid is returned and flagged
as synthetic, in the same spirit as the weather fallback.

    from drone_3darea.terrain import TerrainGrid
    t = TerrainGrid.for_bbox("saronic", (37.60, 38.10, 23.25, 23.85), n=41)
    t.elevation(37.70, 23.49)        # metres above mean sea level
    t.max_between(lat1, lon1, lat2, lon2)   # highest ground under a straight leg
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
DATA_DIR = Path(__file__).resolve().parent / "data" / "terrain"
BATCH = 100
PACE_S = 11.0


@dataclass
class TerrainGrid:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    elev: np.ndarray            # (n_lat, n_lon), metres AMSL, row 0 = lat_min; sea = 0
    synthetic: bool = False
    source: str = "Open-Meteo Elevation API (Copernicus DEM GLO-90)"

    # ------------------------------------------------------------- construction
    @classmethod
    def for_bbox(cls, name: str, bbox: tuple[float, float, float, float], n: int = 41,
                 refresh: bool = False, timeout_s: float = 30.0) -> "TerrainGrid":
        """bbox = (lat_min, lat_max, lon_min, lon_max); n points per side."""
        path = DATA_DIR / f"{name}.json"
        if path.exists() and not refresh:
            return cls.from_json(path)
        lat_min, lat_max, lon_min, lon_max = bbox
        lats = np.linspace(lat_min, lat_max, n)
        lons = np.linspace(lon_min, lon_max, n)
        pts = [(la, lo) for la in lats for lo in lons]
        part = DATA_DIR / f"{name}.part.json"          # partial progress, so an interrupted fetch resumes
        vals: list[float] = json.loads(part.read_text()) if part.exists() and not refresh else []
        try:
            for i in range(len(vals), len(pts), BATCH):
                chunk = pts[i:i + BATCH]
                for attempt in range(8):
                    r = requests.get(ELEVATION_URL, params={
                        "latitude": ",".join(f"{p[0]:.5f}" for p in chunk),
                        "longitude": ",".join(f"{p[1]:.5f}" for p in chunk)}, timeout=timeout_s)
                    if r.status_code != 429:
                        break
                    time.sleep(65.0)                    # per-minute limit of the free tier: wait and retry
                r.raise_for_status()
                vals.extend(float(v) for v in r.json()["elevation"])
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                part.write_text(json.dumps(vals))
                time.sleep(PACE_S)                      # about 550 points per minute
            elev = np.maximum(0.0, np.array(vals).reshape(n, n))   # bathymetry is not needed: sea = 0
            grid = cls(name, lat_min, lat_max, lon_min, lon_max, elev)
            grid.to_json(path)
            part.unlink(missing_ok=True)
            return grid
        except Exception as e:              # offline fallback, clearly flagged
            print(f"[terrain] Open-Meteo unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  f"using a flat synthetic grid ({len(vals)} of {len(pts)} points are cached and will be reused)")
            return cls(name, lat_min, lat_max, lon_min, lon_max, np.zeros((n, n)), synthetic=True)

    @classmethod
    def from_json(cls, path: str | Path) -> "TerrainGrid":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["name"], raw["lat_min"], raw["lat_max"], raw["lon_min"], raw["lon_max"],
                   np.array(raw["elev"], dtype=float), raw.get("synthetic", False), raw.get("source", ""))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "name": self.name, "lat_min": self.lat_min, "lat_max": self.lat_max,
            "lon_min": self.lon_min, "lon_max": self.lon_max, "source": self.source,
            "synthetic": self.synthetic, "fetched": time.strftime("%Y-%m-%d"),
            "elev": np.round(self.elev, 1).tolist()}), encoding="utf-8")

    # ------------------------------------------------------------------ queries
    def elevation(self, lat, lon):
        """Bilinear elevation in metres AMSL (scalars or arrays)."""
        n_lat, n_lon = self.elev.shape
        gy = np.clip((np.asarray(lat, dtype=float) - self.lat_min) / (self.lat_max - self.lat_min) * (n_lat - 1),
                     0, n_lat - 1 - 1e-9)
        gx = np.clip((np.asarray(lon, dtype=float) - self.lon_min) / (self.lon_max - self.lon_min) * (n_lon - 1),
                     0, n_lon - 1 - 1e-9)
        y0, x0 = gy.astype(int), gx.astype(int)
        fy, fx = gy - y0, gx - x0
        e = self.elev
        return (e[y0, x0] * (1 - fx) * (1 - fy) + e[y0, x0 + 1] * fx * (1 - fy)
                + e[y0 + 1, x0] * (1 - fx) * fy + e[y0 + 1, x0 + 1] * fx * fy)

    def max_between(self, lat1: float, lon1: float, lat2: float, lon2: float, samples: int = 50) -> float:
        """Highest ground under the straight leg between two points."""
        f = np.linspace(0.0, 1.0, samples)
        return float(np.max(self.elevation(lat1 + f * (lat2 - lat1), lon1 + f * (lon2 - lon1))))

    def blocks(self, lat: float, lon: float, alt_m: float, margin_m: float = 30.0) -> bool:
        """True when a point at alt_m AMSL is below the ground plus a safety margin."""
        return bool(alt_m < float(self.elevation(lat, lon)) + margin_m)
