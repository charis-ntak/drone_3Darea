"""Fly the reactive pilot in a tactical world and in the real areas, and save 3D route figures.
    python examples/demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from reactive_pilot import fly
from drone_3darea import DroneEnv3D, escalation, load_area
from drone_3darea.plot import plot_routes

os.makedirs("output", exist_ok=True)

env = DroneEnv3D(escalation(level=2, topology="mixed"), seed=3)
m = fly(env)
print(f"tactical mixed, level 2: success {m.success}, path {m.path_length:.0f} m, energy {m.energy_wh:.2f} Wh, "
      f"climb {m.climb_m:.0f} m, minimum clearance {m.min_clearance:.0f} m")
plot_routes(env, [dict(label="reactive pilot", xyz=m.trajectory)], "output/tactical_mixed.png",
            "Tactical world, mixed topology, uncertainty level 2")

for name in ("saronic", "south_evia", "thermaikos"):
    area = load_area(name)
    env = area.make_env(seed=1)
    m = fly(env)
    zs = area.cfg.z_scale
    alt = np.array([p[2] for p in m.trajectory]) / zs
    print(f"{area.title}: world {area.frame.world_size / 1000:.1f} km, straight line {area.meta['straight_km']:.1f} km, "
          f"{len(area.zones)} zones, terrain to {area.elevation.max():.0f} m, success {m.success}, "
          f"path {m.path_length / 1000:.1f} km, altitude {alt.min():.0f} to {alt.max():.0f} m, "
          f"energy {m.energy_wh:.0f} of {area.cfg.battery_capacity_wh:.0f} Wh")
    plot_routes(env, [dict(label="reactive pilot", xyz=m.trajectory)], f"output/{name}.png", area.title)
