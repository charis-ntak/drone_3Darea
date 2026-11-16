"""drone_3Darea: a 3D simulation area for drone path planning under uncertainty.

Tactical worlds with static volumes, terrain and moving traffic, and real low-level areas over Greece
built from airspace zones, Copernicus terrain and Open-Meteo wind.
"""
from .config import DroneEnvConfig, DronePlatform, PLATFORMS, OBJECTIVE_NAMES, NUM_OBJECTIVES, escalation
from .env import DroneEnv3D, Uncertainty3D, EpisodeMetrics3D
from .topologies import FAMILIES, TRAIN_FAMILIES
from .areas import AREAS, Area, load_area
from .terrain import TerrainGrid

__version__ = "0.1.0"
__all__ = ["DroneEnvConfig", "DronePlatform", "PLATFORMS", "OBJECTIVE_NAMES", "NUM_OBJECTIVES", "escalation",
           "DroneEnv3D", "Uncertainty3D", "EpisodeMetrics3D", "FAMILIES", "TRAIN_FAMILIES",
           "AREAS", "Area", "load_area", "TerrainGrid"]
