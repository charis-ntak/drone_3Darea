import math

import numpy as np

from drone_3darea import DroneEnv3D, escalation, load_area, TRAIN_FAMILIES
from drone_3darea.geom3d import clearance3d, ray_dirs, raycast3d, segment_hits
from drone_3darea.topologies import BIG

CYL = np.array([[100.0, 0.0, 20.0, 0.0, 50.0],       # tower, top at 50 m
                [200.0, 0.0, 20.0, 80.0, BIG]])      # elevated volume, floor at 80 m
NONE = np.zeros((0, 4))
AHEAD = np.array([[1.0, 0.0, 0.0]])


def test_rays():
    assert abs(raycast3d((0, 0, 30), AHEAD, CYL, NONE, 1e6, 500.0)[0] - 80.0) < 1e-6     # tower side
    assert abs(raycast3d((0, 0, 60), AHEAD, CYL, NONE, 1e6, 500.0)[0] - 500.0) < 1e-6    # over the tower, under the volume
    assert abs(raycast3d((0, 0, 100), AHEAD, CYL, NONE, 1e6, 500.0)[0] - 180.0) < 1e-6   # elevated volume
    e = math.radians(25)
    t = raycast3d((60, 0, 70), np.array([[math.cos(e), 0.0, -math.sin(e)]]), CYL, NONE, 1e6, 500.0)[0]
    assert abs((70 - t * math.sin(e)) - 50.0) < 1e-6                                       # enters through the top
    sph = np.array([[50.0, 0.0, 30.0, 10.0]])
    assert abs(raycast3d((0, 0, 30), AHEAD, np.zeros((0, 5)), sph, 1e6, 500.0)[0] - 40.0) < 1e-6
    assert ray_dirs(0.0).shape == (16, 3)


def test_clearance_and_collision():
    assert abs(clearance3d((100, 0, 60), CYL, NONE) - 10.0) < 1e-9
    assert abs(clearance3d((200, 0, 70), CYL, NONE) - 10.0) < 1e-9
    assert segment_hits((70, 0, 30), (90, 0, 30), CYL, NONE)
    assert not segment_hits((70, 0, 60), (130, 0, 60), CYL, NONE)


def test_environment_runs_in_every_family():
    for fam in TRAIN_FAMILIES:
        env = DroneEnv3D(escalation(4, fam), seed=0)
        obs = env.reset()
        assert obs.shape == (env.state_dim,) and env.state_dim == 41 and env.num_actions == 27
        done, n = False, 0
        while not done and n < 50:
            obs, r_vec, shared, done, info = env.step(13)      # hold heading, speed and level
            assert r_vec.shape == (4,) and np.all(np.isfinite(obs))
            n += 1


def test_real_areas_load():
    for name in ("saronic", "south_evia"):
        area = load_area(name)
        env = area.make_env(seed=0)
        obs = env.reset()
        assert obs.shape == (41,) and np.all(np.isfinite(obs))
        assert not area.meta["terrain_synthetic"] and area.elevation.max() > 300
        z0 = area.cfg.start_z / area.cfg.z_scale
        assert 30.0 <= z0 <= 450.0
