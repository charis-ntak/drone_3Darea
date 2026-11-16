"""A simple reactive pilot used by the examples: steer to the goal, turn away and change level when the
forward rays are short. It is not a planner to compare against, only a way to fly the environment."""
import numpy as np


def reactive_pilot(env, obs):
    n_h, n_e = env.cfg.num_rays, env.cfg.num_rays_e
    rays = obs[9:9 + env.num_rays]
    h, up, dn = rays[:n_h], rays[n_h:n_h + n_e], rays[n_h + n_e:]
    err, dz = obs[6], obs[2]
    turn = 1 if err > 0.03 else -1 if err < -0.03 else 0
    climb = 1 if dz > 0.03 else -1 if dz < -0.03 else 0
    if min(h[n_h // 2 - 1], h[n_h // 2]) < 0.35:          # obstacle ahead
        turn = 1 if h[n_h // 2 + 1:].mean() > h[:n_h // 2 - 1].mean() else -1
        climb = 1 if up.mean() >= dn.mean() else -1
    elif dn.min() < 0.25 and up.mean() > dn.mean():       # terrain or a volume below: gain height
        climb = 1
    return (turn + 1) * 9 + 1 * 3 + (climb + 1)            # heading change, speed held, vertical command


def fly(env, max_tries=20):
    """Fly until one flight reaches the goal (or max_tries); returns the metrics of the last flight."""
    for _ in range(max_tries):
        obs, done = env.reset(), False
        while not done:
            obs, r_vec, shared, done, info = env.step(reactive_pilot(env, obs))
        if env.metrics.success:
            break
    return env.metrics
