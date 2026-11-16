"""3D route figures: a perspective view of the world with the flown routes and an altitude profile.

    from drone_3darea.plot import plot_routes
    plot_routes(env, [dict(label="planner A", xyz=env.metrics.trajectory)], "route.png")

Altitudes are drawn in true metres (stretched worlds are converted back with cfg.z_scale).
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .topologies import BIG

# colourblind-safe palette (Okabe and Ito)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
STYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def _cylinder(ax, x, y, r, z0, z1, color, alpha, unit, W):
    th = np.linspace(0, 2 * np.pi, 40)
    zz = np.array([z0, z1])
    T, Z = np.meshgrid(th, zz)
    cx = lambda a: np.clip(x + r * np.cos(a), 0, W) / unit      # volumes are cut at the edge of the world
    cy = lambda a: np.clip(y + r * np.sin(a), 0, W) / unit
    ax.plot_surface(cx(T), cy(T), Z, color=color, alpha=alpha, linewidth=0, shade=True, zorder=2)
    for z in (z0, z1):
        ax.plot(cx(th), cy(th), z, color=color, lw=0.6, alpha=0.8, zorder=3)


def plot_routes(env, routes: List[dict], path: str, title: str = "", elev: float = 28.0, azim: float = -58.0,
                moving: bool = True, dpi: int = 300, zone_labels: Optional[List[dict]] = None):
    """env: a DroneEnv3D after at least one flight (its world is drawn); routes: list of dict(label, xyz)."""
    cfg = env.cfg
    zs = cfg.z_scale
    W = cfg.world_size
    km = W > 5000
    unit = 1000.0 if km else 1.0
    lab = "km" if km else "m"
    zlo, zhi = cfg.z_min / zs, cfg.z_max / zs
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig = plt.figure(figsize=(15.5, 5.0))
    ax = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
    axm = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    if env.terrain is not None and env.terrain.max > 0:
        n = env.terrain.n
        g = np.linspace(0, W, n)
        X, Y = np.meshgrid(g, g)
        H = np.minimum(env.terrain.h / zs, zhi * 1.15)
        ax.plot_surface(X / unit, Y / unit, H, cmap="terrain", vmin=-0.25 * zhi, vmax=1.6 * zhi, alpha=0.7,
                        linewidth=0, antialiased=True, rstride=1, cstride=1, zorder=1)
        land = np.ma.masked_less_equal(env.terrain.h / zs, 0.5)
        axm.set_facecolor("#DCEBFA")
        cf = axm.contourf(X / unit, Y / unit, land, levels=12, cmap="YlOrBr", alpha=0.9)
        cb = fig.colorbar(cf, ax=axm, fraction=0.046, pad=0.02)
        cb.set_label("terrain (m)", fontsize=8)
        blocked = np.ma.masked_less(env.terrain.h / zs, zhi)
        if blocked.count():
            axm.contourf(X / unit, Y / unit, blocked, levels=1, colors=["#5D4037"], alpha=0.9)
    for x, y, r, a, b in env.cyl:
        top = zhi if b >= BIG else min(zhi, b / zs)
        bot = max(0.0, a / zs)
        kind = "#B22222" if (b >= BIG and a <= 0) else "#8E44AD" if a > 0 else "#7F8C8D"
        _cylinder(ax, x, y, r, bot, top, kind, 0.22, unit, W)
        axm.add_patch(plt.Circle((x / unit, y / unit), r / unit, facecolor=kind, edgecolor=kind, alpha=0.25, lw=1.0))
        if not km:
            axm.text(x / unit, y / unit, "full" if (b >= BIG and a <= 0) else f"from {bot:.0f} m" if a > 0 else f"to {top:.0f} m",
                     ha="center", va="center", fontsize=6.5, color="#333333")
    if moving:
        for o in env.dynamic_obstacles:
            ax.scatter([o.x / unit], [o.y / unit], [o.z / zs], s=30, color="#555555", marker="^", depthshade=False, zorder=4)
            axm.scatter([o.x / unit], [o.y / unit], s=28, color="#555555", marker="^", zorder=4)

    for i, r in enumerate(routes):
        p = np.asarray(r["xyz"], dtype=float)
        c, st = r.get("color", COLORS[i % len(COLORS)]), r.get("style", STYLES[i % len(STYLES)])
        ax.plot(p[:, 0] / unit, p[:, 1] / unit, p[:, 2] / zs, st, color=c, lw=2.0, label=r["label"], zorder=10)
        axm.plot(p[:, 0] / unit, p[:, 1] / unit, st, color=c, lw=1.8, label=r["label"], zorder=5)
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))])
        ax2.plot(d / unit, p[:, 2] / zs, st, color=c, lw=1.8, label=r["label"])
        if env.terrain is not None and env.terrain.max > 0 and i == 0:
            ax2.fill_between(d / unit, 0, env.terrain.height(p[:, 0], p[:, 1]) / zs, color="#8B7355", alpha=0.45,
                             label="terrain under the first route")
    sx, sy = cfg.start
    gx, gy = cfg.goal
    ax.scatter([sx / unit], [sy / unit], [env.metrics.trajectory[0][2] / zs if not routes else np.asarray(routes[0]["xyz"])[0, 2] / zs],
               s=45, color="#2E8B57", marker="o", depthshade=False, label="start", zorder=11)
    ax.scatter([gx / unit], [gy / unit], [env.gz / zs], s=70, color="#DAA520", marker="*", depthshade=False, label="goal", zorder=11)
    axm.scatter([sx / unit], [sy / unit], s=45, color="#2E8B57", marker="o", zorder=6)
    axm.scatter([gx / unit], [gy / unit], s=90, color="#DAA520", marker="*", edgecolor="k", linewidth=0.4, zorder=6)
    axm.set_xlim(0, W / unit); axm.set_ylim(0, W / unit); axm.set_aspect("equal")
    axm.set_xlabel(f"east ({lab})"); axm.set_ylabel(f"north ({lab})")
    axm.set_title("plan view (red: full height, purple: elevated, grey: overflyable; triangles: traffic)", fontsize=7.5)
    ax.set_xlim(0, W / unit); ax.set_ylim(0, W / unit); ax.set_zlim(0, zhi * 1.05)
    ax.set_xlabel(f"east ({lab})"); ax.set_ylabel(f"north ({lab})"); ax.set_zlabel("altitude (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.45))
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax2.axhline(zlo, color="#999999", lw=0.8, ls=":"); ax2.axhline(zhi, color="#999999", lw=0.8, ls=":")
    ax2.set_xlabel(f"distance along the ground track ({lab})"); ax2.set_ylabel("altitude (m)")
    ax2.set_ylim(0, zhi * 1.08)
    ax2.legend(fontsize=7, frameon=False, loc="best")
    ax2.grid(alpha=0.25)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
