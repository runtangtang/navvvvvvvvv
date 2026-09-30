# -*- coding: utf-8 -*-
"""
tools/viz_grid.py —— P2 规划层可视化

A. U 形陷阱：规划出的路径怎么绕进去（对比"纯贪心会一头扎进去出不来"）
B. 带缺口的墙：从缺口穿过去 + 膨胀半径的影响
C. ★ 路径平滑：117 个锯齿路点 → 5 个可用路点
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/mnt/d/迷雾通/nav')
from core.types import Pose2D, FRAME_SCENE                    # noqa: E402
from maps.grid import OccupancyGrid, make_test_world           # noqa: E402
from planning.global_planner import plan_path                  # noqa: E402


def S(x, y):
    return Pose2D(x, y, 0.0, frame=FRAME_SCENE)


def draw(ax, g, plan=None, raw=None, title='', inflate=0.35):
    gi = g.inflated(inflate)
    ny, nx = gi.occ.shape
    x0, y0 = gi.origin
    # 膨胀区半透明叠一层
    ax.imshow(gi.occ, origin='lower', cmap='Greys', vmin=0, vmax=1.6,
              extent=[x0, x0 + nx * gi.res, y0, y0 + ny * gi.res],
              alpha=0.85, zorder=1)
    if raw is not None and raw.ok:
        xs = [p.x for p in raw.waypoints]
        ys = [p.y for p in raw.waypoints]
        ax.plot(xs, ys, color='tab:orange', lw=1.0, ls=':', zorder=3,
                label='raw (8-connected, 117 pts)')
    if plan is not None and plan.ok:
        xs = [p.x for p in plan.waypoints]
        ys = [p.y for p in plan.waypoints]
        ax.plot(xs, ys, color='tab:red', lw=3.0, zorder=4,
                label='smoothed (%d pts)' % len(plan.waypoints))
        ax.plot(xs, ys, 'o', color='tab:red', ms=6, zorder=5)
    if plan is not None and plan.ok:
        ax.plot(plan.waypoints[0].x, plan.waypoints[0].y, 'o', color='lime',
                ms=13, mec='k', zorder=6, label='start')
        ax.plot(plan.waypoints[-1].x, plan.waypoints[-1].y, '*', color='gold',
                ms=22, mec='k', zorder=6, label='goal')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

    g = make_test_world('utrap')
    p = plan_path(g, S(1.5, 8.5), S(6.3, 4.2))
    draw(axes[0], g, p, title='A. U-shaped trap  (path %.2f m)\n'
                             'a greedy planner dives in and gets stuck; '
                             'the distance field gets out' % p.length)

    g2 = make_test_world('corridor')
    p2a = plan_path(g2, S(1.5, 2.5), S(11.0, 7.5), inflate_radius=0.15)
    draw(axes[1], g2, p2a, title='B. Wall with a gap  (inflate 0.15 m, %.2f m)\n'
                                'wider inflation forces a wider berth'
                                % p2a.length, inflate=0.35)

    g3 = make_test_world('utrap')
    praw = plan_path(g3, S(1.5, 8.5), S(6.3, 4.2), smooth=False)
    psm = plan_path(g3, S(1.5, 8.5), S(6.3, 4.2), smooth=True)
    draw(axes[2], g3, psm, praw,
         title='C. Path smoothing: %d zig-zag points -> %d usable\n'
               'Dijkstra paths are 8-connected staircases; following them '
               'directly makes the robot weave' % (len(praw.waypoints),
                                                   len(psm.waypoints)))

    plt.tight_layout()
    out = '/mnt/d/迷雾通/nav/tools/p2_planning.png'
    plt.savefig(out, dpi=110)
    print('saved %s' % out)


if __name__ == '__main__':
    main()
