# -*- coding: utf-8 -*-
"""
tools/viz_p2_sim.py —— P2 闭环避障结果

A/B. 两个场地：栅格 + 规划路径 + 【真值轨迹】
     ★ 关键看：真值轨迹（实际走的）和规划路径（打算走的）差多少
       —— 差得越多，说明路径跟踪越差、离障碍越近
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/mnt/d/迷雾通/nav'
CASES = [('corridor', 'A. Wall with a gap'),
         ('utrap', 'B. U-shaped trap')]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, (label, title) in zip(axes, CASES):
        p = os.path.join(D, 'p2_%s.npz' % label)
        if not os.path.exists(p):
            ax.text(0.5, 0.5, 'no data for %s' % label, ha='center',
                    transform=ax.transAxes)
            continue
        d = np.load(p)
        occ, origin, res = d['occ'], d['origin'], float(d['res'])
        traj, wp = d['traj'], d['waypoints']
        m = d['meta']
        sx, sy, gx, gy, err, infl, hit, clear = m

        ny, nx = occ.shape
        goc = np.zeros_like(occ, dtype=float)
        goc[occ] = 1.0
        ax.imshow(goc, origin='lower', cmap='Greys', vmin=0, vmax=1.3,
                  extent=[origin[0], origin[0] + nx * res,
                          origin[1], origin[1] + ny * res], zorder=1)

        if len(wp):
            ax.plot(wp[:, 0], wp[:, 1], color='tab:red', lw=2.5, ls='--',
                    zorder=4, label='planned path (%d pts)' % len(wp))
        if len(traj):
            ax.plot(traj[:, 1], traj[:, 2], color='tab:blue', lw=2.2,
                    zorder=5, label='TRUE trajectory')
            ax.plot(traj[0, 1], traj[0, 2], 'o', color='lime', ms=13,
                    mec='k', zorder=6, label='start')
            ax.plot(traj[-1, 1], traj[-1, 2], 'o', color='tab:blue', ms=9,
                    mec='k', zorder=6, label='final')
        ax.plot(gx, gy, '*', color='gold', ms=22, mec='k', zorder=6, label='goal')

        ax.set_title('%s\nfinal error %.3f m | min clearance %.3f m | '
                     'obstacle hits %d' % (title, err, clear, int(hit)),
                     fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.25)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    plt.tight_layout()
    out = os.path.join(D, 'tools', 'p2_sim.png')
    plt.savefig(out, dpi=110)
    print('saved %s' % out)


if __name__ == '__main__':
    main()
