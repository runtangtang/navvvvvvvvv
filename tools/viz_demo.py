# -*- coding: utf-8 -*-
"""
tools/viz_demo.py —— 端到端任务演示的可视化

一张图说清三件事：
  ① 组1 给的场景图（我们自己雷达建的房间）+ 它规定的场景系原点
  ② 组2 给的任务序列（两个导航点 + 一个"不归我们"的握手）
  ③ 我们真的走出来的轨迹（真值）
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/mnt/d/迷雾通/nav'


def main():
    d = np.load(os.path.join(D, 'demo', 'demo_result.npz'))
    traj = d['traj']
    occ = d['occ'].astype(bool)
    origin = d['origin']
    res = float(d['res'])
    so = d['scene_origin']
    tasks = d['tasks']            # (sx, sy, syaw, gx, gy, gyaw) × N

    fig, ax = plt.subplots(figsize=(11, 9.5))
    ny, nx = occ.shape
    ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
              extent=[origin[0], origin[0] + nx * res,
                      origin[1], origin[1] + ny * res], zorder=1)

    # ---- 场景系坐标轴 ----
    ax.plot(0, 0, 'o', color='crimson', ms=11, mec='k', zorder=6,
            label='SCENE origin  (declared by Group 1)')
    L = 1.0
    for v, nm in (((L, 0), 'x'), ((0, L), 'y')):
        ax.annotate('', xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='crimson',
                                    lw=2.2, shrinkA=0, shrinkB=0))
        ax.text(v[0] * 1.08, v[1] * 1.08, nm, color='crimson',
                fontsize=12, fontweight='bold')

    # ---- 任务点 ----
    for i, (sx, sy, syaw, gx, gy, gyaw) in enumerate(tasks, 1):
        ax.plot(sx, sy, 'o', color='lime', ms=13, mec='k', zorder=7,
                label='task start' if i == 1 else None)
        ax.plot(gx, gy, '*', color='gold', ms=24, mec='k', zorder=7,
                label='task goal' if i == 1 else None)
        ax.annotate('T%d  (%.0f,%.0f)' % (i, gx, gy), (gx, gy),
                    textcoords='offset points', xytext=(8, -14),
                    fontsize=10, color='k', fontweight='bold', zorder=8)
        # 起点朝向的小箭头
        ax.annotate('', xy=(sx + 0.5 * np.cos(syaw), sy + 0.5 * np.sin(syaw)),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='-|>', color='green', lw=1.8))

    # ---- 真值轨迹 ----
    ax.plot(traj[:, 1], traj[:, 2], color='tab:blue', lw=2.6, zorder=5,
            label='TRUE trajectory (what we actually drove)')
    ax.plot(traj[0, 1], traj[0, 2], 'o', color='tab:blue', ms=9, mec='k',
            zorder=6)

    ax.set_title('End-to-end mission demo\n'
                 'Group 1 gives the scene graph + frame; Group 2 gives poses;\n'
                 'we only output motion.  Task 2 (handshake) is not ours.',
                 fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('scene x [m]  (Group 1 frame)')
    ax.set_ylabel('scene y [m]  (Group 1 frame)')

    plt.tight_layout()
    out = os.path.join(D, 'tools', 'demo_result2.png')
    plt.savefig(out, dpi=110)
    print('saved %s' % out)


if __name__ == '__main__':
    main()
