# -*- coding: utf-8 -*-
"""
tools/viz_frames.py —— P0 的可视化验收

三张图，分别对应 P0 的三条论断：
  A. 变换是对的           —— 两个坐标系 + 同一条轨迹在两边的关系
  B. 原点在哪无所谓        —— 换 3 个场景系原点，相对位移完全重合
  C. ★ 没有检查会怎样      —— 就是当年栽 4 次的那种错，画出来长什么样

画图一律用英文标签（中文在 matplotlib 里会乱码）。
"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose2D, FRAME_SCENE, FRAME_CAPO          # noqa: E402
from core.frames import build_T_scene_capo, to_scene_pose, SceneConvention  # noqa: E402


def draw_frame(ax, origin, yaw, scale, label, color):
    """画一个坐标系（两根轴）"""
    c, s = math.cos(yaw), math.sin(yaw)
    for ax_v, name, ls in (((c, s), 'x', '-'), ((-s, c), 'y', '--')):
        ax.annotate('', xy=(origin[0] + ax_v[0] * scale, origin[1] + ax_v[1] * scale),
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.0,
                                    linestyle=ls, shrinkA=0, shrinkB=0))
    ax.plot(*origin, 'o', color=color, ms=7, zorder=5)
    ax.text(origin[0] + 0.06, origin[1] + 0.06, label, color=color,
            fontsize=11, fontweight='bold', zorder=6)


# 一条在 capo 系里走的轨迹：前进 1m → 左转 45° → 再前进 0.7m
def make_capo_path():
    pts = []
    for t in np.linspace(0, 1.0, 25):
        pts.append(Pose2D(t, 0.0, 0.0, frame=FRAME_CAPO))
    yaw = math.radians(45)
    for t in np.linspace(0, 0.7, 20):
        pts.append(Pose2D(1.0 + t * math.cos(yaw), t * math.sin(yaw), yaw,
                          frame=FRAME_CAPO))
    return pts


def main():
    capo_start = Pose2D(0.0, 0.0, 0.0, frame=FRAME_CAPO)
    path = make_capo_path()

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    # ================================================ A. 变换是对的
    ax = axes[0]
    scene_start = Pose2D(3.2, 1.8, math.radians(90), frame=FRAME_SCENE)
    T = build_T_scene_capo(scene_start, capo_start)

    draw_frame(ax, (0, 0), 0, 0.55, 'CAPO frame\n(origin = start)', 'tab:blue')
    draw_frame(ax, (scene_start.x, scene_start.y), scene_start.yaw, 0.55,
               'SCENE frame\n(origin = NW corner)', 'tab:red')

    cp = np.array([[p.x, p.y] for p in path])
    sp = np.array([[p.x, p.y] for p in [T @ q for q in path]])
    ax.plot(cp[:, 0], cp[:, 1], color='tab:blue', lw=2.6, label='path in CAPO frame')
    ax.plot(sp[:, 0], sp[:, 1], color='tab:red', lw=2.6, label='SAME path in SCENE frame')
    for i in (0, 24, -1):
        ax.plot([cp[i, 0], sp[i, 0]], [cp[i, 1], sp[i, 1]], 'k:', lw=0.8, alpha=0.6)
    ax.set_title('A. One transform connects the two frames\n'
                 'T_scene_capo built once at startup, never changed', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    # ================================================ B. 原点在哪无所谓
    ax = axes[1]
    origins = [Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE),
               Pose2D(3.2, 1.8, math.radians(90), frame=FRAME_SCENE),
               Pose2D(-12.7, 4.4, math.radians(-137), frame=FRAME_SCENE)]
    colors = ['tab:red', 'tab:green', 'tab:purple']
    for o, col in zip(origins, colors):
        T2 = build_T_scene_capo(o, capo_start)
        rel = [ (T2 @ p).delta(o) for p in path ]      # 相对起点的位移
        rr = np.array([[p.x, p.y] for p in rel])
        ax.plot(rr[:, 0], rr[:, 1], color=col, lw=3.0, alpha=0.75,
                label='scene origin at (%.1f, %.1f, %.0f°)' %
                      (o.x, o.y, math.degrees(o.yaw)))
    ax.set_title('B. The scene origin does NOT matter\n'
                 '3 different origin definitions -> relative motion is IDENTICAL',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('forward [m]'); ax.set_ylabel('left [m]')

    # ================================================ C. 没有检查会怎样
    ax = axes[2]
    T3 = build_T_scene_capo(Pose2D(3.2, 1.8, math.radians(90), frame=FRAME_SCENE),
                            capo_start)
    good = np.array([[p.x, p.y] for p in [T3 @ q for q in path]])

    # ★ 就是当年栽的那个错：把 capo 系的位移当成场景系的位移直接用
    bad = np.array([[p.x, p.y] for p in path]) + np.array([3.2, 1.8])

    ax.plot(good[:, 0], good[:, 1], color='tab:green', lw=3.0,
            label='correct (used T)')
    ax.plot(bad[:, 0], bad[:, 1], color='tab:red', lw=3.0, ls='--',
            label='WRONG (mixed frames, no T)')
    ax.plot([good[-1, 0], bad[-1, 0]], [good[-1, 1], bad[-1, 1]],
            'k:', lw=1.4)
    err = math.hypot(good[-1, 0] - bad[-1, 0], good[-1, 1] - bad[-1, 1])
    ax.text(0.5, 0.06,
            'endpoint error = %.2f m  (and it never raised an error!)' % err,
            transform=ax.transAxes, ha='center', fontsize=10, color='tab:red',
            bbox=dict(boxstyle='round', fc='#ffecec', ec='tab:red'))
    ax.set_title('C. Why the frame label exists\n'
                 'without it: a silent 90 deg error, exactly what bit us 4 times',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    plt.tight_layout()
    out = '/mnt/d/迷雾通/nav/tools/frames_check.png'
    plt.savefig(out, dpi=110)
    print('saved %s' % out)

    # ---- 顺带把适配函数也画出来 ----
    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
    pts = np.array([[x, y] for x in (0, 1) for y in (0, 1)])
    draw_frame(ax2, (0, 0), 0, 0.5, 'internal convention\n(x=fwd, y=left)',
               'tab:blue')
    for x, y in pts:
        p = to_scene_pose(x, y, 0.0, SceneConvention(axes='yx'))
        ax2.plot([0, p.x], [0, p.y], 'o-', color='tab:orange', ms=5, lw=1.2)
        ax2.text(p.x + 0.03, p.y + 0.03, '(%.0f,%.0f)' % (x, y), fontsize=8,
                 color='tab:orange')
    ax2.set_title('Adapter: GO2 robot_odom input (x=right, y=forward)\n'
                  '-> our convention (x=forward, y=left)', fontsize=11)
    ax2.set_aspect('equal'); ax2.grid(alpha=0.3)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('/mnt/d/迷雾通/nav/tools/adapter_check.png', dpi=110)
    print('saved adapter_check.png')


if __name__ == '__main__':
    main()
