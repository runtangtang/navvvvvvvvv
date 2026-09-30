#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_lab_scene.py —— 手工画实验室障碍场景（用于测窄口穿行）

坐标系（狗的起点为原点）
    +x = 狗前方      +y = 狗左方

场景（用户 2026-09-26 现场指定）
    · 狗后面   x = -0.6 全是墙
    · 狗右边   y = -0.8 全是墙
    · 左边封死 (-0.6, 1) → (1, 1) 一道墙
    · 柜子     0.4m 见方，中心 (1, 1)  ⇒ x∈[0.8,1.2] y∈[0.8,1.2]
    · 黑箱子   x = 2 (y 从 0.8 到 3) + y = 0.8 (x 从 2 到 4)
    · 前方封死 x = 4 (y 从 -0.8 到 0.8)
    · 目标     (0.5, 2.0)

★ 要害：柜子右边 (x=1.2) 和黑箱子墙 (x=2) 之间的缝只有 0.8m。
  狗宽 0.32m ⇒ 直着过富余 0.48m；但【一转弯】占宽就暴涨
  （转 45° 占 0.72m，转 90° 占 0.70m）⇒ 缝里绝不能转。
  规划膨胀 0.35 ≈ 半长，所以膨胀后自由通道只有 0.775-0.70 = 0.075m
  ⇒ ★ 这是个【压力测试】，路径几乎是一条线。
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maps.grid import OccupancyGrid      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--res', type=float, default=0.05)
ap.add_argument('--wall-thick', type=float, default=0.06,
                help='墙的厚度 [m]')
ap.add_argument('--gap-x', type=float, default=2.0,
                help='黑箱子左边那道墙的 x（调大 = 缝更宽）')
ap.add_argument('--bounds', nargs=4, type=float,
                default=[-1.5, 5.0, -1.8, 3.5],
                metavar=('X0', 'X1', 'Y0', 'Y1'))
ap.add_argument('--out', default='tools/lab_scene.npz')
ap.add_argument('--png', default=None)
a = ap.parse_args()

GX = a.gap_x          # 黑箱子左墙的 x
WT = a.wall_thick


def main():
    x0, x1, y0, y1 = a.bounds
    res = a.res
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))
    occ = np.zeros((H, W), dtype=bool)

    def fill(xa, xb, ya, yb, what):
        ca = int(math.floor((xa - x0) / res)); cb = int(math.ceil((xb - x0) / res))
        ra = int(math.floor((ya - y0) / res)); rb = int(math.ceil((yb - y0) / res))
        ca = max(0, ca); cb = min(W, cb); ra = max(0, ra); rb = min(H, rb)
        occ[ra:rb, ca:cb] = True
        print('  %-14s x[%.2f,%.2f] y[%.2f,%.2f]  → %d 格'
              % (what, xa, xb, ya, yb, (rb - ra) * (cb - ca)))

    h = WT / 2.0
    # ---- 墙（用户指定）----
    fill(-0.6 - h, -0.6 + h, y0, y1, '狗后面')
    fill(x0, x1, -0.8 - h, -0.8 + h, '狗右边')
    fill(-0.6, 1.0 + h, 1.0 - h, 1.0 + h, '左边封死')
    fill(GX - h, GX + h, 0.8, 3.0, '黑箱子左墙')
    fill(GX, 4.0 + h, 0.8 - h, 0.8 + h, '黑箱子前墙')
    fill(4.0 - h, 4.0 + h, -0.8, 0.8, '前方封死')
    # ---- 柜子：0.4m 见方，中心 (1,1) ----
    fill(0.8, 1.2, 0.8, 1.2, '柜子(1,1)')

    grid = OccupancyGrid(occ, (x0, y0), res)
    np.savez(a.out, occ=occ.astype(np.uint8),
             origin=np.array([x0, y0]), res=np.array(res),
             scene_origin=np.array([0.0, 0.0]))
    gap = GX - 1.2
    print()
    print('  栅格 %d x %d @ %.2fm  障碍格 %d (%.2f%%)'
          % (H, W, res, occ.sum(), 100.0 * occ.sum() / occ.size))
    print('  ★ 柜子右边(x=1.2) 到 黑箱墙(x=%.2f) 的缝 = %.2f m' % (GX, gap))
    print('    ⇒ 膨胀 0.35 后自由通道 = %.2f m' % (gap - 0.70))
    print('  → %s' % a.out)

    # ---- 规划验证 ----
    try:
        from planning.global_planner import plan_path
        from core.types import Pose2D, FRAME_SCENE
        s = Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE)
        t = Pose2D(0.5, 2.0, 0.0, frame=FRAME_SCENE)
        p = plan_path(grid, s, t, inflate_radius=0.35)
        print()
        print('  === 规划验证 (0,0) → (0.5,2.0)，膨胀 0.35 ===')
        print('  ok=%s  路点 %d  路长 %.3f m' % (p.ok, len(p.waypoints), p.length))
        if p.ok:
            pts = [(round(w.x, 2), round(w.y, 2)) for w in p.waypoints]
            print('  路点:', pts)
            gi = grid.inflated(0.35)
            bad = [w for w in p.waypoints if not gi.is_free_xy(w.x, w.y)]
            print('  ★ 落在膨胀障碍里的路点: %d' % len(bad))
            # 穿缝的那段：x 在 [1.2, 2] 之间的点
            seg = [w for w in p.waypoints if 1.2 <= w.x <= GX]
            if seg:
                ys = [w.y for w in seg]
                print('  ★ 穿缝段: %d 个点, y 从 %.3f 到 %.3f（缝在 y≈0.8~1.2）'
                      % (len(seg), min(ys), max(ys)))
    except Exception as e:
        print('  （规划验证失败: %s）' % e)

    # ---- 出图 ----
    png = a.png or (os.path.splitext(a.out)[0] + '.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[x0, x0 + W * res, y0, y0 + H * res])
        # 膨胀后的边界（虚线）
        gi = grid.inflated(0.35)
        ax.contour(np.linspace(x0, x1, W), np.linspace(y0, y1, H),
                   gi.occ.astype(float), levels=[0.5], colors='orange',
                   linewidths=1.0, linestyles='--')
        ax.plot(0, 0, 'o', color='lime', ms=13, mec='k', label='dog start')
        ax.arrow(0, 0, 0.5, 0, color='lime', width=0.02, head_width=0.1,
                 length_includes_head=True)
        ax.plot(0.5, 2.0, '*', color='red', ms=18, label='goal (0.5,2.0)')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('x [m] (forward)'); ax.set_ylabel('y [m] (left)')
        ax.set_title('lab scene: narrow gap %.2fm at x in [1.2, %.1f], y~1.0'
                     % (GX - 1.2, GX))
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(png, dpi=100)
        print('  → 预览图 %s' % png)
    except Exception as e:
        print('  （出图失败: %s）' % e)


if __name__ == '__main__':
    main()
