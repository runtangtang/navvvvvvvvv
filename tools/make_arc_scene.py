#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_arc_scene.py —— 弧形走廊场景（测弯道跟随）

用户 2026-09-26 现场指定（在他手绘图的基础上标准化）：
    · 外墙和"窄口"那个场景几乎一样：
        狗后面 x=-0.6 ／ 狗右边 y=-0.8 ／ 前方封死 x=4 ／
        黑箱子左墙 x=2 (y 0.8~3) ／ 黑箱子前墙 y=0.8 (x 2~4)
    · 左边墙【变短】：(-0.6, 1) → (0.5, 1)
    · 柜子(1,1)【去掉】
    · ★ 新增两道【同心圆弧墙】，圆心 (0.5, 1)：
        内圈 r = r_in      外圈 r = r_out
      ⇒ 狗只能从【两弧之间的环形走廊】走
    · 目标还是 (0.5, 2.0)

★ 走廊宽度 = r_out - r_in。膨胀 0.35 后自由通道 = 宽度 - 0.70
   ⇒ 0.9m 的环廊只剩 0.2m 自由通道，【而且是弯的】⇒ 压力测试

用法:
    python3 tools/make_arc_scene.py                       # 默认 r_in=0.6 r_out=1.5 左半圆
    python3 tools/make_arc_scene.py --r-in 0.7 --r-out 1.6
    python3 tools/make_arc_scene.py --a0 90 --a1 270      # 弧的角度范围 [度]
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
ap.add_argument('--wx', type=float, default=0.5, help='圆心 x')
ap.add_argument('--wy', type=float, default=1.0, help='圆心 y')
ap.add_argument('--r-in', type=float, default=0.6)
ap.add_argument('--r-out', type=float, default=1.5)
ap.add_argument('--a0', type=float, default=90.0, help='弧起始角 [度]')
ap.add_argument('--a1', type=float, default=270.0, help='弧结束角 [度]')
ap.add_argument('--wall-thick', type=float, default=0.06)
ap.add_argument('--gap-x', type=float, default=2.0)
ap.add_argument('--bounds', nargs=4, type=float,
                default=[-1.5, 5.0, -1.8, 3.5])
ap.add_argument('--out', default='tools/arc_scene.npz')
ap.add_argument('--png', default=None)
a = ap.parse_args()

WT = a.wall_thick


def main():
    x0, x1, y0, y1 = a.bounds
    res = a.res
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))
    occ = np.zeros((H, W), dtype=bool)

    def fill(xa, xb, ya, yb, what=None):
        ca = max(0, int(math.floor((xa - x0) / res)))
        cb = min(W, int(math.ceil((xb - x0) / res)))
        ra = max(0, int(math.floor((ya - y0) / res)))
        rb = min(H, int(math.ceil((yb - y0) / res)))
        occ[ra:rb, ca:cb] = True
        if what:
            print('  %-14s x[%.2f,%.2f] y[%.2f,%.2f]' % (what, xa, xb, ya, yb))

    def arc(radius, what):
        """按弧长步进画一段环带（半径 ± 半墙厚）"""
        n = max(60, int(radius * abs(math.radians(a.a1 - a.a0)) / (res * 0.7)))
        cnt = 0
        for i in range(n + 1):
            th = math.radians(a.a0 + (a.a1 - a.a0) * i / n)
            cx = a.wx + radius * math.cos(th)
            cy = a.wy + radius * math.sin(th)
            fill(cx - WT / 2, cx + WT / 2, cy - WT / 2, cy + WT / 2)
            cnt += 1
        print('  %-14s 圆心(%.2f,%.2f) r=%.2f  角度 %.0f°~%.0f°  (%d 段)'
              % (what, a.wx, a.wy, radius, a.a0, a.a1, cnt))

    h = WT / 2
    print('外墙（和窄口场景一致）:')
    fill(-0.6 - h, -0.6 + h, y0, y1, '狗后面')
    fill(x0, x1, -0.8 - h, -0.8 + h, '狗右边')
    fill(a.gap_x - h, a.gap_x + h, 0.8, 3.0, '黑箱左墙')
    fill(a.gap_x, 4.0 + h, 0.8 - h, 0.8 + h, '黑箱前墙')
    fill(4.0 - h, 4.0 + h, -0.8, 0.8, '前方封死')
    fill(-0.6, 0.5 + h, 1.0 - h, 1.0 + h, '左边短墙')
    print('★ 圆弧走廊:')
    arc(a.r_in, '内圈')
    arc(a.r_out, '外圈')

    grid = OccupancyGrid(occ, (x0, y0), res)
    np.savez(a.out, occ=occ.astype(np.uint8),
             origin=np.array([x0, y0]), res=np.array(res),
             scene_origin=np.array([0.0, 0.0]))

    corr = a.r_out - a.r_in
    print()
    print('  栅格 %d x %d @ %.2fm  障碍格 %d (%.2f%%)'
          % (H, W, res, occ.sum(), 100.0 * occ.sum() / occ.size))
    print('  ★ 环廊宽 %.2f m  ⇒ 膨胀 0.35 后自由通道 %.2f m' % (corr, corr - 0.70))
    print('  → %s' % a.out)

    # 规划验证
    try:
        from planning.global_planner import plan_path
        from core.types import Pose2D, FRAME_SCENE
        s = Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE)
        t = Pose2D(0.5, 2.0, 0.0, frame=FRAME_SCENE)
        p = plan_path(grid, s, t, inflate_radius=0.35)
        print()
        print('  === 规划 (0,0) → (0.5,2.0)，膨胀 0.35 ===')
        print('  ok=%s  路点 %d  路长 %.3f m（直线 %.2f m）'
              % (p.ok, len(p.waypoints), p.length, math.hypot(0.5, 2.0)))
        if p.ok:
            gi = grid.inflated(0.35)
            bad = [w for w in p.waypoints if not gi.is_free_xy(w.x, w.y)]
            print('  ★ 落在膨胀障碍里的路点: %d' % len(bad))
            rr = [math.hypot(w.x - a.wx, w.y - a.wy) for w in p.waypoints]
            print('  ★ 路点到圆心的距离: %.2f ~ %.2f（环廊 %.2f~%.2f）'
                  % (min(rr), max(rr), a.r_in, a.r_out))
            if p.ok:
                print('  路点:', [(round(w.x, 2), round(w.y, 2))
                                  for w in p.waypoints][:16])
    except Exception as e:
        print('  （规划验证失败: %s）' % e)

    png = a.png or (os.path.splitext(a.out)[0] + '.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[x0, x0 + W * res, y0, y0 + H * res])
        gi = grid.inflated(0.35)
        ax.contour(np.linspace(x0, x1, W), np.linspace(y0, y1, H),
                   gi.occ.astype(float), levels=[0.5], colors='orange',
                   linewidths=0.9, linestyles='--')
        if 'p' in dir() and p.ok:
            q = np.array([[w.x, w.y] for w in p.waypoints])
            ax.plot(q[:, 0], q[:, 1], color='tab:red', lw=2, ls='--',
                    label='planned')
        ax.plot(0, 0, 'o', color='lime', ms=13, mec='k', label='dog start')
        ax.plot(0.5, 2.0, '*', color='red', ms=18, label='goal (0.5,2.0)')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('x [m] (forward)'); ax.set_ylabel('y [m] (left)')
        ax.set_title('ARC corridor: r_in=%.2f  r_out=%.2f  (width %.2fm)'
                     % (a.r_in, a.r_out, corr))
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(png, dpi=100)
        print('  → 预览图 %s' % png)
    except Exception as e:
        print('  （出图失败: %s）' % e)


if __name__ == '__main__':
    main()
