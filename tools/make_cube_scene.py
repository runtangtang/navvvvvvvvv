#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_cube_scene.py —— 生成【狗相对坐标】的带障碍场景栅格

★ 坐标系（很重要）
    原点 = 狗的起点，x 轴 = 狗起跑时的朝向，y 轴 = 左
    ⇒ 和 nav_node_real 的 --start 0 0 0（T=单位阵）完全一致
    ⇒ 所以【同一张图可以重复用】，不管狗摆在哪、朝哪

用法:
    python3 tools/make_cube_scene.py                    # 默认：正前方 1m 放 0.6m 正方体
    python3 tools/make_cube_scene.py --cube 1.0 0.0 0.6 # x y 棱长
    python3 tools/make_cube_scene.py --none             # 不放障碍（对照）
    python3 tools/make_cube_scene.py --out xxx.npz --png xxx.png
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maps.grid import OccupancyGrid      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cube', nargs=3, type=float, default=[1.0, 0.0, 0.6],
                metavar=('X', 'Y', 'SIDE'),
                help='正方体中心 + 棱长 [m]（狗相对坐标；x 前、y 左）')
ap.add_argument('--none', action='store_true', help='不放障碍（对照组）')
ap.add_argument('--res', type=float, default=0.05)
ap.add_argument('--bounds', nargs=4, type=float,
                default=[-2.0, 8.0, -5.0, 5.0],
                metavar=('X0', 'X1', 'Y0', 'Y1'))
ap.add_argument('--out', default=None)
ap.add_argument('--png', default=None)
a = ap.parse_args()


def main():
    x0, x1, y0, y1 = a.bounds
    res = a.res
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))
    occ = np.zeros((H, W), dtype=bool)

    if not a.none:
        cx, cy, side = a.cube
        h = side / 2.0
        # 世界坐标 → 格子：col=(x-origin_x)/res, row=(y-origin_y)/res
        c_lo = int(math.floor((cx - h - x0) / res))
        c_hi = int(math.ceil((cx + h - x0) / res))
        r_lo = int(math.floor((cy - h - y0) / res))
        r_hi = int(math.ceil((cy + h - y0) / res))
        c_lo = max(0, c_lo); c_hi = min(W, c_hi)
        r_lo = max(0, r_lo); r_hi = min(H, r_hi)
        occ[r_lo:r_hi, c_lo:c_hi] = True
        print('  正方体: 中心 (%.2f, %.2f) 棱长 %.2f  ⇒ 占格 [%d:%d, %d:%d] = %d 个'
              % (cx, cy, side, r_lo, r_hi, c_lo, c_hi, (r_hi - r_lo) * (c_hi - c_lo)))
    else:
        print('  不放障碍（对照组）')

    grid = OccupancyGrid(occ, (x0, y0), res)
    out = a.out or ('tools/cube_scene.npz' if not a.none
                    else 'tools/empty_grid.npz')

    # ★ scene_origin=(0,0)：因为这张图【本身就是狗相对坐标】，
    #   而 live_view 会把图的 origin 减去 scene_origin ⇒ 正好对得上
    np.savez(out, occ=occ.astype(np.uint8),
             origin=np.array([x0, y0]), res=np.array(res),
             scene_origin=np.array([0.0, 0.0]))
    free = int((~occ).sum())
    print('  栅格: %d x %d @ %.2fm  origin (%.1f, %.1f)  障碍占 %.2f%%'
          % (H, W, res, x0, y0, 100.0 * occ.sum() / occ.size))
    print('  → %s    （空格子 %d）' % (out, free))

    # 预览图
    png = a.png or (os.path.splitext(out)[0] + '.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[x0, x0 + W * res, y0, y0 + H * res])
        ax.plot(0, 0, 'o', color='lime', ms=12, mec='k', label='dog start')
        ax.arrow(0, 0, 0.6, 0, color='lime', width=0.02,
                 head_width=0.12, length_includes_head=True)
        ax.plot(2.0, 0.0, '*', color='red', ms=16, label='goal (2,0)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('x [m] (forward)')
        ax.set_ylabel('y [m] (left)')
        ax.set_title('scene (dog-relative): origin = dog, +x = heading')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(png, dpi=100)
        print('  → 预览图 %s' % png)
    except Exception as e:
        print('  （预览图失败: %s）' % e)


if __name__ == '__main__':
    main()
