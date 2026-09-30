#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/build_grid_from_lio.py —— 从 Point-LIO 的世界系点云建【2D 占据栅格】

为什么要有"去倾斜"这一步
------------------------
2026-09-26 实测：狗在实验室走 2.5 分钟建的图，
    拟合地面平面 z = 0.0114x + 0.0398y - 0.5276
    倾斜 绕y 0.66° / 绕x 2.28°  ⇒ 在 6×8m 范围里地面"高度"变了 0.39 m
    去倾斜后残差 std = 0.068 m

★ 我们做栅格要靠"离地高度"挑障碍点。不平的地面会让固定高度带
  在某些地方切到地面（假障碍）、某些地方漏掉真障碍。
⇒ 所以必须先【减掉拟合的地面平面】，再按离地高度挑点。

★ 它还顺带解决"狗站在略微倾斜的地面上"的问题。

坐标系
------
输出栅格的坐标系 = Point-LIO 的世界系（原点 = 开建图时狗的位置）。
★ 和 nav_node_real 的 --start 0 0 0（T=单位阵）一致：
  只要【把狗摆回建图时的位置和朝向】，图就能直接用。

用法:
    python3 tools/build_grid_from_lio.py <点云npz> [--out tools/lab_scene.npz]
        [--h-lo 0.08] [--h-hi 0.60]   # 离地高度带 [m]
        [--res 0.05] [--min-pts 3]    # 格子分辨率 / 判障碍的最少点数
"""
import argparse
import math
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('--out', default=None)
ap.add_argument('--res', type=float, default=0.05)
ap.add_argument('--h-lo', type=float, default=0.08,
                help='障碍高度带下限 [m]（离地）')
ap.add_argument('--h-hi', type=float, default=0.60,
                help='障碍高度带上限 [m]')
ap.add_argument('--min-pts', type=int, default=3,
                help='一个格子里至少几个点才算障碍')
ap.add_argument('--pad', type=float, default=0.5)
ap.add_argument('--no-detilt', action='store_true')
a = ap.parse_args()


def main():
    d = np.load(a.npz)
    p = d['pts']
    print('=' * 62)
    print('  建栅格: %s' % a.npz)
    print('=' * 62)
    print('  点数 %d' % len(p))

    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    # ---- ① 找地面：取 z 的众数附近做平面拟合 ----
    hist, edges = np.histogram(z, bins=60)
    k = int(np.argmax(hist))
    zc = 0.5 * (edges[k] + edges[k + 1])
    band = 0.3
    m = np.abs(z - zc) < band
    print('  z 众数档中心 %.3f  ⇒ 取 ±%.1f m 内的 %d 点拟合地面'
          % (zc, band, m.sum()))

    if a.no_detilt or m.sum() < 100:
        coef = np.array([0.0, 0.0, zc])
    else:
        A = np.column_stack([x[m], y[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    aa, bb, cc = coef
    if not a.no_detilt:
        print('  地面平面 z = %.4fx + %.4fy + %.4f' % (aa, bb, cc))
        print('  倾斜: 绕y %.2f°  绕x %.2f°' %
              (math.degrees(math.atan(aa)), math.degrees(math.atan(bb))))

    # ---- ② 去倾斜：z' = 离地高度 ----
    h = z - (aa * x + bb * y + cc)
    print('  去倾斜后 离地高度: %.2f ~ %.2f m' % (h.min(), h.max()))

    # ---- ③ 挑障碍点 → 2D 投影 ----
    mob = (h > a.h_lo) & (h < a.h_hi)
    q = p[mob]
    print('  障碍带 [%.2f, %.2f] m: %d 点（%.1f%%）'
          % (a.h_lo, a.h_hi, len(q), 100.0 * len(q) / len(p)))

    x0, x1 = x.min() - a.pad, x.max() + a.pad
    y0, y1 = y.min() - a.pad, y.max() + a.pad
    res = a.res
    W = int(math.ceil((x1 - x0) / res))
    H = int(math.ceil((y1 - y0) / res))
    cnt = np.zeros((H, W), dtype=np.int32)
    if len(q):
        col = np.clip(((q[:, 0] - x0) / res).astype(np.int32), 0, W - 1)
        row = np.clip(((q[:, 1] - y0) / res).astype(np.int32), 0, H - 1)
        np.add.at(cnt, (row, col), 1)

    occ = cnt >= a.min_pts
    print('  栅格 %d x %d @ %.2fm  origin (%.2f, %.2f)' % (H, W, res, x0, y0))
    print('  障碍格 %d （占 %.2f%%）' % (occ.sum(), 100.0 * occ.sum() / occ.size))

    out = a.out or (os.path.splitext(a.npz)[0] + '_grid.npz')
    np.savez(out, occ=occ.astype(np.uint8),
             origin=np.array([x0, y0]), res=np.array(res),
             scene_origin=np.array([0.0, 0.0]))   # ★ P-LIO 原点 = 狗起点
    print('  → %s' % out)

    # ---- ④ 出图 ----
    png = os.path.splitext(out)[0] + '.png'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(17, 8))
        ax = axs[0]
        sc = ax.scatter(p[::3, 0], p[::3, 1], s=0.3, c=h[::3], cmap='turbo',
                        vmin=-0.2, vmax=0.6, alpha=0.4)
        plt.colorbar(sc, ax=ax, label='height above floor [m]')
        ax.set_title('%s  (%d pts, tilt-corrected)' % (os.path.basename(a.npz), len(p)))
        ax = axs[1]
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[x0, x0 + W * res, y0, y0 + H * res])
        ax.plot(0, 0, 'o', color='lime', ms=13, mec='k', label='P-LIO origin')
        ax.set_title('occupancy grid: %d obstacle cells (%.2f%%)'
                     % (occ.sum(), 100.0 * occ.sum() / occ.size))
        for ax in axs:
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.set_xlabel('x [m]'); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(png, dpi=90)
        print('  → 预览图 %s' % png)
    except Exception as e:
        print('  （出图失败: %s）' % e)
    print('=' * 62)


if __name__ == '__main__':
    main()
