#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_grid_from_scene.py —— 【带语义的场景图 JSON】 → 【占据栅格 npz】

为什么要有这个（2026-09-28）：
    组1/组2 给的是【语义 3D 场景图】（物体 + 包围盒），
    我们的规划层吃的是【2D 占据栅格】。
    ⇒ 中间这一层转换以前靠手写，现在固化成工具。

★ 坐标系：直接用场景图自己的坐标系（原点=他们定义的，x 轴=他们定义的）
   —— 我们的 s_* 对齐本来就是"场景系"，所以【不需要换算】。

★ 哪些物体该当障碍物（★ 这是最容易搞错的，务必逐条看）：
    ✅ 桌子 / 椅子 / 垃圾桶 / 充电桩 …… 地板上的实体 → 当障碍
    ❌ 【门】：包围盒常常正好套在"狗站的原点"上 ⇒ 当障碍的话【起点就在障碍里】，
              规划直接 no_path。门是通道不是墙。
    ❌ 【桌上的东西】（可乐罐之类）：z 是 0.8~0.9，狗从下面钻过去，
              对 2D 栅格来说它不是地板障碍 ⇒ 不算
    ❌ 【我们要去停靠的目标物】：算了它就到不了
   ⇒ 所以本工具【默认跳过】门/罐子这类；要改就用 --skip / --only 明确指定

用法:
    python3 tools/make_grid_from_scene.py a5007_lab_scene.json --out tools/a5007_scene.npz
    python3 tools/make_grid_from_scene.py xxx.json --skip door,coke_can --res 0.05
    python3 tools/make_grid_from_scene.py xxx.json --only main_desk,office_chair
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maps.grid import OccupancyGrid      # noqa: E402

# ★ 默认跳过的：门（在狗站位上）、桌上的小物体（狗从下面过）
DEFAULT_SKIP = ['door', 'coke_can', 'coke can', 'coke', 'cola']

ap = argparse.ArgumentParser()
ap.add_argument('scene', help='语义场景图 JSON')
ap.add_argument('--out', default=None, help='输出 npz（默认 <scene>_grid.npz）')
ap.add_argument('--res', type=float, default=0.05, help='分辨率 [m]，默认 0.05')
ap.add_argument('--pad', type=float, default=1.0, help='房间边界外再留多少 [m]')
ap.add_argument('--skip', default=None,
                help='不当作障碍的 object_tag，逗号分隔。默认: %s' % ','.join(DEFAULT_SKIP))
ap.add_argument('--only', default=None,
                help='★ 只把哪些 tag 当障碍（给了就忽略 --skip）')
ap.add_argument('--png', default=None, help='预览图')
a = ap.parse_args()


def main():
    with open(a.scene, 'r', encoding='utf-8') as f:
        d = json.load(f)

    rb = d.get('room_bounds') or {}
    x0 = float(rb.get('x_min', -1.0)) - a.pad
    x1 = float(rb.get('x_max', 5.0)) + a.pad
    y0 = float(rb.get('y_min', -2.0)) - a.pad
    y1 = float(rb.get('y_max', 2.0)) + a.pad
    res = a.res
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))
    occ = np.zeros((H, W), dtype=bool)

    only = [s.strip().lower() for s in a.only.split(',')] if a.only else None
    skip = [s.strip().lower() for s in
            (a.skip.split(',') if a.skip else DEFAULT_SKIP)]

    print('  房间 %s   边界 x[%.1f,%.1f] y[%.1f,%.1f]   栅格 %dx%d @ %.2fm'
          % (d.get('room_name', '?'), x0, x1, y0, y1, H, W, res))
    print()
    print('  id  tag                  2D足迹(x,y)              z范围      处理')
    n_obs = 0
    for o in d.get('objects', []):
        tag = str(o.get('object_tag', '')).lower()
        c = o.get('bbox_center')
        e = o.get('bbox_extent')
        if not c or not e:
            continue
        cx, cy, cz = c
        ex, ey, ez = e
        used = (tag in only) if only is not None else (tag not in skip)
        zr = 'z[%.2f,%.2f]' % (cz - ez / 2, cz + ez / 2)
        foot = 'x[%.2f,%.2f] y[%.2f,%.2f]' % (cx - ex / 2, cx + ex / 2,
                                              cy - ey / 2, cy + ey / 2)
        if not used:
            why = '只清单' if only is not None else '默认跳过'
            print('  %-3s %-18s %-24s %-11s ❌ 不算（%s）'
                  % (o.get('id'), tag, foot, zr, why))
            continue

        # 铺进栅格
        c_lo = int(math.floor((cx - ex / 2 - x0) / res))
        c_hi = int(math.ceil((cx + ex / 2 - x0) / res))
        r_lo = int(math.floor((cy - ey / 2 - y0) / res))
        r_hi = int(math.ceil((cy + ey / 2 - y0) / res))
        c_lo = max(0, c_lo); c_hi = min(W, c_hi)
        r_lo = max(0, r_lo); r_hi = min(H, r_hi)
        occ[r_lo:r_hi, c_lo:c_hi] = True
        n_obs += 1
        print('  %-3s %-18s %-24s %-11s ✅ 当障碍（%d 格）'
              % (o.get('id'), tag, foot, zr,
                 (r_hi - r_lo) * (c_hi - c_lo)))

    grid = OccupancyGrid(occ, (x0, y0), res)
    out = a.out or (os.path.splitext(a.scene)[0] + '_grid.npz')
    np.savez(out, occ=occ.astype(np.uint8),
             origin=np.array([x0, y0]), res=np.array(res),
             scene_origin=np.array([0.0, 0.0]))

    print()
    print('  ⇒ %s' % out)
    print('     障碍占 %.2f%%   空格子 %d' % (100.0 * occ.sum() / occ.size,
                                              int((~occ).sum())))

    # ★ 把安全走廊路标也顺手检查一下 —— 它们在障碍里就说明图有问题
    wps = d.get('safe_corridor_waypoints') or []
    if wps:
        print()
        print('  安全走廊路标检查（在我们的栅格里通不通）：')
        for w in wps:
            p = w.get('pos') or []
            if len(p) < 2:
                continue
            r, c = grid.world_to_cell(p[0], p[1])
            ok = grid.in_bounds(r, c) and not occ[r, c]
            print('    %-3s %-18s (%.1f, %.1f)  %s'
                  % (w.get('id'), w.get('name', ''), p[0], p[1],
                     '✅ 空' if ok else '★★ 落在障碍里！'))

    png = a.png or (os.path.splitext(out)[0] + '.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[x0, x0 + W * res, y0, y0 + H * res])
        for o in d.get('objects', []):
            c = o.get('bbox_center')
            if not c:
                continue
            ax.plot(c[0], c[1], '.', ms=4, color='tab:blue')
            ax.annotate('%s' % o.get('object_tag', '')[:10], (c[0], c[1]),
                        fontsize=6, color='tab:blue')
        for w in wps:
            p = w.get('pos') or []
            if len(p) >= 2:
                ax.plot(p[0], p[1], 'o', ms=6, mfc='none', mec='tab:red')
                ax.annotate(w.get('id', ''), (p[0], p[1]), fontsize=7,
                            color='tab:red')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('scene x [m]'); ax.set_ylabel('scene y [m]')
        ax.set_title('%s  occupancy grid  (scene frame)'
                     % d.get('room_name', ''))
        plt.tight_layout(); plt.savefig(png, dpi=100)
        print('  ⇒ 预览图 %s' % png)
    except Exception as e:
        print('  （预览图失败: %s）' % e)


if __name__ == '__main__':
    main()
