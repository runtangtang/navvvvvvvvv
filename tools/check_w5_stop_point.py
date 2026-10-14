# -*- coding: utf-8 -*-
"""检查：SOP 期望的停靠点 (1.9, 0.8)，在【膨胀后】的栅格里能不能站？

★ 为什么会不对：
    停靠点算法是"桌子前沿 x=2.40 外退 0.5m ⇒ x=1.90"
    但这个算法【没看中间还隔着什么】——
    办公椅 office_chair x[1.90, 2.50] y[0.70, 1.30] 正好在那儿！
    ⇒ (1.9, 0.8) 落在办公椅的【左边界的正上方】
    ⇒ 膨胀 0.35 之后，要 x<1.55 或 y<0.35 或 y>1.65 才真自由
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maps.grid import OccupancyGrid                        # noqa: E402
from planning.global_planner import nearest_free           # noqa: E402

d = np.load('/tmp/a5007_new.npz')
occ = d['occ'].astype(bool)
origin = tuple(float(x) for x in d['origin'])
res = float(d['res'])
g = OccupancyGrid(occ, origin, res)
INFLATE = 0.35

gi = g.inflated(INFLATE)
free = gi.free_mask()

print('  规划膨胀半径 inflate_radius = %.2f m' % INFLATE)
print()
print('  点           在原始障碍里?   在【膨胀后】障碍里?   会被挪到哪')
print('  ' + '-' * 68)

for name, (x, y) in [('SOP 停靠点 W5', (1.9, 0.8)),
                     ('桌前沿外退0.6', (1.8, 0.8)),
                     ('再退到椅子左', (1.5, 0.8)),
                     ('从下方绕', (1.9, 0.3)),
                     ('W3 右侧绕行点', (2.2, -0.4)),
                     ('W4 左侧备用', (1.5, 1.0))]:
    r0, c0 = g.world_to_cell(x, y)
    in_raw = (not g.in_bounds(r0, c0)) or occ[r0, c0]
    r1, c1 = gi.world_to_cell(x, y)
    in_inf = (not gi.in_bounds(r1, c1)) or gi.occ[r1, c1]
    nf = nearest_free(free, (r1, c1))
    if nf is None:
        tgt = '★ 找不到'
    else:
        import math
        # ★ cell_to_world 返回的是 (x, y) —— 别写反（我第一版就写反了）
        wx, wy = gi.cell_to_world(nf[0], nf[1])
        tgt = '(%.2f, %.2f)  挪了 %.2f m' % (wx, wy, math.hypot(wx - x, wy - y))
    print('  %-14s %-13s %-19s %s'
          % (name, '★是' if in_raw else '否', '★是' if in_inf else '否', tgt))

print()
print('  ⇒ ★ SOP 的停靠点 (1.9, 0.8) 和办公椅重叠（膨胀后更是差得远），')
print('     实际会被挪到椅子左边 ≈ (1.5, 0.8) —— 离可乐罐约 %.2f m，'
      % ((2.96 - 1.5) / 1))
print('     而不是 SOP 想要的 0.5m。')
