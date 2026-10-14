# -*- coding: utf-8 -*-
"""从大门沿 y=0.8 往桌子走，狗中心【最远】能停到哪？

对比三件事：
   ① 他们的规则：桌前沿(2.4) 外退 0.5m  ⇒ x = 1.9
   ② 我们的物理约束：车体半径 0.35m ⇒ 离【所有】障碍 ≥0.35m
   ③ 办公椅的实际位置：x[1.90, 2.50] y[0.70, 1.30]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maps.grid import OccupancyGrid                       # noqa: E402

for tag, path in [('【带办公椅】', '/tmp/a5007_new.npz'),
                  ('【不带办公椅】', '/tmp/a5007_nochair.npz')]:
    d = np.load(path)
    g = OccupancyGrid(d['occ'].astype(bool),
                      tuple(float(x) for x in d['origin']), float(d['res']))
    gi = g.inflated(0.35)
    print('  %s  （0.35 膨胀后）' % tag)
    ok = []
    for x in np.arange(1.0, 2.45, 0.05):
        r, c = gi.world_to_cell(x, 0.8)
        if gi.in_bounds(r, c) and not gi.occ[r, c]:
            ok.append(x)
    if ok:
        xmax = max(ok)
        print('    狗中心可以停到最远 x = %.2f' % xmax)
        print('    ⇒ 离桌子前沿 (2.40) 有 %.2f m   （我们的车体半径 = 0.35）'
              % (2.40 - xmax))
        print('    ⇒ 离可乐罐 (x=2.96) 有 %.2f m' % (2.96 - xmax))
    else:
        print('    ★ 一个都站不了')

    # ★ 判断 1.9 能不能站（用栅格查，别用浮点 in）
    r, c = gi.world_to_cell(1.9, 0.8)
    can19 = gi.in_bounds(r, c) and not gi.occ[r, c]
    print('    他们的停靠点 x=1.90  →  %s'
          % ('✅ 能站' if can19 else '✗ ★站不了（在障碍的膨胀区里）'))
    print()

print('  ★★ 关键数字：车体半径 0.35m ⇒ 狗【中心】必须离障碍 ≥0.35m')
print('     ⇒ 狗中心离桌子前沿最近只能到 2.40-0.35 = 2.05（栅格化后 2.00）')
print('     ⇒ 而可乐罐在 x=2.96 ⇒ 狗中心到罐子的【物理最近距离】≈ 0.96m')
print()
print('  ⇒ 所以"距可乐罐 0.5m"这个说法，只有在【从狗头/雷达看】时才成立，')
print('     从【狗中心】算的话，极限就是 ~0.96m。这个口径要说清楚。')

