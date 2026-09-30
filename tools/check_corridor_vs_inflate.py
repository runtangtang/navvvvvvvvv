# -*- coding: utf-8 -*-
"""验证：corridor_m 和 inflate_radius 的关系会不会导致【静态障碍误触发】

背景：
    规划时按 inflate_radius=0.35 膨胀 ⇒ 路径上任意点离最近【静态】障碍 ≥0.35m
    （而且路径常常【贴着】这个边界走 —— 见 FollowerConfig 里 inflate_radius 的注释）
    如果 corridor_m ≥ 0.35 ⇒ 那些静态障碍点会被判"在走廊里" ⇒ 一出发就误停

做法：造一条直线路径，把"墙"上的点放在离路径不同距离处，看会不会触发停车。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose2D, FRAME_SCENE                    # noqa: E402
from planning.local_avoid import (AvoidConfig, corridor_polyline,   # noqa: E402
                                  should_stop)

pose = Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE)
goal = Pose2D(2.0, 0.0, 0.0, frame=FRAME_SCENE)
wps = [Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE),
       Pose2D(0.5, 0.0, 0.0, frame=FRAME_SCENE),
       Pose2D(1.0, 0.0, 0.0, frame=FRAME_SCENE),
       Pose2D(1.5, 0.0, 0.0, frame=FRAME_SCENE),
       goal]

# 一堵"墙"：沿 y = d 铺 20 个点（模拟点云里墙面产生的密集点）
def wall(d, x0=0.3, x1=1.8, n=20):
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, np.full(n, d), np.zeros(n)], axis=1)


cfg_common = dict(enabled=True, lookahead_m=2.0, min_points=2, hold_s=1.0)

print('  规划膨胀 inflate_radius = 0.35 m')
print('  ⇒ 路径上任意点离最近【静态】障碍 ≥ 0.35 m（而且常常贴着这个边界走）')
print()
print('  墙离路径    corridor_m=0.40          corridor_m=0.25')
print('  距离 [m]    停?  命中点数            停?  命中点数')
print('  ' + '-' * 62)

polyline = corridor_polyline(wps, pose, 1, goal, 2.0)
for d in [0.20, 0.30, 0.34, 0.35, 0.36, 0.40, 0.45, 0.60, 1.00]:
    pts = wall(d)
    row = '  %5.2f      ' % d
    for cm in [0.40, 0.25]:
        cfg = AvoidConfig(corridor_m=cm, **cfg_common)
        stop, cnt = should_stop(pts, polyline, cfg)
        row += '  %-4s %4d            ' % ('★停' if stop else '不停', cnt)
    print(row)

print()
print('  ★ 结论：corridor_m=0.40 > 0.35 ⇒ 贴着路径的静态障碍会被误判；')
print('          改成 0.25 之后就安全了（只对"比规划时更近的东西"报警）。')
