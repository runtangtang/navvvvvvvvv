# -*- coding: utf-8 -*-
"""★ 独立交叉验证：_SceneObstacleSource 的坐标换算，和 Transform2D 语义一致吗？

为什么要单独验：
    _SceneObstacleSource 是自己手写的 R(yaw)@p + t，
    而 _ScenePoseSource 用的是 Transform2D 的 __matmul__。
    两者【必须给出同一个结果】—— 否则障碍点和位姿就在两个不同的系里，
    避障会在一个"歪掉的"世界模型上工作，而且【不报错】。

做法：随机生成一堆 T 和点，把 (x,y) 分别走两条路，逐点比。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose2D, Transform2D, FRAME_CAPO, FRAME_SCENE  # noqa: E402
from mission.api import _SceneObstacleSource                        # noqa: E402


class _Wrap:
    """把 (N,3) 数组包成 _SceneObstacleSource 要的形状"""
    name = 'fake-capo'

    def __init__(self, arr):
        self._a = arr

    def get(self):
        return self._a

    def healthy(self, max_age=1.0):
        return True

    def error_count(self):
        return 0


rng = np.random.default_rng(0)
worst = 0.0
n_case = 200
for k in range(n_case):
    yaw = rng.uniform(-np.pi, np.pi)
    T = Transform2D(from_frame=FRAME_CAPO, to_frame=FRAME_SCENE,
                    x=rng.uniform(-5, 5), y=rng.uniform(-5, 5), yaw=yaw)
    pts = np.stack([rng.uniform(-10, 10, 50),
                    rng.uniform(-10, 10, 50),
                    rng.uniform(0, 2, 50)], axis=1).astype(np.float32)

    got = _SceneObstacleSource(_Wrap(pts), T).get()

    # 走另一条路：用 Transform2D 自己的 __matmul__（位姿源用的就是它）
    ref = np.array([[(T @ Pose2D(float(x), float(y), 0.0, frame=FRAME_CAPO)).x,
                     (T @ Pose2D(float(x), float(y), 0.0, frame=FRAME_CAPO)).y,
                     float(z)]
                    for x, y, z in pts], dtype=np.float64)

    d = float(np.max(np.abs(got[:, :2] - ref[:, :2])))
    worst = max(worst, d)

print('  跑了 %d 组随机 T，每组 50 个点' % n_case)
print('  ★ 两条路的最大偏差: %.3e m' % worst)
print()
if worst < 1e-4:
    print('  ✅ 一致 —— 障碍点和位姿在同一个坐标系里，避障的世界模型是对的')
else:
    print('  ★★ 不一致！障碍点和位姿不在同一个系 —— 避障会在歪掉的模型上工作')
    raise SystemExit(1)

# 顺带把规格里那个手算例子也验一遍
T = Transform2D(from_frame=FRAME_CAPO, to_frame=FRAME_SCENE,
                x=0.5, y=0.5, yaw=np.pi / 2)
one = _SceneObstacleSource(_Wrap(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)), T).get()
print('  手算例子: CAPO(1,0) + t=(0.5,0.5) rot=90° ⇒ scene (%.2f, %.2f)'
      % (one[0, 0], one[0, 1]))
assert abs(one[0, 0] - 0.5) < 1e-5 and abs(one[0, 1] - 1.5) < 1e-5, '手算例子对不上'
print('  ✅ 和手算一致 (0.50, 1.50)')
