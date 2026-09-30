# -*- coding: utf-8 -*-
"""
demo/mission_demo.py —— ★ 端到端任务演示

模拟真实的分工：
    组1  →  给一张场景图（★ 这次用【我们自己雷达建的房间图】当它），
            并规定场景坐标系
    组2  →  给一串任务：
              任务一  导航 (0, 0, 0)   →  (2, 4, 1.57)
              任务二  握手    —— 不属于导航组，我们不动
              任务三  导航 (2, 4, 1.57) → (5, 2, -1.57)
    我们  →  收到位姿就精确走过去，返回 NavResult

★ 关键演示点
------------
· 时间只对齐一次：开机时狗真的停在组2 给的起点上 → 建立 T，之后全程不变
· 多段任务【不重新对齐】—— 接着走
· 场景系原点定在哪，我们的代码看不见（这次是程序搜出来的一个位置）
· 组2 只给【位姿】，不给路径、不给速度
"""
import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose6D, FRAME_SCENE, FRAME_CAPO        # noqa: E402
from hal.pose_source import OdomSource                         # noqa: E402
from hal.cmd_sink import SportSink                             # noqa: E402
from maps.grid import OccupancyGrid                            # noqa: E402
from mission.api import Navigator, NavigateToConfig            # noqa: E402
from control.fine_approach import ApproachConfig               # noqa: E402
from control.path_follower import FollowerConfig               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/mnt/d/迷雾通/nav/demo/demo_result.npz')
a, _ = ap.parse_known_args()

# ================================================================ 组2 的任务序列
TASKS = [
    ('导航',   (0.0, 0.0, 0.0),    (2.0, 4.0, 1.57)),
    ('握手',   None,               None),          # ★ 不属于导航组
    ('导航',   (2.0, 4.0, 1.57),   (5.0, 2.0, -1.57)),
]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Truth:
    def __init__(self, node, t0):
        qr = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self.p = None
        self.traj = []
        self.t0 = t0
        self.lock = threading.Lock()
        node.create_subscription(Odometry, '/sim/truth', self._cb, qr)

    def _cb(self, m):
        q = m.pose.pose
        with self.lock:
            self.p = (q.position.x, q.position.y, yaw_of(q.orientation))
            self.traj.append((time.time() - self.t0, q.position.x,
                              q.position.y, yaw_of(q.orientation)))

    def get(self):
        with self.lock:
            return self.p


def load_group1_scene(path='/mnt/d/迷雾通/nav/demo_scene.npz'):
    """读组1 给的场景图 + 场景系原点，并【把它换算到我们的内部场景系】。

    组1 说"场景系原点在房间的 (ox, oy)" ⇒ 我们只要把栅格的 origin 平移一下。
    ★ 我们的代码里【看不到】那个原点在哪 —— 它被吸收进栅格的 origin 了。
    """
    d = np.load(path)
    occ = d['occ'].astype(bool)
    map_origin = d['origin']           # 房间图自己的原点（建图时的）
    res = float(d['res'])
    so = d['scene_origin']             # ★ 组1 规定的场景系原点
    grid_origin = (float(map_origin[0] - so[0]), float(map_origin[1] - so[1]))
    return OccupancyGrid(occ, grid_origin, res), (float(so[0]), float(so[1]))


def main():
    t0 = time.time()
    rclpy.init()
    node = Node('mission_demo')
    truth = Truth(node, t0)

    grid, scene_origin = load_group1_scene()
    capo = OdomSource(node, '/SMX/Odom', frame=FRAME_CAPO, name='CAPO')
    sink = SportSink(node, acknowledge_motion=True)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print('=' * 76)
    print('  端到端任务演示')
    print('=' * 76)
    print('【组1】给了场景图：')
    print('       栅格 %s' % grid)
    print('       场景系原点定在房间的 (%.2f, %.2f)' % scene_origin)
    print('       ★ 这个原点在哪，我们的代码【看不见】——被吸收进栅格 origin 了')
    print()
    print('【组2】给了任务序列：')
    for i, (kind, s, g) in enumerate(TASKS, 1):
        if kind == '导航':
            print('       任务%d  %s  %s → %s' % (i, kind, s, g))
        else:
            print('       任务%d  %s  （★ 不归导航组）' % (i, kind))
    print()

    if not capo.wait_ready(20):
        print('  ✗ 等不到位姿 —— sim_dog 起了吗？')
        return 2
    while truth.get() is None:
        time.sleep(0.1)

    cfg = NavigateToConfig(
        pos_tol=0.05, yaw_tol_deg=10.0, timeout=90.0, inflate_radius=0.35,
        approach=ApproachConfig(timeout=40.0),
        follower=FollowerConfig(timeout=80.0),
    )
    nav = Navigator(capo, sink, cfg=cfg, grid=grid,
                    log=lambda s: print('      ' + s))

    # ---- ★ 开机对齐：只做这一次 ----
    start = TASKS[0][1]
    nav.align(Pose6D(start[0], start[1], 0.0, 0.0, 0.0, start[2],
                     frame=FRAME_SCENE))
    print('【对齐】完成（★ 全程只做这一次，多段任务不重新对齐）')
    print()

    results = []
    for i, (kind, s, g) in enumerate(TASKS, 1):
        print('-' * 76)
        if kind != '导航':
            print('任务%d: %s —— 不属于导航组，跳过（我们只做"给位姿走过去"）' % (i, kind))
            results.append((i, kind, None, None))
            print('-' * 76)
            print()
            continue
        print('任务%d: 导航 %s → %s' % (i, s, g))
        p0 = truth.get()
        res = nav.navigate_to(
            Pose6D(g[0], g[1], 0.0, 0.0, 0.0, g[2], frame=FRAME_SCENE))
        p1 = truth.get()
        terr = math.hypot(p1[0] - g[0], p1[1] - g[1])
        tyer = abs(math.degrees(math.atan2(math.sin(p1[2] - g[2]),
                                           math.cos(p1[2] - g[2]))))
        print('      %s' % res.summary())
        print('      ★ 真值误差 位置 %.3f m   朝向 %.1f°' % (terr, tyer))
        results.append((i, kind, res, (terr, tyer)))
        print('-' * 76)
        print()

    # ---- 汇总 ----
    print('=' * 76)
    print('  汇总')
    print('=' * 76)
    for (i, kind, res, te) in results:
        if res is None:
            print('  任务%d %-6s  —— 跳过（不归我们）' % (i, kind))
        else:
            print('  任务%d %-6s  %s  真值误差 %.3f m / %.1f°'
                  % (i, kind, '✅' if res.success else '❌', te[0], te[1]))
    print()

    with truth.lock:
        traj = np.array(truth.traj)
    np.savez(a.out, traj=traj,
             occ=grid.occ, origin=np.array(grid.origin), res=np.array(grid.res),
             scene_origin=np.array(scene_origin),
             tasks=np.array([[t[1][0], t[1][1], t[1][2],
                              t[2][0], t[2][1], t[2][2]] for t in TASKS
                            if t[1] is not None]))
    print('  轨迹已落盘 %s' % a.out)

    try:
        sink.settle()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
