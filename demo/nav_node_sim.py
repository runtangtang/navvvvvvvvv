# -*- coding: utf-8 -*-
"""
demo/nav_node_sim.py —— 【我们的主程序】在仿真里跑

    sim_dog（仿真狗，发 /SMX/Odom）
        ↓
    Navigator（纯逻辑，不依赖 ROS）
        ↓
    RosTopicBridge（订阅 /nav/navigate_to，发 /nav/result + /nav/feedback）

★ 对"组2"来说，它就是在跟一个黑盒说话：给 goal，收 feedback/result。
★ 换成真机时，唯一要改的是把 sim_dog 换成真 CAPO + 真 SportSink。
"""
import os
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose6D, FRAME_CAPO, FRAME_SCENE   # noqa: E402
from hal.pose_source import OdomSource                    # noqa: E402
from hal.cmd_sink import SportSink                        # noqa: E402
from hal.obstacle_source import RosObstacleSource         # noqa: E402
from maps.grid import OccupancyGrid                       # noqa: E402
from mission.api import Navigator, NavigateToConfig       # noqa: E402
from mission.ros_topics import RosTopicBridge             # noqa: E402
# ★ Action 版（真接口）—— 只在 --action 时 import（没装 nav_interfaces 也能跑 topic 版）
from control.fine_approach import ApproachConfig          # noqa: E402
from control.path_follower import FollowerConfig          # noqa: E402


def load_scene(path='/mnt/d/迷雾通/nav/demo_scene.npz'):
    d = np.load(path)
    occ = d['occ'].astype(bool)
    mo = d['origin']
    so = d['scene_origin']
    return OccupancyGrid(occ, (float(mo[0] - so[0]), float(mo[1] - so[1])),
                         float(d['res']))


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--start', nargs=3, type=float, default=None,
                help='★ 可选：启动即对齐 x y yaw（弧度）。不给（推荐）= '
                     '等第一个 goal 的 s_* 来对齐')
ap.add_argument('--action', action='store_true',
                help='★ 用正式 Action 接口（nav_interfaces）而不是 JSON/topic')
ap.add_argument('--dynamic-avoid', action='store_true',
                help='★ 开动态避障（默认关）。需要障碍点源才有意义')
ap.add_argument('--grid', default=None,
                help='★ 场景栅格 npz；不给就用 demo_scene.npz。'
                     '（2026-09-26 加：为了能在仿真里测带障碍的自定义场景，'
                     '比如 tools/cube_scene.npz —— 之前路径写死了）')
_a, _ = ap.parse_known_args()


def main():
    rclpy.init()
    node = Node('nav_main')
    log = lambda s: print('  [nav] %s' % s, flush=True)

    grid = load_scene(_a.grid) if _a.grid else load_scene()
    log('用栅格: %s' % grid)
    capo = OdomSource(node, '/SMX/Odom', frame=FRAME_CAPO, name='CAPO(sim)')
    sink = SportSink(node, acknowledge_motion=True)

    # ★★ 必须先把 spin 起起来，否则回调永远不触发 → wait_ready 必然超时
    #    （同一天里这个坑我踩了三次：Truth 节点、以及这里两次）
    #
    # ★★★ Action 模式【必须用 MultiThreadedExecutor】：
    #     Action 的 execute_callback 会一直阻塞到任务结束；
    #     单线程 executor 下它会把【500Hz 的位姿订阅】一起堵死 ⇒ 导航直接废掉。
    if _a.action:
        from rclpy.executors import MultiThreadedExecutor
        _ex = MultiThreadedExecutor()
        _ex.add_node(node)
        threading.Thread(target=_ex.spin, daemon=True).start()
    else:
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cfg = NavigateToConfig(
        pos_tol=0.05, yaw_tol_deg=10.0, timeout=120.0,
        follow_timeout=90.0, approach_timeout=40.0,
        inflate_radius=0.35,
        approach=ApproachConfig(timeout=40.0),
        follower=FollowerConfig(timeout=90.0),
        dynamic_avoid=_a.dynamic_avoid,
    )
    obstacles = RosObstacleSource(node) if _a.dynamic_avoid else None
    nav = Navigator(capo, sink, cfg=cfg, grid=grid, log=log,
                    obstacle_source=obstacles)

    if not capo.wait_ready(20):
        print('✗ 等不到位姿 —— sim_dog 起了吗？')
        return 2

    # ★★ 开机对齐（2026-09-27 改：默认等 goal 的 s_*，不再默认 (0,0,0)）
    import math as _m
    if _a.start is not None:
        nav.align(Pose6D(_a.start[0], _a.start[1], 0, 0, 0,
                         _a.start[2], frame=FRAME_SCENE))
        print('【nav】★ 用 --start 显式对齐 @ (%.3f, %.3f, %.1f° = %.4f rad)'
              % (_a.start[0], _a.start[1], _m.degrees(_a.start[2]), _a.start[2]),
              flush=True)
        print('【nav】  ⇒ goal 里的 s_* 之后只做一致性校验', flush=True)
    else:
        print('【nav】★ 未做开机对齐 —— 等第一个 goal 的 s_* 来对齐',
              flush=True)
    print('【nav】等待 /nav/navigate_to ...', flush=True)

    if _a.action:
        from mission.ros_action import RosActionServer
        bridge = RosActionServer(node, nav, log=log)
    else:
        bridge = RosTopicBridge(node, nav, log=log)
    try:
        while True:                 # spin 已经在后台线程跑了，这里只等
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sink.settle()
        except Exception:
            pass
        os._exit(0)


if __name__ == '__main__':
    main()
