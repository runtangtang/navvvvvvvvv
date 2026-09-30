# -*- coding: utf-8 -*-
"""
tests/test_p1_sim.py —— P1 验收：仿真里跑通点对点

一次跑一个用例（由 run_p1.sh 驱动），用例参数：
    --goal-dist    走多远 [m]
    --vx-mode      const | prop

验收项（对应任务规划 P1）
  ① 无里程计误差：   走 1m，最终误差 < 3cm
  ② 注入 -6% 尺度：  最终误差 ≈ 6cm（★ 证明缺陷注入能量化影响）
  ③ 注入 240ms 延迟：观察是否失稳
  ④ 接比例控制：     必须复现卡死（★ 证明仿真还有效）

输出的判定依据：把 /sim/truth（仿真里的"真值"）和 /SMX/Odom（CAPO 读数）对比。
（/sim/truth 只有仿真里有；真机验收在 P4。）
"""
import argparse
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose2D, Pose6D, FRAME_CAPO, FRAME_SCENE   # noqa: E402
from hal.pose_source import OdomSource                              # noqa: E402
from hal.cmd_sink import SportSink                                  # noqa: E402
from control.fine_approach import ApproachConfig                   # noqa: E402
from mission.api import Navigator, NavigateToConfig                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--goal-dist', type=float, default=1.0)
ap.add_argument('--vx-mode', default='const')
ap.add_argument('--capo-scale', type=float, default=1.0,
                help='只是显示用 —— 实际的尺度由 sim_dog 那边注入')
ap.add_argument('--label', default='')
ap.add_argument('--expect', default='reach', choices=('reach', 'stall', 'any'),
                help='reach=必须到位; stall=必须卡死(比例控制); any=只看数据')
a, _ = ap.parse_known_args()


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Truth:
    """仿真里才有 —— 真值位姿（给验收用）

    ★ 注意：订阅建在【传进来的那个 node】上，不要自己再开一个 Node ——
      之前自己开了一个，但只 spin 了主 node，这个的回调永远不触发，
      结果 `while truth.get() is None` 把测试卡死了 9 分钟。
    """

    def __init__(self, node):
        qr = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self.p = None
        self.traj = []                      # 真值轨迹，给可视化用
        self.t0 = time.time()
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


def main():
    rclpy.init()
    node = Node('p1_test')
    truth = Truth(node)

    capo = OdomSource(node, '/SMX/Odom', frame=FRAME_CAPO, name='CAPO(或sim)')
    sink = SportSink(node, acknowledge_motion=True)   # sim_dog 在听同一个话题
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print('=' * 72)
    print('  P1 验收 %s' % ('— ' + a.label if a.label else ''))
    print('  目标距离 %.2f m | 控制律 %s | simulator capo-scale %.3f'
          % (a.goal_dist, a.vx_mode, a.capo_scale))
    print('=' * 72)

    if not capo.wait_ready(20):
        print('  ✗ 等不到位姿 —— sim_dog 起来了吗？')
        return 2
    while truth.get() is None:
        time.sleep(0.1)

    cfg = NavigateToConfig(
        pos_tol=0.03,
        yaw_tol_deg=5.0,
        timeout=40.0,
        approach=ApproachConfig(vx_mode=a.vx_mode, timeout=30.0),
    )
    nav = Navigator(capo, sink, cfg=cfg, log=lambda s: print('    ' + s))

    # ---- 开机对齐：假设起点就是场景系原点、朝 +x ----
    T = nav.align(Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, frame=FRAME_SCENE))
    capo0 = capo.get().data
    truth0 = truth.get()

    # ---- 目标：正前方 goal_dist 米 ----
    goal = Pose6D(a.goal_dist, 0.0, 0.0, 0.0, 0.0, 0.0, frame=FRAME_SCENE)
    res = nav.navigate_to(goal)
    if res.success and not res.reason == 'timeout':
        pass

    # ---- 统计 ----
    capo1 = capo.get().data
    truth1 = truth.get()

    capo_moved = math.hypot(capo1.x - capo0.x, capo1.y - capo0.y)
    capo_fwd = capo1.x - capo0.x
    truth_moved = math.hypot(truth1[0] - truth0[0], truth1[1] - truth0[1])
    truth_err = math.hypot(truth1[0] - a.goal_dist, truth1[1])
    k = capo_moved / truth_moved if truth_moved > 1e-6 else float('nan')

    print()
    print('-' * 72)
    print('  %s' % res.summary())
    print('-' * 72)
    print('  %-26s %10s' % ('命令走', '%.3f m' % a.goal_dist))
    print('  %-26s %10.3f m' % ('CAPO(仿真里程计) 报位移', capo_moved))
    print('  %-26s %10.3f m' % ('★ 仿真真值 实际位移', truth_moved))
    print('  %-26s %10.3f m' % ('★ 真值下的最终误差', truth_err))
    print('  %-26s %10.3f' % ('里程计尺度 k = CAPO/真值', k))
    print('  %-26s %10.4f m  (★ 恒等式，≈0 是应该的)'
          % ('导航器报的内部误差', res.internal_error))
    print('  %-26s %10.3f m  (估计)' % ('不确定度', res.uncertainty))
    print('  %-26s %10.1f s' % ('用时', res.elapsed))
    print('  sink 统计: %s' % sink.stats())
    print('-' * 72)

    # ---- 判定 ----
    print()
    if a.expect == 'stall':
        # 比例控制：判据是【失败/卡死】，不是走了多远。
        # 真机 2026-09-22 卡在 7.6cm；仿真卡在 7.8cm —— 同一个机制。
        stall = (res.reason in ('stuck', 'timeout', 'position_out_of_tol')
                 or not res.success)
        pos = 0.04 < truth_err < 0.20
        print('  ④ 比例控制卡死复现: %s  (reason=%s, 停在离目标 %.3f m)'
              % ('✅ 复现' if (stall and pos) else '❌ 没复现', res.reason, truth_err))
        print('     真机基准：卡在 0.076 m')
        ok = stall and pos
    elif a.expect == 'reach':
        good = res.success and truth_err < 0.03
        print('  最终真值误差 %.4f m（阈值 0.03）: %s'
              % (truth_err, '✅ 通过' if good else '❌ 超标'))
        ok = good
    else:
        print('  （any：不判定）')
        ok = True
    print('=' * 72)

    try:
        import numpy as np
        with truth.lock:
            traj = np.array(truth.traj)
        np.savez('/mnt/d/迷雾通/nav/p1_%s.npz' % (a.label or 'case'),
                 traj=traj,
                 capo=np.array([(capo0.x, capo0.y), (capo1.x, capo1.y)]),
                 meta=np.array([a.goal_dist, truth_err, capo_moved, truth_moved,
                                k if k == k else -1, 1.0 if res.success else 0.0]),
                 goal=np.array([a.goal_dist, 0.0]))
    except Exception as e:
        print('  (保存轨迹失败: %s)' % e)

    try:
        sink.settle()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
