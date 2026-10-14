#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读观察 /nav/obstacle_points —— ★ 在【狗上】跑

为什么要有它（2026-10-08）：
    动态避障上真机时"走廊里到底有没有点"看不见。
    障碍节点只打印总点数，导航节点只在【跟随过程中】才做走廊检查，
    而 `ros2 topic echo` 从笔记本 WSL 收不到（连 /SMX/Odom 都空）。
    ★ 现有的 check_obstacle_points.sh / recv_obstacle.sh 是给【旧】障碍节点
      （话题 /obstacle_points、走 TCP 中继）用的，对新节点不适用。

它印什么：
    把每个点从 CAPO 世界系转到【狗体坐标系】—— 前 / 左 / 高，
    然后按导航节点的走廊判据（前方 LOOKAHEAD 内、左右 ±HALF）统计：
        走廊内有多少点  ⇒ ★ ≥ MIN_POINTS 就会触发"停车等待"

用法（狗上）:
    bash ~/nav_ws/dogenv_run.sh <本文件> [秒数=20] [走廊长=2.0] [走廊半宽=0.25]

★ 纯订阅，绝不发布任何东西、不发任何指令。
"""
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
LOOK = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0     # 走廊长度（前方）
HALF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25    # 走廊半宽（左/右）
MIN_PTS = 2                                                  # 与 local_avoid 的 min_points 一致

# ★ --dump <文件>：把最后几帧的点 + 当时的狗位姿写出来，取回笔记本画图
DUMP = None
if '--dump' in sys.argv:
    DUMP = sys.argv[sys.argv.index('--dump') + 1]
DUMP_N = 60                   # ★ 转储全程（报告间隔 2s ⇒ 60 帧 = 120 秒）
_frames = []                  # [(t, pose(x,y,yaw), pts(N,3) in capo 系)]


class Inspect(Node):
    def __init__(self):
        super().__init__('inspect_obstacle_points')
        q = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Float32MultiArray, '/nav/obstacle_points',
                                 self.pts_cb, q)
        self.create_subscription(Odometry, '/SMX/Odom', self.odom_cb, q)
        self.pts = None
        self.pts_t = 0.0
        self.pose = None
        self.frames = 0
        self.hits = []          # 每帧"走廊内点数"

    def pts_cb(self, m):
        try:
            a = np.asarray(m.data, dtype=np.float32)
            self.pts = a.reshape(-1, 3) if a.size else np.zeros((0, 3), np.float32)
            self.pts_t = time.time()
        except Exception as e:
            self.get_logger().error('pts 回调异常：%s' % e)

    def odom_cb(self, m):
        try:
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose = (p.x, p.y, yaw)
        except Exception as e:
            self.get_logger().error('odom 回调异常：%s' % e)


def main():
    rclpy.init()
    n = Inspect()
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < DUR:
        rclpy.spin_once(n, timeout_sec=0.05)
        now = time.time()
        if now - last_report < 2.0:
            continue
        last_report = now
        if n.pts is None or n.pose is None:
            print('  t=%5.1f  等数据…  (pts=%s pose=%s)'
                  % (now - t0, n.pts is not None, n.pose is not None), flush=True)
            continue
        age = now - n.pts_t
        x, y, yaw = n.pose
        c, s = math.cos(yaw), math.sin(yaw)
        P = n.pts
        if len(P):
            dx = P[:, 0] - x
            dy = P[:, 1] - y
            fwd = dx * c + dy * s            # 前方 [m]
            lft = -dx * s + dy * c           # 左侧 [m]
            h = P[:, 2]                      # 机体系高度
        else:
            fwd = lft = h = np.zeros(0)
        in_corr = (fwd > 0) & (fwd < LOOK) & (np.abs(lft) < HALF)
        ncorr = int(in_corr.sum())
        n.hits.append(ncorr)
        n.frames += 1
        if DUMP is not None and len(P):
            _frames.append((now - t0, (x, y, yaw), np.array(P, dtype=float)))
            if len(_frames) > DUMP_N:
                _frames.pop(0)
        print('  t=%5.1f  总点 %4d  走廊内 %3d %s   前方最近 %s  高度 %s'
              % (now - t0, len(P), ncorr,
                 '★★ 会停!' if ncorr >= MIN_PTS else '（不停）',
                 ('%.2fm' % fwd[fwd > 0].min()) if (len(fwd) and (fwd > 0).any()) else '  — ',
                 ('%.2f~%.2f' % (h.min(), h.max())) if len(h) else '  — '),
              flush=True)
        if len(P):
            # 前方 3m 内的点，按"前方距离"分桶，看看障碍大概在哪
            m = (fwd > 0) & (fwd < 3.0)
            if m.any():
                b = np.histogram(fwd[m], bins=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])[0]
                print('        前方分布 0-0.5:%-3d 0.5-1:%-3d 1-1.5:%-3d '
                      '1.5-2:%-3d 2-2.5:%-3d 2.5-3:%-3d'
                      % tuple(b), flush=True)

    print()
    print('=' * 60)
    if n.hits:
        h = np.asarray(n.hits)
        print('  共 %d 帧；走廊内点数：均值 %.1f  最大 %d  '
              '★ 会触发的帧: %d/%d'
              % (len(h), h.mean(), h.max(), int((h >= MIN_PTS).sum()), len(h)))
        if (h >= MIN_PTS).any():
            print('  ⇒ ★ 动态避障【会停】')
        else:
            print('  ⇒ ★ 走廊里从来没到过 %d 点 ⇒ 动态避障【不会停】' % MIN_PTS)
            print('     可能原因：① 障碍不在走廊（±%.2fm）里 ② 不在高度带'
                  ' ③ 雷达看不到（视野/遮挡） ④ 椅子太矮' % HALF)
    else:
        print('  没采到帧 —— /nav/obstacle_points 或 /SMX/Odom 没数据？')
    print('=' * 60)

    if DUMP is not None:
        with open(DUMP, 'w') as f:
            f.write('# /nav/obstacle_points 转储（CAPO 世界系；z = 机体系高度）\n')
            f.write('# LOOK=%.2f HALF=%.2f MIN_PTS=%d\n' % (LOOK, HALF, MIN_PTS))
            for tt, (px, py, pyaw), P in _frames:
                f.write('# t=%.2f pose=%.4f %.4f %.4f\n'
                        % (tt, px, py, math.degrees(pyaw)))
                for p in P:
                    f.write('%.4f %.4f %.4f\n' % (p[0], p[1], p[2]))
        print('  ★ 已转储 %d 帧 → %s' % (len(_frames), DUMP))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
