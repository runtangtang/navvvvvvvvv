#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""狗上轻量障碍感知节点：点云逐点去畸变后发布 CAPO 系 XYZ。

本文件基于 ``D:/capo_live/脚本_历史/dog_obstacle_node.py`` 改造；
外参、过滤范围、位姿缓冲和逐点去畸变公式保持原样，只把输出换成
``/nav/obstacle_points`` 的 ``Float32MultiArray``。
"""
import argparse
import math
import time
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray


# ★★★ 不用 sensor_msgs_py —— 狗的 Foxy 没这个包。
#     直接按 msg.fields 里的 offset 自己解析（任何 ROS2 版本都能跑）
_NP_TYPE = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
            5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def parse_cloud(msg):
    """把 PointCloud2 解析成结构化 numpy 数组。"""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        names.append(f.name)
        formats.append(_NP_TYPE.get(f.datatype & 0x7f, np.uint8))
        offsets.append(f.offset)
    dt = np.dtype({'names': names, 'formats': formats,
                   'offsets': offsets, 'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dt)


ap = argparse.ArgumentParser()
# ★★ 2026-10-08：判据改了 —— 现在用的是官方机身系点云 cloud_base
#   那里【地面是一条精确的水平线 z ≈ -0.29】
#   ⇒ 判据简化成：【比地面高的点 = 障碍】
#   zlo = 地面之上的门槛（-0.25 ⇒ 比地面高 4cm 以上才算）
#   zhi = 上界（挡掉天花板之类，实测用不到）
# ★★★ 2026-10-08 实测修正：-0.25 太松 —— 会把【地毯/地面起伏】放进来
#   实测（真机、走廊里没人）：
#       地面     z ≈ -0.29
#       地毯/起伏 z ≈ -0.29 ~ -0.16（只比地面高 0.11~0.13m）← 这些是【假障碍】
#       真障碍   z 明显更高（箱子顶到 +0.10，即高出地面 0.39m）
#   ⇒ 门槛提到 -0.15（比地面高 ~0.14m）
#     代价：比地面高不到 14cm 的东西看不见（狗本来也能跨过去，可接受）
ap.add_argument('--zlo', type=float, default=-0.15)
ap.add_argument('--zhi', type=float, default=1.5)
ap.add_argument('--rmin', type=float, default=0.8)    # ★ 近场（滤掉雷达自身/狗自己）
ap.add_argument('--rmax', type=float, default=8.0)
ap.add_argument('--rate', type=float, default=10.0)   # 发送降频
# ★★ 2026-10-08 加：累积窗口 [秒]
#   为什么：点云太稀疏 —— 文档实测「人在 1.5m 处只有 3~8 个点」，
#   真机实测：人走过正前方时，走廊里【只有某一瞬】有 13 点，前后都是 0
#   ⇒ 单帧判据会"时有时无" ⇒ 狗走走停停
#   ★ 解法：把最近 N 秒的点【累积】后再发 —— 站着不动的人会被多帧叠出来
#   代价：延迟 +N 秒（狗才 0.5m/s，可接受）
ap.add_argument('--acc-s', type=float, default=0.8)
ap.add_argument('--acc-max', type=int, default=600)   # 累积点数上限（防爆）
args = ap.parse_args()


# ★★★★★ 2026-10-08 晚：★ 不用外参了 —— 改用【官方的机身系点云】
#
#   背景：我们用 /utlidar/cloud（★ frame_id = 'utlidar_lidar'，【雷达系】）
#         自己乘一个外参转到机身系。那个外参一直是老大难：
#     · 2026-09-21 发现抄了 CMU 的值（错）→ 改成 diag(-1,1,1)
#     · 2026-10-08 发现它缺一个 ~15° 俯仰 ⇒ 地面在机体系里是【斜的】
#       ⇒ 从前方 ~1m 起"爬"进高度带 ⇒ 地面被当成障碍 ⇒ 动态避障【一直误触发】
#     · Codex 任务 04 离线标定出候选（15.0943° → 0.0227°），
#       但发现原矩阵 det = -1（是【反射】不是旋转）⇒ 手性问题没根治
#
#   ★ 然后发现：狗上一直有个【官方转好的机身系点云】
#        /utlidar/cloud_base      frame_id = 'base_link'
#     实测（真机、箱子在前方）：
#        · 地面  z ≈ -0.29，各距离桶的中位数都是 -0.27~-0.31  ⇒ ★【地面是平的】
#        · 箱子在 1.5~2.5m 处被稳定看到（z 到 +0.10）
#        · 正前方 ±0.8m × 2.5m：高于地面的点 【0 个】
#     ⇒ ★★ 官方已经做了外参变换，而且【做对了】
#     ⇒ 我们【不再自己做任何坐标变换】，外参全删
#
#   ⚠️ 回滚（如果要回到自己转的老路）：
#        cp obstacle_node.py.bak-20261008-precallb obstacle_node.py
#
#   坐标系约定（base_link 是标准 ROS，REP-103）：
#        x = 前   y = 【左】   z = 上
#        地面 z ≈ -0.29（= base_link 原点离地约 0.29m）
#      ★ 实测核对：前方 1.15m 处有一根 5 点的竖线在 y ≈ -0.95（= 右侧 0.95m），
#        与标准约定一致（尚未做"人站左边"的正向验证 —— 待补）


class DogObstacle(Node):
    def __init__(self):
        super().__init__('dog_obstacle')
        qos = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        # 位姿缓冲：按【到达时刻】存（不用 header.stamp —— 两个时钟不一样）
        self.buf = deque(maxlen=4000)
        self.create_subscription(Odometry, '/SMX/Odom', self.odom_cb, qos)
        # ★ 2026-10-08：改用【官方的机身系点云】（不用自己转外参了，见文件头）
        self.create_subscription(PointCloud2, '/utlidar/cloud_base',
                                 self.cloud_cb, qos)
        self.pub = self.create_publisher(Float32MultiArray,
                                         '/nav/obstacle_points', 10)
        self.n = 0
        self.t_last_send = 0.0
        self.acc = deque()      # ★ 累积窗口：[(到达时刻, (N,3) 世界系点), ...]
        self.get_logger().info(
            'dog_obstacle up  z∈[%.2f,%.2f]  r∈[%.1f,%.1f]  发送 %.0f Hz  '
            '累积 %.2fs(上限 %d 点)'
            % (args.zlo, args.zhi, args.rmin, args.rmax, args.rate,
               args.acc_s, args.acc_max))

    def odom_cb(self, m):
        # ★ 回调绝不抛异常。
        try:
            self._odom_cb(m)
        except Exception as exc:
            self.get_logger().error('odom 回调异常：%s' % exc)

    def _odom_cb(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        if self.buf:
            prev = self.buf[-1][3]
            while yaw - prev > math.pi:
                yaw -= 2 * math.pi
            while yaw - prev < -math.pi:
                yaw += 2 * math.pi
        self.buf.append((time.time(), p.x, p.y, yaw))

    def _publish(self, out):
        msg = Float32MultiArray()
        msg.data = np.asarray(out, dtype=np.float32).reshape(-1).tolist()
        self.pub.publish(msg)
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info('发了 %d 帧，本帧 %d 点 (%.1f KB)'
                                   % (self.n, len(out), len(out) * 12 / 1024))

    def cloud_cb(self, msg):
        # ★ 回调绝不抛异常。
        try:
            self._cloud_cb(msg)
        except Exception as exc:
            self.get_logger().error('cloud 回调异常：%s' % exc)

    def _cloud_cb(self, msg):
        if not self.buf:
            return
        t_now = time.time()
        # 降频：不需要每帧都发
        if t_now - self.t_last_send < 1.0 / args.rate:
            return
        self.t_last_send = t_now

        a = parse_cloud(msg)
        if not len(a):
            self._publish(np.empty((0, 3), dtype=np.float32))
            return
        xyz = np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)
        ok = np.isfinite(xyz).all(axis=1)
        xyz = xyz[ok]
        tt = a['time'][ok].astype(np.float64) if 'time' in a.dtype.names \
            else np.zeros(len(xyz))
        if not len(xyz):
            self._publish(np.empty((0, 3), dtype=np.float32))
            return

        # ★ 2026-10-08：cloud_base 已经是【机身系】，【不再做任何坐标变换】
        B = xyz
        # 高度带：地面在 z ≈ -0.29（水平）⇒ 比地面高的就是障碍
        m1 = (B[:, 2] > args.zlo) & (B[:, 2] < args.zhi)
        B = B[m1]; tt = tt[m1]
        if not len(B):
            self._publish(np.empty((0, 3), dtype=np.float32))
            return
        # 距离带
        d = np.hypot(B[:, 0], B[:, 1])
        m2 = (d > args.rmin) & (d < args.rmax)
        B = B[m2]; tt = tt[m2]
        if not len(B):
            self._publish(np.empty((0, 3), dtype=np.float32))
            return

        # ★ 每个点用它自己时刻的位姿（等价于去畸变）
        buf = np.array(self.buf, dtype=np.float64)
        span = float(tt.max() - tt.min())
        t_pt = (t_now - span) + (tt - tt.min())
        X = np.interp(t_pt, buf[:, 0], buf[:, 1])
        Y = np.interp(t_pt, buf[:, 0], buf[:, 2])
        A = np.interp(t_pt, buf[:, 0], buf[:, 3])
        c, s = np.cos(A), np.sin(A)
        out = np.empty((len(B), 3), dtype=np.float32)
        out[:, 0] = c * B[:, 0] - s * B[:, 1] + X
        out[:, 1] = s * B[:, 0] + c * B[:, 1] + Y
        out[:, 2] = B[:, 2]                      # 机体系高度（正 = 比机身原点高）

        # ★★ 2026-10-08：累积最近 acc_s 秒的点再发（对抗点云稀疏，见文件头说明）
        self.acc.append((time.time(), out))
        cutoff = time.time() - max(args.acc_s, 0.0)
        while self.acc and self.acc[0][0] < cutoff:
            self.acc.popleft()
        if len(self.acc) > 1:
            pts = np.vstack([a for _, a in self.acc])
        else:
            pts = out
        if len(pts) > args.acc_max:              # 上限：留最新的
            pts = pts[-args.acc_max:]
        self._publish(pts)


def main():
    rclpy.init()
    try:
        rclpy.spin(DogObstacle())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
