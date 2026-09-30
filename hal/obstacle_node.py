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
ap.add_argument('--zlo', type=float, default=0.8)     # 机体系高度下界（滤地面/自体）
ap.add_argument('--zhi', type=float, default=2.0)     # 上界
ap.add_argument('--rmin', type=float, default=1.2)    # 近场（雷达盲区+自体）
ap.add_argument('--rmax', type=float, default=8.0)
ap.add_argument('--rate', type=float, default=10.0)   # 发送降频，进一步减小数据量
args = ap.parse_args()


# 雷达 → 机身(IMU) 外参（本项目验证过的；★ z 那一行不翻转）
EXT_R = np.array([[-1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
EXT_T = np.array([0.315, 0.0, 0.3])


class DogObstacle(Node):
    def __init__(self):
        super().__init__('dog_obstacle')
        qos = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        # 位姿缓冲：按【到达时刻】存（不用 header.stamp —— 两个时钟不一样）
        self.buf = deque(maxlen=4000)
        self.create_subscription(Odometry, '/SMX/Odom', self.odom_cb, qos)
        self.create_subscription(PointCloud2, '/utlidar/cloud', self.cloud_cb, qos)
        self.pub = self.create_publisher(Float32MultiArray,
                                         '/nav/obstacle_points', 10)
        self.n = 0
        self.t_last_send = 0.0
        self.get_logger().info(
            'dog_obstacle up  z∈[%.2f,%.2f]  r∈[%.1f,%.1f]  发送 %.0f Hz'
            % (args.zlo, args.zhi, args.rmin, args.rmax, args.rate))

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

        # 雷达 → 机身
        B = xyz @ EXT_R.T + EXT_T
        # 高度带（滤地面 + 狗自己）
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

        self._publish(out)


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
