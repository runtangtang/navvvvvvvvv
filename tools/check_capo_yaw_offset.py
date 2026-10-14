#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读检查 CAPO yaw 与 GO2 原始 IMU yaw 之间是否有恒定偏置。

★ 这是上传到狗上运行的只读脚本；它只订阅 /SMX/Odom 和 /lowstate，
  不创建发布器，不发送运动命令。Codex 不在狗上运行此脚本。

推荐运行方式（dogenv_run.sh 内含正确的 source 顺序）：
    bash ~/nav_ws/dogenv_run.sh ~/nav_ws/nav/tools/check_capo_yaw_offset.py

等价环境顺序：
    source /opt/ros/foxy/setup.bash
    source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
    source /home/unitree/nav_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    # CYCLONEDDS_URI 按 ~/nav_ws/dogenv_run.sh 设置后再运行本文件。

★ 输出的量就是 yaw0_out_（= 补丁从 yaw 里减掉的那个基准）：
      yaw0_out_  =  IMU 原始 yaw  −  CAPO 输出的 yaw
  推导：补丁是 yaw_out = OrientationRPY[6] − yaw0_out_，
        而 OrientationRPY[6] ≈ IMU 原始 yaw（滤波器直接观测它）
        ⇒ yaw0_out_ = IMU − CAPO

★ 判据：
    恒定且 |均值| > 5°  ⇒ 嫌疑 P 成立（10-08 那次预期 ≈ 【−154°】）
    |均值| ≤ 5°        ⇒ 嫌疑 P 不成立（元凶是 L：leg_ori_enable）

★★ 重要：本脚本只有在【CAPO 还带着补丁】时才有意义。
   回退补丁并重新编译之后，CAPO 输出 yaw == IMU yaw ⇒ 差值必然 ≈0。
   ⇒ 流程：★ 先跑本脚本 → 再回退/重编。
"""

import math
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowState


def yaw_xyzw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_wxyz(values):
    w, x, y, z = (float(values[i]) for i in range(4))
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def unwrap_near(angle, previous):
    if previous is None:
        return angle
    while angle - previous > math.pi:
        angle -= 2.0 * math.pi
    while angle - previous < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class YawOffsetChecker(Node):
    def __init__(self):
        super().__init__('check_capo_yaw_offset')
        capo_qos = QoSProfile(depth=100,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        lowstate_qos = QoSProfile(depth=20,
                                  reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST)
        self.capo_sub = self.create_subscription(
            Odometry, '/SMX/Odom', self.capo_cb, capo_qos)
        self.lowstate_sub = self.create_subscription(
            LowState, '/lowstate', self.lowstate_cb, lowstate_qos)
        self.timer = self.create_timer(0.5, self.report)
        self.started = time.monotonic()
        self.capo_yaw = None
        self.imu_yaw = None
        self.capo_pos = None
        self.last_capo_yaw = None
        self.last_imu_yaw = None
        self.last_difference = None
        self.differences = []

    def capo_cb(self, msg):
        # ★ 回调绝不抛异常（抛了会杀 executor）。
        try:
            pose = msg.pose.pose
            yaw = yaw_xyzw(pose.orientation)
            self.capo_yaw = unwrap_near(yaw, self.last_capo_yaw)
            self.last_capo_yaw = self.capo_yaw
            self.capo_pos = (float(pose.position.x), float(pose.position.y))
        except Exception as exc:
            self.get_logger().error('CAPO 回调异常：%s' % exc)

    def lowstate_cb(self, msg):
        # 字段和 import 已由本机 live_bridge.py 确认：
        # unitree_go.msg.LowState.imu_state.quaternion，顺序为 w,x,y,z。
        try:
            yaw = yaw_wxyz(msg.imu_state.quaternion)
            self.imu_yaw = unwrap_near(yaw, self.last_imu_yaw)
            self.last_imu_yaw = self.imu_yaw
        except Exception as exc:
            self.get_logger().error('LowState 回调异常：%s' % exc)

    def report(self):
        try:
            if self.capo_yaw is None or self.imu_yaw is None or self.capo_pos is None:
                return
            # ★ yaw0_out_ = IMU − CAPO（不是 CAPO − IMU，见文件头推导）
            difference = wrap_pi(self.imu_yaw - self.capo_yaw)
            difference = unwrap_near(difference, self.last_difference)
            self.last_difference = difference
            self.differences.append(math.degrees(difference))
            print('t=%5.1f  CAPO yaw=%+8.2f°  IMU yaw=%+8.2f°  '
                  '★yaw0_out_=%+8.2f°   CAPO pos=(%.3f, %.3f)'
                  % (time.monotonic() - self.started,
                     math.degrees(self.capo_yaw),
                     math.degrees(self.imu_yaw),
                     self.differences[-1],
                     self.capo_pos[0], self.capo_pos[1]), flush=True)
        except Exception as exc:
            self.get_logger().error('定时统计异常：%s' % exc)

    def print_summary(self):
        if not self.differences:
            print('没有同时收到 CAPO 与 IMU 数据，无法统计。', flush=True)
            return
        values = np.asarray(self.differences, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        print('\n★ yaw0_out_ 统计（IMU − CAPO，%d 个 0.5s 样本）' % len(values), flush=True)
        print('均值=%+.2f°  标准差=%.2f°  min=%+.2f°  max=%+.2f°'
              % (mean, std, float(np.min(values)), float(np.max(values))),
              flush=True)
        if std < 2.0 and abs(mean) > 5.0:
            print('判据：★ 嫌疑 P 【成立】（恒定且非零）—— 补丁确实是元凶。', flush=True)
            print('      ⇒ 10-08 那次预期约 −154°；数值对得上就更确凿。', flush=True)
        elif abs(mean) <= 5.0:
            print('判据：★ 嫌疑 P 【不成立】（均值接近 0）', flush=True)
            print('      ⇒ 元凶是 L（leg_ori_enable）而不是补丁 —— ★ 这个结论要记下来！', flush=True)
        else:
            print('判据：差值不是稳定常量，不能确认嫌疑 P。', flush=True)


def main():
    rclpy.init()
    node = YawOffsetChecker()
    print('只读订阅已启动：/SMX/Odom + /lowstate（Ctrl+C 结束）', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.print_summary()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(0)


if __name__ == '__main__':
    main()
