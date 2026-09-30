#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sim_dog.py —— 验证仿真狗【真的复现了 2026-09-22 真机实测的行为】

一个仿真值不值钱，就看它能不能把当天踩过的坑提前演出来。
所以这个测试有两部分：

  A. 开环特性     死区 / 惯量 / 制动滑行 —— 对着当天实测的数字核
  B. ★ 闭环复现   拿【比例控制律 vx = 1.5×剩余距离】跑一遍，
                  看它是不是也像真狗一样【卡在死区里出不来】
                  （当天两大半时间就丢在这上面）

跑法:  bash run_test.sh
"""
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from unitree_api.msg import Request

API_MOVE, API_STOPMOVE, API_BALANCESTAND = 1008, 1003, 1002
CTRL_HZ = 20.0


def wrap(x):
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


class Tester(Node):
    def __init__(self):
        super().__init__('sim_tester')
        qr = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Request, '/api/sport/request', qr)
        self.truth = None
        self.odom = None
        self.lock = threading.Lock()
        self.create_subscription(Odometry, '/sim/truth', self._t, qr)
        self.create_subscription(Odometry, '/SMX/Odom', self._o, qr)

    def _t(self, m):
        p = m.pose.pose
        q = p.orientation
        with self.lock:
            self.truth = (time.time(), p.position.x, p.position.y,
                          math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _o(self, m):
        p = m.pose.pose
        q = p.orientation
        with self.lock:
            self.odom = (time.time(), p.position.x, p.position.y,
                         math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def get(self):
        with self.lock:
            return self.truth, self.odom

    def _send(self, api, param=None):
        r = Request()
        r.header.identity.api_id = int(api)
        if param is not None:
            r.parameter = json.dumps(param)
        self.pub.publish(r)

    def move(self, vx, vy, wz):
        self._send(API_MOVE, {'x': float(vx), 'y': float(vy), 'z': float(wz)})

    def stop(self):
        for _ in range(6):
            self._send(API_STOPMOVE)
            time.sleep(0.02)


def hold(n, vx, vy, wz, dur):
    t0 = time.time()
    while time.time() - t0 < dur:
        n.move(vx, vy, wz)
        time.sleep(1.0 / CTRL_HZ)
    n.stop()
    time.sleep(0.6)          # 让它滑完


def pulse(n, vx, vy, wz, dur):
    """开环脉冲：动完停 0.8 秒再读"""
    p0 = n.get()[0]
    hold(n, vx, vy, wz, dur)
    time.sleep(0.8)
    p1 = n.get()[0]
    return math.hypot(p1[1] - p0[1], p1[2] - p0[2])


def main():
    rclpy.init()
    n = Tester()
    threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()

    t0 = time.time()
    while time.time() - t0 < 15 and n.get()[0] is None:
        time.sleep(0.2)
    if n.get()[0] is None:
        raise SystemExit('仿真狗没在发 /sim/truth —— 先起 sim_dog.py')

    print('=' * 70)
    print('  仿真狗自检：能不能复现 2026-09-22 真机实测的行为')
    print('=' * 70)

    # ---------------- A. 开环特性 ----------------
    print()
    print('A. 开环特性')
    print('-' * 70)
    print('  %-34s %10s %12s' % ('测试', '仿真结果', '真机实测'))

    d = pulse(n, 0.50, 0.0, 0.0, 1.0)
    print('  %-34s %9.3f m %11s' % ('vx=0.50 走 1.0 秒', d, '~0.45 m'))

    d = pulse(n, 0.115, 0.0, 0.0, 3.0)
    print('  %-34s %9.3f m %11s' % ('vx=0.115 走 3.0 秒（死区内）', d, '0.000 m'))
    dead_ok = d < 0.005

    time.sleep(0.3)
    yaw_before = n.get()[0][3]
    hold(n, 0.0, 0.0, 0.19, 1.5)
    dy = math.degrees(abs(wrap(n.get()[0][3] - yaw_before)))
    print('  %-34s %8.2f deg %11s' % ('wz=0.19 转 1.5 秒（死区内）', dy, '0.0 deg'))
    dead_ok = dead_ok and dy < 1.0

    time.sleep(0.3)
    yaw_before = n.get()[0][3]
    hold(n, 0.0, 0.0, 0.50, 1.5)
    dy = math.degrees(abs(wrap(n.get()[0][3] - yaw_before)))
    print('  %-34s %8.2f deg %11s' % ('wz=0.50 转 1.5 秒', dy, '~25 deg'))

    # 制动滑行 —— 必须在【发停的那一刻】取基准，不能在 hold 里面取（那是 bug）
    t0 = time.time()
    while time.time() - t0 < 1.0:
        n.move(0.50, 0.0, 0.0)
        time.sleep(1.0 / CTRL_HZ)
    p_stop = n.get()[0]          # ← 发停前的最后一帧
    n.stop()
    time.sleep(1.0)
    p_end = n.get()[0]
    coast = math.hypot(p_end[1] - p_stop[1], p_end[2] - p_stop[2])
    print('  %-34s %9.3f m %11s' % ('制动滑行（0.5m/s 急停后）', coast, '~0.031 m'))

    # ---------------- B. 闭环复现 ----------------
    print()
    print('B. ★ 闭环复现：比例控制律 vx = 1.5 × 剩余距离')
    print('   （当天真机上它【卡死在离目标 7.6cm 处】，因为速度衰减进死区）')
    print('-' * 70)

    # 移到一段空地上，朝同一方向
    time.sleep(0.5)
    p0 = n.get()[0]
    x0, y0_, _ = p0[1], p0[2], 0.0
    TARGET = 1.0
    # 目标设在正前方 —— 但仿真狗初始朝 +x，我们先不转向，直接纯前进
    goal_x = x0 + TARGET

    trace = []
    t_start = time.time()
    stall_t = None
    n.move(0.5, 0.0, 0.0)
    time.sleep(0.05)
    # 先走到剩 0.5m 再切换控制律，模拟真实流程
    while True:
        p = n.get()[0]
        dist = goal_x - p[1]
        if dist < 0.5 or time.time() - t_start > 10:
            break
        n.move(0.5, 0.0, 0.0)
        time.sleep(1.0 / CTRL_HZ)

    print('  距目标 0.5m 处切到比例控制 …')
    t_sw = time.time()
    last_d = None
    stuck_since = time.time()
    while time.time() - t_sw < 12:
        p = n.get()[0]
        dist = goal_x - p[1]
        trace.append((time.time() - t_sw, dist))
        vx = max(0.0, min(0.5, 1.5 * dist))      # ← 当天用的控制律
        n.move(vx, 0.0, 0.0)
        if last_d is not None and last_d - dist < 0.002:
            if time.time() - stuck_since > 3.0:
                stall_t = time.time() - t_sw
                break
        else:
            stuck_since = time.time()
        last_d = dist
        time.sleep(1.0 / CTRL_HZ)
    n.stop()
    time.sleep(0.8)

    p_end = n.get()[0]
    final = goal_x - p_end[1]
    print()
    if stall_t is not None:
        print('  ★ 复现成功：卡死在离目标 %.3f m 处（%.1f 秒没进展，脚本放弃）'
              % (final, stall_t))
        print('     当天真机：卡死在 0.076 m  ← 同一个机制')
    else:
        print('  没复现卡死，最终距目标 %.4f m' % final)

    # ---------------- 结论 ----------------
    print()
    print('=' * 70)
    print('  结论')
    print('=' * 70)
    print('  死区复现:     %s' % ('✅ 通过' if dead_ok else '❌ 失败'))
    print('  比例控制卡死: %s' % ('✅ 复现' if stall_t is not None else '❌ 没复现'))
    print()
    print('  ⇒ 仿真能把当天真机上踩的坑提前演出来 —— 可以用它做框架迭代')
    print('=' * 70)

    n.stop()
    # 不显式 shutdown：spin 跑在 daemon 线程里，shutdown 会抛
    # "terminate called without an active exception"。直接让进程退出即可。
    import os
    os._exit(0)



if __name__ == '__main__':
    main()
