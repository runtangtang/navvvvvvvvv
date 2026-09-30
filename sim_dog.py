#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_dog.py —— 仿真狗（离线替代真 GO2）

为什么需要它
------------
2026-09-22 一整天，几乎每次迭代都是"上狗跑一次 5~10 分钟"：烧电池、要清场、
要人拿遥控器、而且还出过两次险情。真机上跑不了几个来回就天黑了。

这个仿真是把【当天实测到的东西】编进去，让导航框架的迭代能完全离线做：
改一行代码 → 仿真里跑 → 看结果，几秒钟一轮，而且**可复现**。

它同时是"里程计缺陷注入器"——用来量化"里程计差 X% 会让最终误差差多少"。

--------------------------------------------------------------------------
订阅  /api/sport/request     unitree_api/msg/Request
        1008 Move          {'x': vx, 'y': vy, 'z': wz}   机体系速度
        1003 StopMove
        1002 BalanceStand

发布  /SMX/Odom              nav_msgs/msg/Odometry   ← 模拟 CAPO（带缺陷）
      /sim/truth             nav_msgs/msg/Odometry   ← 真实位姿（给分析用）

--------------------------------------------------------------------------
模型参数（默认值全部来自 2026-09-22 真机实测）
--------------------------------------------------------------------------
✦ 执行器死区   |vx| < 0.15  → 输出 0    实测: vx=0.115 三秒不动 / vx=0.5 正常
               |wz| < 0.25  → 输出 0    实测: wz=0.19  九秒不转 / wz=0.5 正常
✦ 一阶惯性     τ ≈ 0.15 s              实测: 起步 50ms 内速度起来
✦ 制动滑行     同样用 τ 减速            实测: 0.5 m/s 急停后滑 3.1 cm
✦ 停稳转向     可配，默认 0（实测到 +7° 但方向/机制不明，先不写死）
✦ 转身挪位     可配，默认 0（实测 转 14.5° 挪 3.6cm，机制不明）

--------------------------------------------------------------------------
可注入的「里程计缺陷」（这是本文件最有用的地方）
--------------------------------------------------------------------------
  --capo-scale 0.94    CAPO 少算 6%（真实世界 1m，它报 0.94m）
  --capo-latency 240   输出延迟 240ms（模拟 Point-LIO 那种滞后）
  --capo-noise 0.002   位置噪声 σ=2mm（模拟 Point-LIO 的抖动）
  --seed 42            固定随机种子 ⇒ 完全可复现

用法
----
  # 干净环境（假设里程计完美）
  python3 sim_dog.py

  # 模拟真实的 CAPO（少算 6%、2mm 噪声）
  python3 sim_dog.py --capo-scale 0.94 --capo-noise 0.002

  # 模拟 Point-LIO 那样又慢又抖
  python3 sim_dog.py --capo-latency 0.24 --capo-noise 0.007
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from unitree_api.msg import Request

API_MOVE, API_STOPMOVE, API_BALANCESTAND = 1008, 1003, 1002

ap = argparse.ArgumentParser()
ap.add_argument('--rate', type=float, default=100.0, help='仿真步进/发布频率 [Hz]')
ap.add_argument('--x0', type=float, default=0.0)
ap.add_argument('--y0', type=float, default=0.0)
ap.add_argument('--yaw0', type=float, default=0.0, help='初始朝向 [deg]')
# ---- 执行器特性（实测默认值）----
ap.add_argument('--vx-deadband', type=float, default=0.15,
                help='前进速度死区 [m/s]（实测 0.115 不动 / 0.5 动）')
ap.add_argument('--wz-deadband', type=float, default=0.25,
                help='角速度死区 [rad/s]（实测 0.19 不动 / 0.5 转）')
ap.add_argument('--wz-gain', type=float, default=0.50,
                help='角速度实际达成的比例。实测真狗只有指令的约一半：'
                     '指令 0.4 rad/s → 实测 0.21；指令 0.8 → 实测 0.4')
ap.add_argument('--vx-gain', type=float, default=0.90,
                help='前进速度实际达成的比例（步态起步损失）')
ap.add_argument('--tau', type=float, default=0.15,
                help='起步/跟随的一阶时间常数 [s]')
ap.add_argument('--tau-stop', type=float, default=0.07,
                help='停车时间常数 [s]。StopMove 比一阶衰减更干脆：'
                     '实测 0.5 m/s 急停滑 3.1cm ⇒ 0.5×0.07≈0.035 对得上。'
                     '（用 --tau 会让滑行偏大一倍）')
ap.add_argument('--settle-yaw', type=float, default=0.0,
                help='停下后身体的额外转角 [deg]（实测约 7，机制不明，默认关闭）')
ap.add_argument('--walk-veer', type=float, default=0.0,
                help='★ 走路时的自然跑偏 [deg/m]（实测约 3.6，正值=左偏）。'
                     '为什么必须有：真狗走路不是绝对直的，一偏就会让'
                     '"到【近处】路点的方位角"几何爆炸，从而暴露跟随器的 bug。'
                     '仿真里不加这个 ⇒ 狗永远走得笔直 ⇒ 永远复现不出真机那个绕圈')
ap.add_argument('--walk-veer-sign', type=float, default=1.0,
                help='跑偏方向：+1 左偏（实测）/ −1 右偏')
# ---- 里程计缺陷注入 ----
ap.add_argument('--capo-scale', type=float, default=1.0,
                help='里程计尺度（0.94 = 少算 6%）')
ap.add_argument('--capo-latency', type=float, default=0.0, help='输出延迟 [s]')
ap.add_argument('--capo-noise', type=float, default=0.0, help='位置噪声 σ [m]')
ap.add_argument('--capo-yaw-noise', type=float, default=0.0, help='朝向噪声 σ [rad]')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--quiet', action='store_true')
a, _ = ap.parse_known_args()

rng = np.random.RandomState(a.seed)


def wrap(x):
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def quat_from_yaw(y):
    q = Quaternion()
    q.w = math.cos(y * 0.5)
    q.z = math.sin(y * 0.5)
    return q


class SimDog(Node):
    def __init__(self):
        super().__init__('sim_dog')
        qr = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)

        # ---- 真实状态 ----
        self.x = a.x0
        self.y = a.y0
        self.yaw = math.radians(a.yaw0)
        self.vx = 0.0          # 机体系
        self.vy = 0.0
        self.wz = 0.0

        # ---- 指令 ----
        self.cmd = (0.0, 0.0, 0.0)
        self.stopped = True     # 上电即静止，必须先收到 Move 才动
        self.last_cmd_t = time.time()
        self.n_move = 0
        self.n_stop = 0

        # ---- 里程计缺陷 ----
        self.origin = (a.x0, a.y0, math.radians(a.yaw0))
        self.hist = []          # 延迟用的环形缓冲 (t, x, y, yaw)

        self.pub_odom = self.create_publisher(Odometry, '/SMX/Odom', qr)
        self.pub_truth = self.create_publisher(Odometry, '/sim/truth', qr)
        self.create_subscription(Request, '/api/sport/request', self.on_req, qr)

        self.dt = 1.0 / a.rate
        self.t_prev = time.time()
        self.t0 = time.time()
        self.create_timer(self.dt, self.step)

        if not a.quiet:
            self.get_logger().info(
                'sim_dog: deadband vx<%.2f wz<%.2f | tau=%.2f | '
                'odom scale=%.3f latency=%.0fms noise=%.3f'
                % (a.vx_deadband, a.wz_deadband, a.tau, a.capo_scale,
                   a.capo_latency * 1000, a.capo_noise))

    # ------------------------------------------------------------------
    def on_req(self, m):
        try:
            api = int(m.header.identity.api_id)
            if api == API_MOVE:
                p = json.loads(m.parameter) if m.parameter else {}
                self.cmd = (float(p.get('x', 0.0)), float(p.get('y', 0.0)),
                            float(p.get('z', 0.0)))
                self.stopped = False
                self.last_cmd_t = time.time()
                self.n_move += 1
            elif api in (API_STOPMOVE, API_BALANCESTAND):
                self.n_stop += 1
                if api == API_STOPMOVE:
                    self.cmd = (0.0, 0.0, 0.0)
                    self.stopped = True
                    if a.settle_yaw:
                        # optional: the body swings a bit as the legs settle
                        self.yaw = wrap(self.yaw + math.radians(a.settle_yaw))
        except Exception as e:
            self.get_logger().warn('bad request: %s' % e)

    # ------------------------------------------------------------------
    def step(self):
        now = time.time()
        dt = now - self.t_prev
        self.t_prev = now

        vx_d, vy_d, wz_d = self.cmd
        if self.stopped:
            vx_d = vy_d = wz_d = 0.0

        # ---- 执行器死区：低于阈值【完全不动】（实测行为）----
        #     先判死区，再乘增益 —— 死区是判【指令】的（真机 0.19 的指令就不转）
        if abs(vx_d) < a.vx_deadband:
            vx_d = 0.0
        else:
            vx_d *= a.vx_gain
        if abs(wz_d) < a.wz_deadband:
            wz_d = 0.0
        else:
            wz_d *= a.wz_gain

        # ---- 一阶惯性（加速和减速共用同一个 τ）----
        tau = a.tau_stop if (self.stopped or vx_d == 0.0) else a.tau
        k = 1.0 - math.exp(-dt / max(tau, 1e-6))
        self.vx += (vx_d - self.vx) * k
        self.wz += (wz_d - self.wz) * k

        # ---- 积分（机体系 -> 世界系）----
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * c - self.vy * s) * dt
        self.y += (self.vx * s + self.vy * c) * dt

        # ---- ★ 走路时的自然跑偏：每走【1 米】转 walk_veer 度（实测约 3.6）
        #      只在前进时生效（转向/静止不叠加）
        veer = 0.0
        if a.walk_veer != 0.0 and abs(self.vx) > 1e-3:
            veer = (math.radians(a.walk_veer) * a.walk_veer_sign
                    * self.vx * dt)
        self.yaw = wrap(self.yaw + self.wz * dt + veer)

        # ---- 记录真值历史（供延迟注入）----
        self.hist.append((now, self.x, self.y, self.yaw))
        if len(self.hist) > int(a.capo_latency * a.rate) + 50:
            self.hist.pop(0)

        # ---- 真值 ----
        self.publish(self.pub_truth, self.x, self.y, self.yaw)

        # ---- 模拟 CAPO：延迟 + 尺度 + 噪声 ----
        if a.capo_latency > 0 and len(self.hist) > 2:
            want = now - a.capo_latency
            hx, hy, hyaw = self.hist[0][1], self.hist[0][2], self.hist[0][3]
            for e in self.hist:
                if e[0] >= want:
                    hx, hy, hyaw = e[1], e[2], e[3]
                    break
        else:
            hx, hy, hyaw = self.x, self.y, self.yaw

        ox, oy, oyaw = self.origin
        dx, dy = hx - ox, hy - oy
        # CAPO 是相对里程计：尺度误差作用在【相对起点的位移】上
        ox2 = ox + a.capo_scale * dx + rng.randn() * a.capo_noise
        oy2 = oy + a.capo_scale * dy + rng.randn() * a.capo_noise
        oyaw2 = wrap(oyaw + wrap(hyaw - oyaw) + rng.randn() * a.capo_yaw_noise)
        self.publish(self.pub_odom, ox2, oy2, oyaw2)

    # ------------------------------------------------------------------
    def publish(self, pub, x, y, yaw):
        m = Odometry()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.child_frame_id = 'base_link'
        m.pose.pose.position.x = float(x)
        m.pose.pose.position.y = float(y)
        m.pose.pose.position.z = 0.0
        m.pose.pose.orientation = quat_from_yaw(yaw)
        m.twist.twist.linear.x = float(self.vx)
        m.twist.twist.angular.z = float(self.wz)
        pub.publish(m)


def main():
    rclpy.init()
    n = SimDog()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
