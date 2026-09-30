# -*- coding: utf-8 -*-
"""
demo/group2_sim.py —— 【扮演组2】发目标、收 feedback/result

这个脚本不是我们的导航逻辑 —— 它模拟【组2 的客户端】：
    ① 往 /nav/navigate_to 发一个 goal（JSON，字段和 .action 的 goal 段一致）
    ② 一路收 /nav/feedback
    ③ 最后收 /nav/result

⇒ 完整走一遍对外接口，验证"别人这么用能不能用"。

用法: bash run_bridge_demo.sh
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', default='2,4,1.57;5,2,-1.57',
                help='分号分隔的 "x,y,yaw" 列表')
ap.add_argument('--pos-tol', type=float, default=0.05)
ap.add_argument('--yaw-tol', type=float, default=0.15)
ap.add_argument('--gap', type=float, default=3.0, help='任务之间等多久 [s]')
a, _ = ap.parse_known_args()

TASKS = []
for s in a.tasks.split(';'):
    x, y, yaw = [float(v) for v in s.split(',')]
    TASKS.append((x, y, yaw))


class Group2(Node):
    """★ 我们（组2）这边的代码，用得就是 .action 客户端的等价形式"""

    def __init__(self):
        super().__init__('group2_client')
        qr = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(String, '/nav/navigate_to', qr)
        self.create_subscription(String, '/nav/feedback', self._fb, qr)
        self.create_subscription(String, '/nav/result', self._res, qr)
        self.result = None
        self.n_fb = 0
        self.last_fb = None

    def _fb(self, m):
        try:
            d = json.loads(m.data)
            self.n_fb += 1
            self.last_fb = d
        except Exception:
            pass

    def _res(self, m):
        try:
            self.result = json.loads(m.data)
        except Exception:
            pass

    def send(self, gx, gy, gyaw):
        """★ 这就是"给我们的目标"—— 起点用开机对齐的那个，写死 (0,0,0)"""
        d = {
            's_x': 0.0, 's_y': 0.0, 's_z': 0.0,
            's_roll': 0.0, 's_pitch': 0.0, 's_yaw': 0.0,
            'g_x': float(gx), 'g_y': float(gy), 'g_z': 0.0,
            'g_roll': 0.0, 'g_pitch': 0.0, 'g_yaw': float(gyaw),
            'pos_tol': float(a.pos_tol), 'yaw_tol': float(a.yaw_tol),
            'timeout': 90.0, 'max_speed': 0.0,
        }
        m = String()
        m.data = json.dumps(d)
        self.pub.publish(m)
        self.result = None
        self.n_fb = 0
        print('  → 发出目标 (%.2f, %.2f, %.2f rad)' % (gx, gy, gyaw))


def main():
    rclpy.init()
    n = Group2()
    print('=' * 70)
    print('  扮演组2：走一遍对外接口')
    print('=' * 70)
    t0 = time.time()
    while time.time() - t0 < 3:
        rclpy.spin_once(n, timeout_sec=0.1)

    rc = 0
    for i, (gx, gy, gyaw) in enumerate(TASKS, 1):
        print()
        print('── 任务 %d/%d ──' % (i, len(TASKS)))
        n.send(gx, gy, gyaw)
        t = time.time()
        while time.time() - t < 180:
            rclpy.spin_once(n, timeout_sec=0.05)
            if n.result is not None:
                break
        if n.result is None:
            print('  ✗ 120 秒没收到 result')
            rc = 1
        else:
            r = n.result
            print('  ← result: success=%s reason=%s 用时 %.1fs '
                  'internal_error=%.3f uncertainty=%.3f'
                  % (r['success'], r['reason'], r['elapsed'],
                     r['internal_error'], r['uncertainty']))
            print('     期间收到 feedback %d 条（%.1f Hz）'
                  % (n.n_fb, n.n_fb / max(r['elapsed'], 1e-6)))
            if not r['success']:
                rc = 1
        if i < len(TASKS):
            time.sleep(a.gap)

    print()
    print('=' * 70)
    print('  全部任务完成' if rc == 0 else '  有任务未成功')
    print('=' * 70)
    import os
    os._exit(rc)


if __name__ == '__main__':
    main()
