# -*- coding: utf-8 -*-
"""
demo/group2_action.py —— 扮演组2，用【正式 Action 接口】发目标

★ 这就是组2 那边要写的客户端的样板：
    · 发 goal（起点6D + 终点6D + 容差 + 超时 + 限速）
    · 收 feedback（10Hz，画进度条用）
    · 收 result（一次）
    · 可以中途 cancel()

用法（笔记本 WSL）:
    python3 demo/group2_action.py --tasks "2.0,0.0,0.0"
    python3 demo/group2_action.py --tasks "1.0,1.0,1.57;2.0,0.0,0.0"   # 多段，; 分隔
    python3 demo/group2_action.py --tasks "2.0,0.0,0.0" --cancel-after 5

⚠️ 格式（两种写法都行）:
       x,y,yaw                （3 个数 —— z/roll/pitch 自动填 0）
       x,y,z,roll,pitch,yaw   （6 个数 —— 完整 6D）
   ★★ 所有角度一律【弧度】（不是度），没有"度"的入口。
      要 90° 就写 1.5708；要 5° 就写 0.0873。

⚠️ 我们【现在只用到 x / y / yaw】，z / roll / pitch 会被忽略，
   但★建议填真实值（以后会用上时不用改客户端），不知道就填 0。

★★ 多段任务：--start 只给【第一段】的起点；
   之后每段的起点【自动 = 上一段的终点】（s_2 = g_1）。
   服务端就是靠这个来判断"你还认不认得狗在哪"的 —— 每段都发同一个 s_*
   会被判成"不一致"，从而触发一次【重对齐 + 报警】（那是误报）。
"""
import argparse
import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from nav_interfaces.action import NavigateTo

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', default='2.0,0.0,0.0',
                help='任务列表，分号分隔；每个是 "x,y,yaw" 或 '
                     '"x,y,z,roll,pitch,yaw"（角度都是弧度）')
ap.add_argument('--pos-tol', type=float, default=0.05, help='[m]')
ap.add_argument('--yaw-tol', type=float, default=0.03, help='[rad]')
ap.add_argument('--timeout', type=float, default=120.0)
ap.add_argument('--max-speed', type=float, default=0.0, help='0=不限')
ap.add_argument('--gap', type=float, default=3.0, help='任务之间等多久 [s]')
ap.add_argument('--no-chain', action='store_true',
                help='★ 测试用：每段都用同一个 --start，不做 s_2=g_1 的链式推导。'
                     '用来验证服务端的"起点不一致 ⇒ 重对齐+报警"那条路')
ap.add_argument('--cancel-after', type=float, default=0.0,
                help='>0 则在 N 秒后取消当前任务（测 cancel 用）')
ap.add_argument('--start', nargs='+', type=float,
                default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                help='★【第一段】的起点 6D: x y z roll pitch yaw（角度都是【弧度】）；'
                     '也可以只给 3 个 x y yaw。'
                     '★ 之后每段的起点自动 = 上一段的终点（s_2 = g_1）')
a, _ = ap.parse_known_args()


class C(Node):
    def __init__(self):
        super().__init__('group2_action_client')
        self.cli = ActionClient(self, NavigateTo, '/nav/navigate_to')
        self.n_fb = 0
        self.t0 = 0.0
        self._goal_handle = None
        self._cancel_at = a.cancel_after

    # ---- feedback ----
    def on_feedback(self, msg):
        self.n_fb += 1
        fb = msg.feedback
        if self.n_fb % 10 == 1:      # 每秒打一行，别刷屏
            print('    [fb %3d] stage=%-14s 剩 %.2fm / %.1f°'
                  % (self.n_fb, fb.stage, fb.remaining_dist,
                     math.degrees(fb.remaining_yaw)), flush=True)
        if self._cancel_at > 0 and time.time() - self.t0 > self._cancel_at:
            self._cancel_at = 0.0
            print('    ★ 到了 %0.1fs，主动 cancel' % a.cancel_after)
            try:
                self._goal_handle.cancel_goal_async()
            except Exception as e:
                print('    cancel 失败: %s' % e)

    def run_one(self, label, gx, gy, gz, gr, gp, gyaw, start):
        print('── %s ──' % label)
        print('  → 发出目标  s=(%.2f, %.2f, %.3f rad)  g=(%.2f, %.2f, %.3f rad)'
              % (start[0], start[1], start[5], gx, gy, gyaw), flush=True)
        goal = NavigateTo.Goal()
        # ★★ 2026-09-27 修：这里原来对 --start[2] 做了 math.radians()
        #    ⇒ 【同一个文件里两种单位】：--start 收度、--tasks 收弧度。
        #    这个文件是发给组2 当模板的，正是"把度数当弧度"那个坑的温床。
        #    现在统一成【弧度】，和 .action 定义、和全部文档一致。
        # ★ 6 个字段【全填】—— 现在只用 x/y/yaw，但建议填真实值（见文件头）
        # ★★ s_* 是【这一段的起点】：第一段由 --start 给，之后自动 = 上一段的终点
        #    （见 main()）—— 这正是组2 应该有的行为：s_2 应该等于 g_1。
        (goal.s_x, goal.s_y, goal.s_z,
         goal.s_roll, goal.s_pitch, goal.s_yaw) = start
        (goal.g_x, goal.g_y, goal.g_z,
         goal.g_roll, goal.g_pitch, goal.g_yaw) = gx, gy, gz, gr, gp, gyaw
        goal.pos_tol = a.pos_tol
        goal.yaw_tol = a.yaw_tol
        goal.timeout = a.timeout
        goal.max_speed = a.max_speed

        if not self.cli.wait_for_server(timeout_sec=15.0):
            print('  ✗ 等不到 Action 服务端（导航节点起了吗？）')
            return False

        self.n_fb = 0
        self.t0 = time.time()
        send = self.cli.send_goal_async(goal,
                                        feedback_callback=self.on_feedback)
        # ★★ 不能用 rclpy.spin_until_future_complete ——
        #    节点已经被后台的 MultiThreadedExecutor spin 着，两个 executor 抢同一个
        #    节点会【死锁】。（2026-09-26 踩过：客户端卡在"发出目标"不动）
        #    ⇒ 手动等 future 完成（后台 executor 会把它推进）
        t_wait = time.time()
        while not send.done() and time.time() - t_wait < 10.0:
            time.sleep(0.05)
        if not send.done():
            print('  ✗ 10 秒没等到服务端响应')
            return False
        gh = send.result()
        if gh is None or not gh.accepted:
            print('  ✗ 目标被拒绝')
            return False
        self._goal_handle = gh
        res_fut = gh.get_result_async()
        while not res_fut.done():
            time.sleep(0.1)
        r = res_fut.result().result
        print('  ← result: success=%s reason=%s 用时 %.1fs  internal_error=%.3f '
              'uncertainty=%.3f' % (r.success, r.reason, r.elapsed,
                                    r.internal_error, r.uncertainty))
        print('     期间收到 feedback %d 条' % self.n_fb, flush=True)
        return bool(r.success)


def main():
    tasks = []
    for part in a.tasks.split(';'):
        part = part.strip()
        if not part:
            continue
        v = [float(x) for x in part.split(',')]
        if len(v) == 3:
            v = [v[0], v[1], 0.0, 0.0, 0.0, v[2]]      # 补上 z/roll/pitch
        elif len(v) != 6:
            sys.exit('任务格式错: %r（要 "x,y,yaw" 或 "x,y,z,roll,pitch,yaw"）'
                     % part)
        tasks.append(v)

    rclpy.init()
    n = C()
    ex = MultiThreadedExecutor()
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    print('=' * 70)
    print('  扮演组2（正式 Action 接口）  共 %d 个任务' % len(tasks))
    print('  pos_tol %.3f m  yaw_tol %.3f rad (%.1f°)  timeout %.0fs'
          % (a.pos_tol, a.yaw_tol, math.degrees(a.yaw_tol), a.timeout))
    print('=' * 70)

    ok_all = True
    # ★★ 每段的起点：第一段用 --start 给的，之后【自动 = 上一段的终点】
    #    这正是组2 该有的行为（s_2 = g_1）。
    #    ★ 如果每段都发同一个 s_*，第二段就会和我们的认知差一整段距离
    #      ⇒ 服务端会判"不一致"并【重对齐 + 报警】—— 那是误报，不是 bug。
    cur_start = (list(a.start) + [0.0, 0.0, 0.0])[:6]
    for i, t in enumerate(tasks, 1):
        gx, gy, gz, gr, gp, gyaw = t
        ok = n.run_one('任务 %d/%d' % (i, len(tasks)),
                       gx, gy, gz, gr, gp, gyaw, cur_start)
        ok_all = ok_all and ok
        if not a.no_chain:
            cur_start = [gx, gy, gz, gr, gp, gyaw]  # ★ 下一段从这一段终点接着走
        if i < len(tasks):
            time.sleep(a.gap)

    print('=' * 70)
    print('  全部任务完成' if ok_all else '  有任务未成功')
    print('=' * 70)
    try:
        ex.shutdown()
    except Exception:
        pass
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
