# -*- coding: utf-8 -*-
"""
tests/test_handoff.py —— 交接条件验收（纯 Python，不依赖 ROS）

背景
----
末端精调（fine_approach）是【直着朝目标走】、**不沿路径**。
⇒ 交接点之后若路径还有拐角，直走会把拐角【切掉】（可能蹭到）。
⇒ ★ 2026-10-15 改：交接 = 距终点 ≤ fine_range【且】剩余路径【无转角】
   （无转角 = 当前目标路点已推进到终点本身）。
   这时剩下的路本来就是一条直线，直走不吃亏。

验收项
  ① 终点前还有拐角（拐点 < 0.2m）⇒ 【不】在拐弯前交接（要拐过弯才交）
  ② 直线路径 ⇒ 正常在 fine_range 附近交接（别把交接卡死）
"""
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.path_follower import FollowerConfig, PathFollower  # noqa: E402
from core.types import Pose2D, Timed, Velocity, FRAME_SCENE     # noqa: E402
from planning.global_planner import Plan                        # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  [OK]   %s' % name)
    except Exception as e:
        FAIL.append((name, e))
        print('  [FAIL] %s\n         %s: %s' % (name, type(e).__name__, e))


def S(x, y, yaw=0.0):
    return Pose2D(x, y, yaw, frame=FRAME_SCENE)


class SimRig:
    """同时充当位姿源和指令出口，只在内存中积分速度（同 test_avoid 的做法）。"""

    name = 'handoff-sim'

    def __init__(self, dt=0.01):
        self.dt = dt
        self.pose = S(0.0, 0.0)
        self.cmd = Velocity()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.dt):
            with self._lock:
                c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
                self.pose = S(self.pose.x + self.cmd.vx * c * self.dt,
                              self.pose.y + self.cmd.vx * s * self.dt,
                              self.pose.yaw + self.cmd.wz * self.dt)

    def get(self):
        with self._lock:
            return Timed(time.time(), None, self.pose)

    def error_count(self):
        return 0

    def move(self, v):
        with self._lock:
            self.cmd = v

    def stop(self):
        self.move(Velocity())

    def close(self):
        self._stop.set()


def run_follow(start, wps, goal, fine_range=0.20):
    rig = SimRig()
    rig.pose = S(*start)
    cfg = FollowerConfig(v_cruise=0.6, settle_s=0.02, control_hz=100.0,
                         fine_range=fine_range, timeout=20.0, no_progress_s=2.0)
    fol = PathFollower(cfg, rig, rig)
    r = fol.follow(Plan(True, list(wps), 1.0), goal)
    rig.close()
    return r


# ================================================================
def t_no_handoff_before_bend():
    """① 终点前 0.16m 处有拐角 ⇒ 交接必须发生在【拐过去之后】

    路径：沿 y=-0.15 走到 (1.95,-0.15)，再拐 71° 向上到终点 (2.0,0)。
    拐点离终点 hypot(0.05,0.15)=0.158 < 0.2 ⇒
      旧条件（只看距离）会在 x≈1.88 就交接（把拐角切掉）；
      新条件要求"拐过弯"⇒ 交接时 x 必须 > 1.93。
    """
    goal = S(2.0, 0.0)
    wps = [S(1.0, -0.15), S(1.95, -0.15), goal]
    r = run_follow((1.0, -0.15), wps, goal)
    assert r.reason == 'handed_off', 'reason=%s' % r.reason
    assert r.final_pose.x > 1.93, \
        '在拐角前就交接了：x=%.3f（应该拐过 (1.95,-0.15) 再交）' % r.final_pose.x


def t_straight_hands_off_normally():
    """② 直线路径 ⇒ 正常交接（在 fine_range 附近，不会被卡住）"""
    goal = S(2.0, 0.0)
    wps = [S(1.0, 0.0), S(1.5, 0.0), goal]
    r = run_follow((1.0, 0.0), wps, goal)
    assert r.reason == 'handed_off', 'reason=%s' % r.reason
    d = math.hypot(goal.x - r.final_pose.x, goal.y - r.final_pose.y)
    assert 0.15 < d <= 0.21, '交接点离终点 %.3f m（应在 ~0.2）' % d


# ================================================================
def main():
    print('=' * 74)
    print('  交接条件验收（无转角 + fine_range）')
    print('=' * 74)
    check('① 终点前有拐角 ⇒ 拐过去之后才交接', t_no_handoff_before_bend)
    check('② 直线路径 ⇒ 正常在 fine_range 附近交接', t_straight_hands_off_normally)
    print('-' * 74)
    print('  通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for name, e in FAIL:
            print('  ❌ %s: %s' % (name, e))
        return 1
    print('  ✅ 交接条件全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
