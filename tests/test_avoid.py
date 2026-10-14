# -*- coding: utf-8 -*-
"""动态避障阶段 1 验收：纯 Python，不依赖 ROS，不连接真机。"""
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.path_follower import FollowerConfig, PathFollower  # noqa: E402
from core.types import Pose2D, Timed, Velocity, FRAME_SCENE      # noqa: E402
from mission.api import NavigateToConfig                         # noqa: E402
from planning.global_planner import Plan                         # noqa: E402
from planning.local_avoid import AvoidConfig                     # noqa: E402


PASS, FAIL = [], []


def S(x, y, yaw=0.0):
    return Pose2D(x, y, yaw, frame=FRAME_SCENE)


class FakeObstacleSource:
    def __init__(self):
        self._pts = np.empty((0, 3), dtype=float)
        self._lock = threading.Lock()
        self._recv_t = time.time()

    def set(self, pts):
        with self._lock:
            self._pts = np.asarray(pts, dtype=float)
            self._recv_t = time.time()

    def get(self):
        with self._lock:
            return self._pts.copy()

    def healthy(self, max_age=1.0):
        return True

    def error_count(self):
        return 0


class SimRig:
    """同时充当位姿源和指令出口，只在内存中积分速度。"""
    name = 'avoid-sim'

    def __init__(self, dt=0.01):
        self.dt = dt
        self.pose = S(0.0, 0.0)
        self.cmd = Velocity()
        self.hist = []
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
            self.hist.append((time.time(), v))

    def stop(self):
        self.move(Velocity())

    def close(self):
        self._stop.set()

    def x(self):
        with self._lock:
            return self.pose.x

    def recent_zero(self, since):
        with self._lock:
            return any(t >= since and v == Velocity() for t, v in self.hist)


def wait_until(pred, timeout, message):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def run_case(enabled, obstacle_y, inject_midway=False):
    rig = SimRig()
    obstacles = FakeObstacleSource()
    goal = S(2.0, 0.0)
    wps = [S(0.0, 0.0), S(0.5, 0.0), S(1.0, 0.0),
           S(1.5, 0.0), goal]
    plan = Plan(True, wps, 2.0)
    cfg = FollowerConfig(v_cruise=0.8, settle_s=0.02, control_hz=100.0,
                         fine_range=0.12, timeout=7.0, no_progress_s=1.0)
    avoid = AvoidConfig(enabled=enabled, lookahead_m=1.0, corridor_m=0.4,
                        min_points=2, hold_s=0.20)
    follower = PathFollower(cfg, rig, rig, avoid=avoid,
                            obstacle_source=obstacles)
    box = {}

    def work():
        box['result'] = follower.follow(plan, goal)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        if inject_midway:
            wait_until(lambda: rig.x() >= 0.60, 3.0, '仿真狗没走到注入位置')
        x0 = rig.x()
        obstacles.set([[x0 + 0.35, obstacle_y, 0.0],
                       [x0 + 0.40, obstacle_y + 0.02, 0.0],
                       [x0 + 0.45, obstacle_y - 0.02, 0.0]])
        injected_at = time.time()
        return rig, obstacles, box, th, injected_at
    except Exception:
        rig.close()
        raise


def t_stops_and_waits():
    rig, obstacles, box, th, injected_at = run_case(True, 0.0, True)
    try:
        wait_until(lambda: rig.recent_zero(injected_at), 1.0,
                   '注入走廊内点簇后没有收到零速')
        assert th.is_alive(), '停车后 follow() 立刻结束了，应该原地等待'
        x_stopped = rig.x()
        time.sleep(0.25)
        assert abs(rig.x() - x_stopped) < 0.03, '停车等待期间仍在前进'
        obstacles.set(np.empty((0, 3)))
        th.join(5.0)
        assert not th.is_alive(), '障碍移开后没有恢复并完成跟随'
        assert box['result'].success and box['result'].reason == 'handed_off'
    finally:
        rig.close()


def t_disabled_ignores_points():
    nav_cfg = NavigateToConfig()
    assert nav_cfg.dynamic_avoid is False, 'NavigateToConfig 默认必须关闭动态避障'
    assert nav_cfg.avoid.enabled is False, 'AvoidConfig 默认必须关闭动态避障'
    rig, obstacles, box, th, _ = run_case(nav_cfg.dynamic_avoid, 0.0)
    try:
        th.join(5.0)
        assert not th.is_alive(), '默认关闭时没有照常走完'
        assert box['result'].success and box['result'].reason == 'handed_off'
    finally:
        rig.close()


def t_outside_corridor_does_not_stop():
    rig, obstacles, box, th, _ = run_case(True, 1.0)
    try:
        th.join(5.0)
        assert not th.is_alive(), '走廊外点导致任务没有走完'
        assert box['result'].success and box['result'].reason == 'handed_off'
    finally:
        rig.close()


def t_static_at_inflate_radius_does_not_stop():
    """★★ 2026-09-28：贴着路径的【静态】障碍不能触发停车。

    为什么（这条是踩出来的）：规划按 inflate_radius=0.35 膨胀 ⇒ 路径上任意点离
    最近【静态】障碍 ≥0.35m，而且路径常常【贴着这个边界走】。
    如果 corridor_m ≥ 0.35，那些静态点就会被当成"新出现的东西" ⇒ 一出发就误停。

    实测（tools/check_corridor_vs_inflate.py）：墙离路径 0.35m、20 个点，
        corridor_m=0.40 ⇒ 命中 20 点 ⇒ 停   ✗
        corridor_m=0.25 ⇒ 命中  0 点 ⇒ 不停 ✓
    """
    from planning.local_avoid import corridor_polyline, should_stop
    assert AvoidConfig().corridor_m < 0.35, (
        '★ 默认 corridor_m 必须【小于】规划膨胀半径 0.35 —— 否则贴着路径的'
        '静态障碍会被误判成动态障碍（实测 0.40 会全误停）')

    pose, goal = S(0.0, 0.0), S(2.0, 0.0)
    wps = [S(0.0, 0.0), S(0.5, 0.0), S(1.0, 0.0), S(1.5, 0.0), goal]
    poly = corridor_polyline(wps, pose, 1, goal, 2.0)
    for d in (0.30, 0.35, 0.36):
        wall = np.stack([np.linspace(0.3, 1.8, 20),
                         np.full(20, d), np.zeros(20)], axis=1)
        stop, cnt = should_stop(wall, poly, AvoidConfig(enabled=True))
        assert not stop, (
            '离路径 %.2fm（≈规划膨胀）的静态障碍不该触发停车，实际命中 %d 点'
            % (d, cnt))


def t_avoid_too_long_reports_blocked():
    """★★ 2026-10-08 真机加的：避障【一直不清】时必须报 blocked，不能耗到总超时。

    为什么（真机踩出来的）：
      等待期间会【重置】无进展窗口 —— 这是【故意的】（让"停着等人走过"不被误判）。
      但后果是：障碍一直不清时会一路等到【总超时 90s】⇒ 报 timeout。
      而组2 的 SPAV 【只在 reason=='blocked' 时】绕行，拿到 timeout 会走
      "安全熔断"直接终止整个任务。
      2026-10-08 真机实测：狗停下来等了 90s ⇒ 报 timeout ⇒ ★ 组2 的自愈没触发。

    修法：等够 avoid_blocked_s 还堵着 ⇒ 报 blocked，交给上层绕行。
    """
    rig = SimRig()
    obstacles = FakeObstacleSource()
    goal = S(2.0, 0.0)
    wps = [S(0.0, 0.0), S(0.5, 0.0), S(1.0, 0.0), S(1.5, 0.0), goal]
    plan = Plan(True, wps, 2.0)
    cfg = FollowerConfig(v_cruise=0.8, settle_s=0.02, control_hz=100.0,
                         fine_range=0.12, timeout=30.0, no_progress_s=1.0,
                         avoid_blocked_s=0.8)      # ★ 测试里缩到 0.8s
    avoid = AvoidConfig(enabled=True, lookahead_m=1.0, corridor_m=0.4,
                        min_points=2, hold_s=0.20)
    follower = PathFollower(cfg, rig, rig, avoid=avoid,
                            obstacle_source=obstacles)
    box = {}

    def work():
        box['result'] = follower.follow(plan, goal)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        wait_until(lambda: rig.x() >= 0.60, 3.0, '仿真狗没走到注入位置')
        x0 = rig.x()
        # ★ 注入一个【永远不清】的障碍
        obstacles.set([[x0 + 0.35, 0.0, 0.0], [x0 + 0.40, 0.02, 0.0]])
        wait_until(lambda: rig.recent_zero(time.time() - 0.5), 2.0,
                   '注入后没有收到零速')
        th.join(8.0)
        assert not th.is_alive(), '障碍一直不清时 follow() 没有自己退出'
        r = box['result']
        assert r.reason == 'blocked', (
            '★ 避障一直不清时必须报 blocked（组2 只在 blocked 时绕行），实际 %r'
            % r.reason)
        assert not r.success
    finally:
        rig.close()


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  [OK]   %s' % name)
    except Exception as exc:
        FAIL.append((name, exc))
        print('  [FAIL] %s\n         %s: %s'
              % (name, type(exc).__name__, exc))


if __name__ == '__main__':
    print('动态避障阶段 1：纯离线验收')
    check('① 路径前方注入点簇 → 零速等待，障碍移开后继续', t_stops_and_waits)
    check('② 默认关闭 → 注入点簇也照常走完', t_disabled_ignores_points)
    check('③ 点在走廊外 → 不停车并照常走完', t_outside_corridor_does_not_stop)
    check('④ 贴路径的静态障碍（≈规划膨胀处）不误触发',
          t_static_at_inflate_radius_does_not_stop)
    check('⑤ ★ 障碍一直不清 → 等够 avoid_blocked_s 后报 blocked（不是 timeout）',
          t_avoid_too_long_reports_blocked)
    print('\n结果：%d/%d 通过' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        raise SystemExit(1)
