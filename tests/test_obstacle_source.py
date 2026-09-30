# -*- coding: utf-8 -*-
"""动态避障阶段 2 验收：话题解析、坐标变换和失效安全。"""
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.path_follower import FollowerConfig, PathFollower  # noqa: E402
from core.types import Transform2D, FRAME_CAPO, FRAME_SCENE      # noqa: E402
from hal.obstacle_source import MockObstacleSource, RosObstacleSource  # noqa: E402
from mission.api import NavigateToConfig, _SceneObstacleSource  # noqa: E402
from planning.global_planner import Plan                         # noqa: E402
from planning.local_avoid import AvoidConfig                     # noqa: E402
from tests.test_avoid import S, SimRig                            # noqa: E402


PASS, FAIL = [], []


class FakeArrayMessage:
    def __init__(self, data):
        self.data = data


def make_follow_case(points, enabled):
    rig = SimRig()
    obstacles = MockObstacleSource()
    obstacles.set(points)
    goal = S(2.0, 0.0)
    wps = [S(0.0, 0.0), S(0.5, 0.0), S(1.0, 0.0),
           S(1.5, 0.0), goal]
    plan = Plan(True, wps, 2.0)
    cfg = FollowerConfig(v_cruise=0.8, settle_s=0.02, control_hz=100.0,
                         fine_range=0.12, timeout=7.0, no_progress_s=1.0)
    avoid = AvoidConfig(enabled=enabled, lookahead_m=1.0,
                        min_points=2, hold_s=0.20)
    abort = threading.Event()
    logs, box = [], {}
    follower = PathFollower(cfg, rig, rig, avoid=avoid,
                            obstacle_source=obstacles,
                            should_abort=abort.is_set,
                            log=logs.append)

    def work():
        box['result'] = follower.follow(plan, goal)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    return rig, obstacles, abort, logs, box, th


def t_topic_to_scene():
    original = np.array([[1.0, 0.0, 0.2],
                         [-2.0, 3.0, 0.4]], dtype=np.float32)
    msg = FakeArrayMessage(original.reshape(-1).tolist())
    parsed = RosObstacleSource.parse_message(msg)
    assert parsed.shape == (2, 3)
    assert parsed.dtype == np.float32
    assert np.allclose(parsed, original)

    inner = MockObstacleSource(parsed)
    T = Transform2D(FRAME_CAPO, FRAME_SCENE,
                    x=0.5, y=0.5, yaw=math.pi / 2)
    scene = _SceneObstacleSource(inner, T).get()
    expected = np.array([[0.5, 1.5, 0.2],
                         [-2.5, -1.5, 0.4]])
    assert np.allclose(scene, expected, atol=1e-6), (scene, expected)
    inner.set(None)
    assert _SceneObstacleSource(inner, T).get() is None


def t_none_means_stop():
    rig, _, abort, logs, box, th = make_follow_case(None, True)
    try:
        time.sleep(0.30)
        assert th.is_alive(), '感知失效时 follow() 应保持等待，不能立即退出'
        assert rig.x() < 0.03, '感知失效时仍然前进了 %.3fm' % rig.x()
        assert any('SENSOR FAILURE' in line for line in logs), logs
        abort.set()
        th.join(2.0)
        assert not th.is_alive(), '测试结束时跟随线程没有响应 abort'
    finally:
        abort.set()
        rig.close()


def t_empty_means_clear():
    rig, _, abort, _, box, th = make_follow_case(
        np.empty((0, 3), dtype=np.float32), True)
    try:
        th.join(5.0)
        assert not th.is_alive(), '空数组被误判为失效/障碍，任务没有走完'
        assert box['result'].success and box['result'].reason == 'handed_off'
    finally:
        abort.set()
        rig.close()


def t_disabled_ignores_points_and_none():
    assert NavigateToConfig().dynamic_avoid is False
    cluster = np.array([[0.3, 0.0, 0.0],
                        [0.4, 0.02, 0.0],
                        [0.5, -0.02, 0.0]], dtype=np.float32)
    for points in (cluster, None):
        rig, _, abort, _, box, th = make_follow_case(points, False)
        try:
            th.join(5.0)
            assert not th.is_alive(), '默认关闭时任务没有照常走完'
            assert box['result'].success and box['result'].reason == 'handed_off'
        finally:
            abort.set()
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
    print('动态避障阶段 2：纯离线验收')
    check('① Float32MultiArray 解析 + CAPO→scene 变换', t_topic_to_scene)
    check('② get()=None（感知失效）→ 零速等待', t_none_means_stop)
    check('③ get()=空数组（确认无障碍）→ 照常走完', t_empty_means_clear)
    check('④ 默认关闭 → 点簇和 None 都不影响既有行为',
          t_disabled_ignores_points_and_none)
    print('\n结果：%d/%d 通过' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        raise SystemExit(1)
