# -*- coding: utf-8 -*-
"""
tests/test_p2_plan.py —— P2 规划层验收（纯逻辑，不需要 ROS）

验收项
  ① 空场地：直线走得通
  ② ★ U 形陷阱：不卡死（复刻 2026-09-21 的验收场景）
  ③ 带缺口的墙：从缺口穿过去
  ④ 路径全程在【膨胀后的自由区】里 —— 不蹭墙
  ⑤ 平滑有效：抽稀后路点数明显变少，且端点保持
  ⑥ 坐标系用错必须抛异常
  ⑦ 起点/终点落在障碍里 → 明确返回失败原因（不是崩）
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import Pose2D, FrameMismatch, FRAME_SCENE, FRAME_CAPO  # noqa: E402
from maps.grid import OccupancyGrid, make_test_world                    # noqa: E402
from planning.global_planner import plan_path, dijkstra_free, smooth_path  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  [OK]   %s' % name)
    except Exception as e:
        FAIL.append((name, e))
        print('  [FAIL] %s\n         %s: %s' % (name, type(e).__name__, e))


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError('期望抛 %s，没抛' % exc.__name__)


def S(x, y, yaw=0.0):
    return Pose2D(x, y, yaw, frame=FRAME_SCENE)


# ================================================================
def t_empty():
    g = make_test_world('empty')
    p = plan_path(g, S(2.0, 5.0), S(10.0, 5.0))
    assert p.ok, p.reason
    # 直线的话长度 ≈ 8m
    assert 7.5 < p.length < 9.0, 'length=%.2f' % p.length
    # 起终点必须精确
    assert abs(p.waypoints[0].x - 2.0) < 1e-6
    assert abs(p.waypoints[-1].x - 10.0) < 1e-6


def t_utrap():
    """★ U 形陷阱：从上方绕进去。纯贪心会一头扎进 U 里出不来。"""
    g = make_test_world('utrap')
    p = plan_path(g, S(2.0, 8.0), S(6.3, 4.0))     # 目标在 U 内部
    assert p.ok, p.reason
    gi = g.inflated(0.35)
    for w in p.waypoints:
        assert gi.is_free_xy(w.x, w.y), '路径点 (%.2f,%.2f) 在障碍里' % (w.x, w.y)
    print('         U 形：路径长 %.2f m，%d 个路点' % (p.length, len(p.waypoints)))


def t_corridor():
    g = make_test_world('corridor')
    p = plan_path(g, S(2.0, 5.0), S(11.0, 5.0))
    assert p.ok, p.reason
    gi = g.inflated(0.35)
    for w in p.waypoints:
        assert gi.is_free_xy(w.x, w.y), '(%.2f,%.2f) 在障碍里' % (w.x, w.y)
    # 必须从缺口过（缺口在 y≈6.2），所以路径的 y 应该抬高
    ys = [w.y for w in p.waypoints]
    assert max(ys) > 5.8, '没从缺口过？最大 y = %.2f' % max(ys)


def t_no_obstacle_hit():
    for kind, s, gl in (('utrap', S(2.0, 8.0), S(6.3, 4.0)),
                        ('corridor', S(2.0, 2.5), S(11.0, 8.0))):
        g = make_test_world(kind)
        p = plan_path(g, s, gl)
        assert p.ok, '%s: %s' % (kind, p.reason)
        gi = g.inflated(0.35)
        for w in p.waypoints:
            assert gi.is_free_xy(w.x, w.y), '%s: (%.2f,%.2f) 在障碍里' % (kind, w.x, w.y)


def t_smoothing_works():
    g = make_test_world('utrap')
    raw = plan_path(g, S(2.0, 8.0), S(6.3, 4.0), smooth=False)
    sm = plan_path(g, S(2.0, 8.0), S(6.3, 4.0), smooth=True)
    assert raw.ok and sm.ok
    print('         平滑前 %d 个路点 → 平滑后 %d 个' % (len(raw.waypoints),
                                                      len(sm.waypoints)))
    assert len(sm.waypoints) < len(raw.waypoints), '平滑没起作用'
    # 端点必须保持
    assert abs(sm.waypoints[0].x - raw.waypoints[0].x) < 1e-6
    assert abs(sm.waypoints[-1].x - raw.waypoints[-1].x) < 1e-6


def t_frame_guard():
    g = make_test_world('empty')
    expect_raises(FrameMismatch, lambda: plan_path(
        g, Pose2D(2, 5, 0, frame=FRAME_CAPO), S(10, 5)))
    expect_raises(FrameMismatch, lambda: plan_path(
        g, S(2, 5), Pose2D(10, 5, 0, frame=FRAME_CAPO)))


def t_blocked_returns_reason():
    g = make_test_world('utrap')
    # 起点在 U 的墙里
    p = plan_path(g, S(5.1, 4.0), S(6.3, 4.0))
    # 墙里的点会被 nearest_free 拉到自由格，所以应该还能规划出来
    assert p.ok or p.reason in ('start_blocked', 'no_path')
    # 完全封死的目标
    g2 = make_test_world('empty')
    g2.occ[:, :] = True
    p2 = plan_path(g2, S(2, 5), S(10, 5))
    assert not p2.ok and p2.reason in ('start_blocked', 'goal_blocked', 'no_path')


def t_inflate_respected():
    """膨胀半径变大 → 路径必须离墙更远"""
    g = make_test_world('corridor')
    p1 = plan_path(g, S(2.0, 5.0), S(11.0, 5.0), inflate_radius=0.15)
    p2 = plan_path(g, S(2.0, 5.0), S(11.0, 5.0), inflate_radius=0.55)
    assert p1.ok and p2.ok
    print('         膨胀 0.15m → 路径 %.2f m ; 膨胀 0.55m → %.2f m'
          % (p1.length, p2.length))
    assert p2.length >= p1.length - 0.01, '膨胀大了路径反而更短？'


def main():
    print('=' * 70)
    print('  P2 验收（规划层）')
    print('=' * 70)
    groups = [
        ('① 空场地', [('直线走得通', t_empty)]),
        ('② U 形陷阱', [('不卡死，且绕开障碍', t_utrap)]),
        ('③ 带缺口的墙', [('从缺口穿过去', t_corridor)]),
        ('④ 不蹭墙', [('路径全程在膨胀后的自由区', t_no_obstacle_hit)]),
        ('⑤ 路径平滑', [('抽稀有效且端点保持', t_smoothing_works)]),
        ('⑥ 坐标系防呆', [('用错坐标系抛异常', t_frame_guard)]),
        ('⑦ 边界情况', [('起点/终点被堵 → 明确失败原因', t_blocked_returns_reason),
                     ('膨胀半径生效', t_inflate_respected)]),
    ]
    for title, fns in groups:
        print()
        print('-- %s' % title)
        for name, fn in fns:
            check(name, fn)

    print()
    print('=' * 70)
    print('  通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for n, e in FAIL:
            print('    ✗ %s' % n)
        print('=' * 70)
        return 1
    print('  ✅ P2 规划层全部通过')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
