# -*- coding: utf-8 -*-
"""
tests/test_final_approach.py —— 预瞄点（turn_shift 根治）验收（纯逻辑，不需要 ROS）

背景
----
走到终点后，末端要"原地转身对准目标朝向"—— 转身会把身体挪偏 ~10cm
（转 14.5° 挪 3.6cm；转 134° 挪 10cm）⇒ 位置本来达标，转完就超差（turn_shift）。

★ 治法：让【最后一段路的方向 = 目标朝向】——
在终点【背后】d 米放一个【预瞄点 pre】，规划到 pre、末尾补上 goal；
狗走到终点时已经朝对，不用"到了再大转"。

验收项
  ① 关（默认）：final_approach_d == 0，规划行为不变
  ② 开：最后一段的方向 ≈ 目标朝向（关着时是另一回事 —— 对比）
  ③ 终点背后被占（pre 不可行）⇒ 自动退回普通规划，plan 仍然 ok
  ④ 阶梯：0.8 不行、0.6 可行 ⇒ 自动用 0.6
  ⑤ 起终点太近（< d）⇒ 不折腾
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
from core.types import Pose2D, FRAME_SCENE, wrap                       # noqa: E402
from maps.grid import OccupancyGrid                                    # noqa: E402
from planning.global_planner import plan_path                          # noqa: E402

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


def make_world(rects=(), W=12.0, H=10.0, res=0.05):
    """手写世界：外围墙 + 任意矩形障碍（和 maps/grid.make_test_world 同风格）"""
    occ = np.zeros((int(H / res), int(W / res)), dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    for (x0, y0, x1, y1) in rects:
        occ[int(y0 / res):int(y1 / res), int(x0 / res):int(x1 / res)] = True
    return OccupancyGrid(occ, (0.0, 0.0), res)


def last_leg_dir(p):
    a, b = p.waypoints[-2], p.waypoints[-1]
    return math.atan2(b.y - a.y, b.x - a.x)


# ================================================================
def t_off_by_default():
    """① 关（默认）：final_approach_d == 0，行为不变"""
    g = make_world()
    p = plan_path(g, S(2.0, 5.0), S(10.0, 5.0, math.pi / 2))
    assert p.ok, p.reason
    assert p.final_approach_d == 0.0, p.final_approach_d
    # 关着时，最后一段是"approach 方向"（≈ +x），不是目标朝向（π/2）
    d = abs(math.degrees(wrap(last_leg_dir(p) - math.pi / 2)))
    assert d > 45.0, '关着时最后一段竟然对上了目标朝向？dir误差 %.1f°' % d


def t_on_makes_last_leg_face_goal_yaw():
    """② 开：最后一段的方向 ≈ 目标朝向"""
    g = make_world()
    goal = S(10.0, 5.0, math.pi / 2)          # 终点朝 +y
    p = plan_path(g, S(2.0, 5.0), goal, final_approach_lengths=(0.8, 0.6))
    assert p.ok, p.reason
    assert p.final_approach_d == 0.8, p.final_approach_d
    # 最后一段方向 ≈ 目标朝向
    d = abs(math.degrees(wrap(last_leg_dir(p) - goal.yaw)))
    assert d < 2.0, '最后一段方向误差 %.1f°（应 <2°）' % d
    # 终点仍精确
    assert abs(p.waypoints[-1].x - 10.0) < 1e-6
    assert abs(p.waypoints[-1].y - 5.0) < 1e-6
    # 预瞄点精确落在 (10, 4.2)（沿目标朝向反方向退 0.8）
    near = min(hypot_(w.x - 10.0, w.y - 4.2) for w in p.waypoints)
    assert near < 0.06, '预瞄点没落在 (10,4.2) 附近：最近 %.3f m' % near


def hypot_(a, b):
    return math.hypot(a, b)


def t_pre_blocked_falls_back():
    """③ 终点背后是墙（0.8/0.6 都被占）⇒ 退回普通规划，plan 仍 ok"""
    # 墙：x∈[9.5,10.6]、y∈[3,7]（膨胀后 x≈[9.15,10.95]）
    # 终点 (11.4,5) 朝 +x ⇒ pre 在 (10.6,5)/(10.8,5) —— 都在膨胀后的墙里
    g = make_world([(9.5, 3.0, 10.6, 7.0)])
    p = plan_path(g, S(2.0, 5.0), S(11.4, 5.0, 0.0), final_approach_lengths=(0.8, 0.6))
    assert p.ok, p.reason
    assert p.final_approach_d == 0.0, 'pre 明明不可行却用了 d=%.2f' % p.final_approach_d
    assert abs(p.waypoints[-1].x - 11.4) < 1e-6      # 终点还是终点


def t_ladder_picks_shorter():
    """④ 阶梯：1.2 的 pre 被占、0.8 可行 ⇒ 自动用 0.8"""
    # 小箱子在 (9.3,5)：膨胀后 x∈[8.95,9.65]
    # 终点 (8.0,5) 朝 -x（π）⇒ pre@1.2=(9.2,5) 被占；pre@0.8=(8.8,5) 干净
    g = make_world([(9.25, 4.95, 9.35, 5.05)])
    p = plan_path(g, S(2.0, 5.0), S(8.0, 5.0, math.pi),
                  final_approach_lengths=(1.2, 0.8))
    assert p.ok, p.reason
    assert abs(p.final_approach_d - 0.8) < 1e-9, \
        '应该退到 0.8，实际 %.2f' % p.final_approach_d
    d = abs(math.degrees(wrap(last_leg_dir(p) - math.pi)))
    assert d < 2.0, '最后一段方向误差 %.1f°' % d


def t_too_close_skips():
    """⑤ 起终点太近（< d）⇒ 不折腾"""
    g = make_world()
    p = plan_path(g, S(9.0, 5.0), S(9.6, 5.0, 0.0), final_approach_lengths=(0.8,))
    assert p.ok, p.reason
    assert p.final_approach_d == 0.0, p.final_approach_d


# ================================================================
def main():
    print('=' * 74)
    print('  预瞄点（turn_shift 根治）验收')
    print('=' * 74)
    check('① 关（默认）：final_approach_d == 0，行为不变', t_off_by_default)
    check('② 开：最后一段的方向 ≈ 目标朝向', t_on_makes_last_leg_face_goal_yaw)
    check('③ 终点背后被占 ⇒ 退回普通规划', t_pre_blocked_falls_back)
    check('④ 阶梯：1.2 不行、0.8 可行 ⇒ 用 0.8', t_ladder_picks_shorter)
    check('⑤ 起终点太近 ⇒ 不折腾', t_too_close_skips)
    print('-' * 74)
    print('  通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for name, e in FAIL:
            print('  ❌ %s: %s' % (name, e))
        return 1
    print('  ✅ 预瞄点全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
