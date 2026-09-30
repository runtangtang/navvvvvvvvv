# -*- coding: utf-8 -*-
"""
tests/test_frames.py —— P0 验收

不需要 pytest，直接跑：python3 tests/test_frames.py

验收项（对应任务规划 P0）
  ① T_scene_capo @ pose_capo → frame='scene' 且数值正确
  ② T 的方向不对              → 抛 FrameMismatch
  ③ pose + pose（裸相加）     → 抛 FrameMismatch
  ④ T 与逆变换自洽            → T_inv @ (T @ p) ≈ p
  ⑤ 适配函数：GO2 robot_odom(前y右x) / 度数 / 顺时针 → 正确转成内部约定
  ⑥ ★ "原点在哪看不见" —— 换一个场景系原点定义，同一段运动结果不变
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import (FrameMismatch, Pose2D, Transform2D,
                        FRAME_SCENE, FRAME_CAPO)
from core.frames import (SceneConvention, T_from_correspondence,
                         build_T_scene_capo, to_scene_pose, require_frame,
                         wrap)

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
    except Exception as e:
        raise AssertionError('期望 %s，实际抛了 %s: %s' % (exc.__name__, type(e).__name__, e))
    raise AssertionError('期望抛 %s，但什么都没抛' % exc.__name__)


def near(a, b, tol=1e-9):
    assert abs(a - b) < tol, '%.12g != %.12g (tol %.1g)' % (a, b, tol)


# ================================================================ ① + ④
def t_capo2scene():
    """组2 说起点在场景系 (3.2, 1.8, 90°)，CAPO 此刻读 (0, 0, 0)。"""
    scene_start = Pose2D(3.2, 1.8, math.radians(90), frame=FRAME_SCENE)
    capo_now = Pose2D(0.0, 0.0, 0.0, frame=FRAME_CAPO)
    T = build_T_scene_capo(scene_start, capo_now)

    # CAPO 往前走 1m（capo系 +x）
    p = T @ Pose2D(1.0, 0.0, 0.0, frame=FRAME_CAPO)
    assert p.frame == FRAME_SCENE
    # 场景系 yaw=90° ⇒ capo 的 +x 对应场景系的 +y
    near(p.x, 3.2, 1e-9)
    near(p.y, 2.8, 1e-9)

    # ④ 逆变换自洽
    q = T.inverse() @ p
    near(q.x, 1.0); near(q.y, 0.0); near(q.yaw, 0.0, 1e-9)
    assert q.frame == FRAME_CAPO


def t_rotate_and_inv():
    T = Transform2D(FRAME_CAPO, FRAME_SCENE,
                    2.0, -1.0, math.radians(37))
    for p in (Pose2D(0, 0, 0, FRAME_CAPO),
              Pose2D(1.3, -0.7, math.radians(-120), FRAME_CAPO)):
        back = T.inverse() @ (T @ p)
        near(back.x, p.x, 1e-9)
        near(back.y, p.y, 1e-9)
        near(wrap(back.yaw - p.yaw), 0.0, 1e-9)


# ================================================================ ② + ③
def t_wrong_direction():
    T = Transform2D(FRAME_CAPO, FRAME_SCENE, 0, 0, 0)
    # 传一个 scene 系的位姿进去（方向反了）
    expect_raises(FrameMismatch,
                  lambda: T @ Pose2D(0, 0, 0, frame=FRAME_SCENE))


def t_no_naked_add():
    a = Pose2D(1, 2, 0, frame=FRAME_SCENE)
    b = Pose2D(1, 2, 0, frame=FRAME_CAPO)
    expect_raises(FrameMismatch, lambda: a + b)
    expect_raises(FrameMismatch, lambda: a - b)


def t_mismatch_transform_chain():
    A = Transform2D(FRAME_CAPO, FRAME_SCENE, 0, 0, 0)
    B = Transform2D(FRAME_CAPO, FRAME_SCENE, 0, 0, 0)   # 接不上（应该 from=scene）
    expect_raises(FrameMismatch, lambda: A @ B)


def t_delta_compose():
    """delta/compose 是互逆的：p.compose(p.delta(q)) == q"""
    a = Pose2D(1.0, 2.0, math.radians(30), frame=FRAME_SCENE)
    b = Pose2D(-0.5, 0.8, math.radians(-10), frame=FRAME_SCENE)
    rel = a.delta(b)                 # 从 a 到 b 的相对位姿
    assert rel.frame == FRAME_SCENE
    back = a.compose(rel)            # 从 a 走 rel → 应该到 b
    near(back.x, b.x); near(back.y, b.y)
    near(wrap(back.yaw - b.yaw), 0.0, 1e-9)

    # 反过来也成立
    back2 = b.compose(b.delta(a))
    near(back2.x, a.x); near(back2.y, a.y)


# ================================================================ ⑤ 适配函数
def t_adapter_go2_robot_odom():
    """GO2 robot_odom：前=y 右=x。往前走 1m 它给 (0, 1)，我们应该得到 (1, 0)。"""
    conv = SceneConvention(axes='yx')
    p = to_scene_pose(0.0, 1.0, 0.0, conv)
    near(p.x, 1.0); near(p.y, 0.0)
    assert p.frame == FRAME_SCENE

    # 往右 1m：它给 (1, 0)，我们应该得到 (0, -1)（右 = 负 y）
    p = to_scene_pose(1.0, 0.0, 0.0, conv)
    near(p.x, 0.0); near(p.y, -1.0)


def t_adapter_degrees_clockwise():
    """他们用【度数】且【顺时针为正】：转 90°(右转) → 我们应该是 -90°。"""
    conv = SceneConvention(angle_unit='deg', yaw_sign=-1.0)
    p = to_scene_pose(0, 0, 90.0, conv)
    near(p.yaw, math.radians(-90), 1e-12)

    # 标准情况：度数 + 逆时针
    p2 = to_scene_pose(0, 0, 90.0, SceneConvention(angle_unit='deg'))
    near(p2.yaw, math.radians(90), 1e-12)


def t_adapter_left_handed():
    conv = SceneConvention(handedness='left')
    p = to_scene_pose(1.0, 1.0, math.radians(30), conv)
    near(p.x, 1.0); near(p.y, -1.0)
    near(p.yaw, math.radians(-30), 1e-12)


# ================================================================ ⑥ 原点无关性
def t_origin_invariance():
    """★ 核心论断：换一个场景系原点定义，【同一段运动】结果不变。

    第一次：原点 = 狗开机的位置（起点位姿 = (0,0,0)）
    第二次：原点 = 房间西北角（起点位姿 = (3.2, 1.8, 90°)）
    两次算出来的"走了多远 / 朝哪"必须【完全一样】。
    """
    capo_start = Pose2D(0.0, 0.0, 0.0, frame=FRAME_CAPO)
    capo_end = Pose2D(1.0, 0.2, math.radians(5), frame=FRAME_CAPO)

    results = []
    for scene_start in (Pose2D(0.0, 0.0, 0.0, frame=FRAME_SCENE),
                        Pose2D(3.2, 1.8, math.radians(90), frame=FRAME_SCENE),
                        Pose2D(-12.7, 4.4, math.radians(-137), frame=FRAME_SCENE)):
        T = build_T_scene_capo(scene_start, capo_start)
        p_end_scene = T @ capo_end
        rel = p_end_scene.delta(scene_start)     # 相对起点的位移（场景系）
        results.append((rel.x, rel.y, rel.yaw))

    for r in results[1:]:
        near(r[0], results[0][0], 1e-9)
        near(r[1], results[0][1], 1e-9)
        near(r[2], results[0][2], 1e-9)


def t_require_frame():
    expect_raises(FrameMismatch,
                  lambda: require_frame(Pose2D(0, 0, 0, FRAME_CAPO),
                                        FRAME_SCENE, 'unit test'))


# ================================================================ main
def main():
    print('=' * 66)
    print('  P0 验收：坐标系地基')
    print('=' * 66)
    groups = [
        ('①④ 变换正确 + 逆变换自洽', [
            ('T_scene_capo @ pose_capo → frame=scene 且数值对', t_capo2scene),
            ('T 与逆变换自洽（任意旋转/平移）', t_rotate_and_inv),
        ]),
        ('②③ 混用必须抛异常', [
            ('T 方向反了 → FrameMismatch', t_wrong_direction),
            ('裸相加 pose+pose → FrameMismatch', t_no_naked_add),
            ('变换链接不上 → FrameMismatch', t_mismatch_transform_chain),
            ('delta/compose 互逆', t_delta_compose),
        ]),
        ('⑤ 场景系适配函数', [
            ('GO2 robot_odom（前y右x）→ 内部约定', t_adapter_go2_robot_odom),
            ('度数 / 顺时针为正 → 内部约定', t_adapter_degrees_clockwise),
            ('左手系 → 内部约定', t_adapter_left_handed),
        ]),
        ('⑥ ★ 场景系原点在哪，结果不变', [
            ('换 3 个不同原点定义，相对位移完全一致', t_origin_invariance),
        ]),
        ('  入口断言', [
            ('require_frame 用错就报错', t_require_frame),
        ]),
    ]
    for title, fns in groups:
        print()
        print('-- %s' % title)
        for name, fn in fns:
            check(name, fn)

    print()
    print('=' * 66)
    print('  通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print('  失败项:')
        for n, e in FAIL:
            print('    - %s' % n)
        print('=' * 66)
        return 1
    print('  ✅ P0 全部通过')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
