# -*- coding: utf-8 -*-
"""
core/frames.py —— 坐标系对齐（本项目最容易错、也最关键的一步）

背景
----
组1 给「场景图坐标系」，组2 给「目标位姿（场景系）」，
CAPO 给的是「相对起点的里程计读数（capo系）」。
两者之间就差一个刚体变换 T。

★ 关键洞察：场景系的原点在哪，我们的代码【根本看不见】
--------------------------------------------------------
    组1 说:  场景图原点在房间西北角，x 朝东
    组2 说:  狗现在在场景图的 (3.2, 1.8, 90°)
    CAPO 说: 我现在读数是 (0.00, 0.00, 0°)
        ⇒ T_scene_capo = 平移(3.2, 1.8) ∘ 旋转(90°)
        ⇒ 【原点在哪里，自动被吸收进 T 里了】

    所以我们【不需要】再定义"我们自己的坐标系" —— 那只是多一层没用的变换、
    多一个会错的地方。我们需要的是：
        ① 固定【内部约定】（右手系 / z上 / yaw逆时针 / 米 / 弧度）
        ② 一个【适配函数】把组1/组2 的数据规范化进这个约定

★ 而 T 只在【开机时建立一次】，之后全程不变
--------------------------------------------------------
    ❌ 绝不做"每段导航重新对齐"：
       CAPO 漂了还强行对齐 = 宣告我们在标称位置，实际不在
       ⇒ 现实和地图不一致 ⇒ 更容易撞

    代价：开机对齐是整个系统最容易错、也最致命的一步 ——
          狗必须真的停在组2给的起点位姿上。
"""
import math
from dataclasses import dataclass
from typing import Optional

from .types import (FrameMismatch, Pose2D, Transform2D, wrap,
                    FRAME_SCENE, FRAME_CAPO)


# ================================================================ 建立 T
def T_from_correspondence(p_from: Pose2D, p_to: Pose2D) -> Transform2D:
    """已知【同一个物理位姿】在两个坐标系里的表达，求变换 T: from → to。

        p_to = T @ p_from
        θ = p_to.yaw − p_from.yaw
        t = p_to.xy − R(θ) @ p_from.xy

    ★ 这是建立场景系↔CAPO系 的 T 的【唯一】方式，也是全系统最关键的一步。
    """
    if p_from.frame == p_to.frame:
        raise FrameMismatch(
            '两个位姿的 frame 相同（都是 %r），求不出变换。'
            '（是不是把 capo 系和场景系的数搞反了？）' % p_from.frame)

    th = wrap(p_to.yaw - p_from.yaw)
    c, s = math.cos(th), math.sin(th)
    # t = p_to - R(θ) @ p_from
    tx = p_to.x - (c * p_from.x - s * p_from.y)
    ty = p_to.y - (s * p_from.x + c * p_from.y)
    T = Transform2D(from_frame=p_from.frame, to_frame=p_to.frame,
                    x=tx, y=ty, yaw=th)

    # 自检：变换回去必须吻合（浮点误差量级）
    back = T @ p_from
    err = math.hypot(back.x - p_to.x, back.y - p_to.y)
    if err > 1e-9:
        raise AssertionError('T 建立后自检失败，残差 %.3g m' % err)
    return T


def build_T_scene_capo(start_scene: Pose2D, capo_reading: Pose2D) -> Transform2D:
    """开机对齐：组2 给的起点位姿（场景系） + CAPO 此刻的读数 → T_scene_capo。

    ⚠️ 调用前提：狗【真的】停在组2给的起点位姿上。错多少，全程错多少。
    """
    require_frame(start_scene, FRAME_SCENE, 'build_T_scene_capo(start_scene)')
    require_frame(capo_reading, FRAME_CAPO, 'build_T_scene_capo(capo_reading)')
    return T_from_correspondence(capo_reading, start_scene)


# ================================================================ 断言助手
def require_frame(p: Pose2D, frame: str, where: str = ''):
    """在关键入口断言坐标系，让错误【当场】暴露而不是飘到后面。"""
    if p.frame != frame:
        raise FrameMismatch(
            '%s期望 frame=%r，实际 %r\n  %r' % (where + ': ' if where else '', frame, p.frame, p))


def to_frame(T: Transform2D, p: Pose2D, where: str = '') -> Pose2D:
    try:
        return T @ p
    except FrameMismatch as e:
        raise FrameMismatch('%s%s' % (where + ': ' if where else '', e))


# ================================================================ 场景系适配
@dataclass(frozen=True)
class SceneConvention:
    """组1/组2 给的场景图坐标系的【约定】。

    我们内部约定固定为：右手系 / x前 y左 / yaw 逆时针 / 米 / 弧度
    他们给的东西如果不一样，就在这一层转 —— ★ 只改这一个地方。
    """
    angle_unit: str = 'rad'        # 'rad' | 'deg'
    yaw_sign: float = 1.0          # +1 = 逆时针为正；-1 = 顺时针为正
    handedness: str = 'right'      # 'right' | 'left'
    axes: str = 'xy'               # 'xy' = 标准(前x左y)
                                   # 'yx' = GO2 robot_odom 那种(前y右x)
    # 轴向交换时是否翻转其中一个轴（左手系需要）
    flip_y: bool = False


def to_scene_pose(x: float, y: float, yaw: float,
                  conv: SceneConvention = SceneConvention(),
                  where: str = '') -> Pose2D:
    """把组1/组2 给的原始数值 → 我们的内部约定（frame='scene'）。"""
    # ① 角度单位
    a = math.radians(yaw) if conv.angle_unit == 'deg' else yaw
    # ② 转向正负
    a = a * conv.yaw_sign
    # ③ 轴向
    if conv.axes == 'xy':
        px, py = x, y
    elif conv.axes == 'yx':
        # 他们: +x=右, +y=前  →  我们: +x=前, +y=左
        px, py = y, -x
    else:
        raise ValueError('未知的 axes 约定: %r' % conv.axes)
    # ④ 手性
    if conv.handedness == 'left':
        py = -py
        a = -a
    if conv.flip_y:
        py = -py
    if where:
        pass
    return Pose2D(px, py, wrap(a), frame=FRAME_SCENE)


# ================================================================ 时钟
class ClockSource:
    """狗上有【两个时钟】，差 56 年（2026-09-22 实测）。

        CAPO 时间戳      → 狗 Linux 时钟（1970，未同步）
        Point-LIO/sport  → 雷达/主板时钟（真实 2026）

    规矩：
        · 控制回路【只用 recv_t】（本机收到的单调时刻）
        · stamp 只做记录/参考，★ 绝不拿它跨源比较
    """
    DOG_LINUX = 'dog_linux'     # 1970，需要 sync_time.sh 同步
    LIDAR_BOARD = 'lidar_board'  # 真实时间


def describe_clock_mismatch() -> str:
    return ('CAPO 用狗 Linux 时钟，Point-LIO/sport 用雷达主板时钟，'
            '两者差约 56 年 —— 跨源比时间戳必然出错，控制一律用 recv_t')
