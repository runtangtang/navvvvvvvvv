# -*- coding: utf-8 -*-
"""
core/types.py —— 带坐标系标签的位姿 / 变换

★ 为什么要有这个文件（这是整个架构里最值钱的一块）
--------------------------------------------------
我们在【坐标系】上栽过 4 次：障碍感知节点、单帧 ICP 外参、联合标定、cloud 建图。
而且每一次【都不报错】—— 90° 的偏移不会崩，只会让结果"看着不太对劲"。

所以这里把坐标系做成【类型的一部分】：
    混用 → 当场抛 FrameMismatch，而不是悄悄算出一个错的数

约定（全项目统一，和 CAPO / 机器人学界一致）
--------------------------------------------------
    右手系 / z 轴朝上 / x 朝前 / y 朝左 / yaw 逆时针为正 / 米 / 弧度

⚠️ 注意：狗自己的 robot_odom 是【前=y 右=x】，和这套【不是一回事】。
   本项目统一用这一套；读 robot_odom 时必须立刻转换，绝不让它流进来。
"""
import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------- 坐标系标签
# 集中定义，避免各处写字符串写错
FRAME_SCENE = 'scene'    # 组1/组2 的场景图坐标系（全局）
FRAME_CAPO = 'capo'      # CAPO /SMX/Odom（标准 IMU 约定，相对起点）
FRAME_BODY = 'body'      # 当前机体系（只用于速度指令）
FRAME_GRID = 'grid'      # 栅格坐标系（= 场景系 + origin/resolution）

KNOWN_FRAMES = {FRAME_SCENE, FRAME_CAPO, FRAME_BODY, FRAME_GRID}


class FrameMismatch(Exception):
    """坐标系混用。★ 这个异常意味着代码里有个真 bug，不要 try 掉它。"""


def wrap(a: float) -> float:
    """角度归一化到 (-pi, pi]"""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------- 位姿
@dataclass(frozen=True)
class Pose2D:
    """平面位姿。frame 是【必填】的 —— 这一步就是防呆的核心。"""
    x: float
    y: float
    yaw: float
    frame: str

    def __post_init__(self):
        if not self.frame:
            raise ValueError('Pose2D 必须指定 frame（不能留空）')

    # ---- 别名，读起来清楚点 ----
    @property
    def xy(self):
        return (self.x, self.y)

    # ---- ★ 禁止裸相加/相减：那是最典型的坐标系混用 ----
    def __add__(self, other):
        raise FrameMismatch(
            '不能把两个 Pose2D 相加。\n'
            '  想叠加相对位姿 → 用 p.compose(rel)\n'
            '  想求相对位姿   → 用 p.delta(other)\n'
            '  （两个位姿在不同的 frame 里相加，是 90%% 坐标系 bug 的来源）')

    __sub__ = __add__

    # ---- 正确的两种组合方式 ----
    def delta(self, other: 'Pose2D') -> 'Pose2D':
        """【从 self 到 other】的相对位姿 = self⁻¹ ∘ other（同一个 frame 里）。

        不变的等式：p.compose(p.delta(q)) == q
        """
        self._same_frame(other)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = other.x - self.x, other.y - self.y
        return Pose2D(c * dx + s * dy,          # R(-self.yaw) @ d
                      -s * dx + c * dy,
                      wrap(other.yaw - self.yaw),
                      frame=self.frame)

    def compose(self, rel: 'Pose2D') -> 'Pose2D':
        """把【相对位姿】叠加到 self 上（rel 必须与 self 同 frame）。"""
        self._same_frame(rel)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(self.x + c * rel.x - s * rel.y,
                      self.y + s * rel.x + c * rel.y,
                      wrap(self.yaw + rel.yaw),
                      frame=self.frame)

    def retag(self, frame: str, why: str = '') -> 'Pose2D':
        """★ 危险操作：只换标签不改数值。命名故意难听，用之前先想清楚。

        合法用途：把一个【本来就是】新 frame 的数值标上正确的标签
                  （比如从文件里读出来已知是场景系的坐标）。
        """
        return Pose2D(self.x, self.y, self.yaw, frame=frame)

    def _same_frame(self, other: 'Pose2D'):
        if other.frame != self.frame:
            raise FrameMismatch(
                '坐标系不一致：self.frame=%r, other.frame=%r' % (self.frame, other.frame))

    def __repr__(self):
        return 'Pose2D(%.3f, %.3f, %.1f°, frame=%s)' % (
            self.x, self.y, math.degrees(self.yaw), self.frame)


@dataclass(frozen=True)
class Pose6D:
    """给对外接口用的 6D 位姿。导航只用平面 3 个自由度。"""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    frame: str

    @property
    def planar(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw, frame=self.frame)

    @staticmethod
    def from_planar(p: Pose2D, z: float = 0.0, roll: float = 0.0,
                    pitch: float = 0.0) -> 'Pose6D':
        return Pose6D(p.x, p.y, z, roll, pitch, p.yaw, frame=p.frame)

    def __repr__(self):
        return 'Pose6D(%.3f, %.3f, z=%.3f, rpy=(%.1f, %.1f, %.1f)°, frame=%s)' % (
            self.x, self.y, self.z, math.degrees(self.roll),
            math.degrees(self.pitch), math.degrees(self.yaw), self.frame)


# ---------------------------------------------------------------- 变换
@dataclass(frozen=True)
class Transform2D:
    """刚体变换：p_to = R(yaw) @ p_from + (x, y)

    from_frame / to_frame 是【显式】的 —— 用错方向会当场报错。
    """
    from_frame: str
    to_frame: str
    x: float
    y: float
    yaw: float

    def __matmul__(self, other):
        """T @ Pose2D  → Pose2D     |     T @ Transform2D → Transform2D"""
        if isinstance(other, Pose2D):
            if other.frame != self.from_frame:
                raise FrameMismatch(
                    '变换方向不对：T 是 %s→%s，但传入的位姿 frame=%s\n'
                    '  （想反着用就取 T.inverse()）'
                    % (self.from_frame, self.to_frame, other.frame))
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            return Pose2D(self.x + c * other.x - s * other.y,
                          self.y + s * other.x + c * other.y,
                          wrap(self.yaw + other.yaw),
                          frame=self.to_frame)

        if isinstance(other, Transform2D):
            if other.from_frame != self.to_frame:
                raise FrameMismatch(
                    '变换链对不上：%s→%s 后面接的应该是 from_frame=%s 的变换，'
                    '实际是 %s' % (self.from_frame, self.to_frame, self.to_frame,
                                   other.from_frame))
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            return Transform2D(
                from_frame=other.from_frame, to_frame=self.to_frame,
                x=self.x + c * other.x - s * other.y,
                y=self.y + s * other.x + c * other.y,
                yaw=wrap(self.yaw + other.yaw))

        raise TypeError('Transform2D @ 只支持 Pose2D / Transform2D，收到 %r' % type(other))

    def inverse(self) -> 'Transform2D':
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Transform2D(from_frame=self.to_frame, to_frame=self.from_frame,
                           x=-(c * self.x + s * self.y),
                           y=-(-s * self.x + c * self.y),
                           yaw=wrap(-self.yaw))

    @staticmethod
    def identity(frame: str) -> 'Transform2D':
        return Transform2D(frame, frame, 0.0, 0.0, 0.0)

    def __repr__(self):
        return 'T(%s→%s, t=(%.3f, %.3f), rot=%.2f°)' % (
            self.from_frame, self.to_frame, self.x, self.y, math.degrees(self.yaw))


# ---------------------------------------------------------------- 速度
@dataclass(frozen=True)
class Velocity:
    """机体系速度指令。★ 永远是【机体系】，不带 frame 标签 —— 因为它没有原点。

    （位姿需要 frame，速度不需要：速度是"怎么动"，不是"在哪"。
      但它的【正方向】必须和 body 系的约定一致：x前 y左 z上，yaw逆时针为正。）
    """
    vx: float = 0.0     # 前向 m/s
    vy: float = 0.0     # 左向 m/s（GO2 能横移；我们的控制器目前恒为 0）
    wz: float = 0.0     # 偏航角速度 rad/s，逆时针为正

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def clipped(self, vmax: float, wmax: float) -> 'Velocity':
        return Velocity(max(-vmax, min(vmax, self.vx)),
                        max(-vmax, min(vmax, self.vy)),
                        max(-wmax, min(wmax, self.wz)))

    def __repr__(self):
        return 'V(vx=%+.3f, vy=%+.3f, wz=%+.3f)' % (self.vx, self.vy, self.wz)


ZERO_VEL = Velocity()


# ---------------------------------------------------------------- 带时间戳
@dataclass
class Timed:
    """带【两个时间】的数据。

    ★ 狗上有两个时钟，差 56 年：
        CAPO 时间戳     → 狗 Linux 时钟（1970，未同步）
        Point-LIO/sport → 雷达/主板时钟（真实 2026）

    所以：
        recv_t  = 本机收到的时刻（单调）—— ★ 控制回路【只用这个】
        stamp   = 消息自带时间戳（可能来自另一个时钟，只做参考/记录）
    """
    recv_t: float
    stamp: Optional[float]
    data: Any

    def age(self, now: Optional[float] = None) -> float:
        """数据有多旧（用 recv_t 算，跨源可比）"""
        return (now if now is not None else _now()) - self.recv_t


def _now() -> float:
    import time
    return time.time()
