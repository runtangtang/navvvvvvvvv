# -*- coding: utf-8 -*-
"""
io/pose_source.py —— 位姿来源（★ 一份代码多种后端）

设计要点
--------
① **输出永远带 frame 标签**，绝不吐裸 (x, y)。
   控制层拿到的东西天然知道自己是哪个系的。

② **输出带两个时间**（Timed）：
       recv_t  本机收到时刻   ← ★ 控制回路只用这个
       stamp   消息自带时间戳  ← 可能来自另一个时钟（狗上差 56 年），只做记录

③ **回调绝不抛异常**。2026-09-22 真发生过：一个回调抛了 → rclpy executor 线程死
   → 所有订阅静默停止 → 而主线程还在发运动指令，狗白走了 2.7m。
   这里把异常计数记下来，让 watchdog 能看见。

④ "真机 / 仿真 / 回放" 不是三个类，是**同一个类的不同参数** ——
   CAPO 和 sim_dog 发的是同一个话题，没有第二份代码，就没有"仿真对真机错"的鸿沟。
"""
import math
import threading
import time
from typing import Optional, Protocol

from core.types import Timed, Pose2D, FrameMismatch, FRAME_CAPO


def yaw_of_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoseSource(Protocol):
    name: str

    def get(self) -> Optional[Timed]: ...
    def healthy(self, max_age: float = 0.3) -> bool: ...
    def error_count(self) -> int: ...


# ================================================================ 真源
class OdomSource:
    """订阅一个 nav_msgs/Odometry，输出 Timed[Pose2D]。

    真机:      OdomSource(node, '/SMX/Odom',        frame='capo', name='CAPO')
    仿真:      OdomSource(node, '/SMX/Odom',        frame='capo', name='sim-CAPO')
    备份:      OdomSource(node, '/state_estimation', frame='capo', name='Point-LIO')
               ★ 注意 Point-LIO 有自己的 frame 和时钟；如果要用它当位姿源，
                 必须把它当作另一个 capo 系处理（它的原点由 Point-LIO 自己定），
                 并重新建立 T —— 不能和 CAPO 的 T 混用。
    """

    def __init__(self, node, topic: str, frame: str = FRAME_CAPO,
                 name: Optional[str] = None, qos_depth: int = 200):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from nav_msgs.msg import Odometry
        self.node = node
        self.topic = topic
        self.frame = frame
        self.name = name or topic
        self._lock = threading.Lock()
        self._latest: Optional[Timed] = None
        self._n = 0
        self._errs = 0
        qr = QoSProfile(depth=qos_depth, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        node.create_subscription(Odometry, topic, self._cb, qr)

    def _cb(self, m):
        # ★ 绝不抛异常：抛了会杀掉 executor，订阅静默停止，而主线程还在发指令
        try:
            p = m.pose.pose
            st = m.header.stamp
            t = Timed(recv_t=time.time(),
                      stamp=st.sec + st.nanosec * 1e-9,
                      data=Pose2D(p.position.x, p.position.y,
                                  yaw_of_quat(p.orientation), frame=self.frame))
            with self._lock:
                self._latest = t
                self._n += 1
        except Exception:
            with self._lock:
                self._errs += 1

    def get(self) -> Optional[Timed]:
        with self._lock:
            return self._latest

    def error_count(self) -> int:
        with self._lock:
            return self._errs

    def count(self) -> int:
        with self._lock:
            return self._n

    def healthy(self, max_age: float = 0.3) -> bool:
        with self._lock:
            if self._latest is None:
                return False
            return (time.time() - self._latest.recv_t) <= max_age

    def wait_ready(self, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get() is not None:
                return True
            time.sleep(0.1)
        return False


# ================================================================ 假源（测试）
class MockSource:
    """测试/回放用：手动喂位姿。用于单测控制律、状态机，不需要 ROS。"""

    def __init__(self, frame: str = FRAME_CAPO, name: str = 'mock'):
        self.frame = frame
        self.name = name
        self._latest: Optional[Timed] = None
        self._errs = 0

    def feed(self, p: Pose2D, recv_t: Optional[float] = None):
        if p.frame != self.frame:
            raise FrameMismatch('MockSource(%s) 收到 frame=%s 的位姿' % (self.frame, p.frame))
        self._latest = Timed(recv_t=recv_t if recv_t is not None else time.time(),
                             stamp=None, data=p)

    def get(self) -> Optional[Timed]:
        return self._latest

    def error_count(self) -> int:
        return self._errs

    def healthy(self, max_age: float = 0.3) -> bool:
        if self._latest is None:
            return False
        return (time.time() - self._latest.recv_t) <= max_age


# ================================================================ 组合
class ScaledSource:
    """把某个源的位移乘一个尺度。仿真里当【缺陷注入器】，真机上当【尺度校正】。

    ## 仿真：注入缺陷
    量化"里程计少算 X% 会让最终误差差多少"。

    ## 真机：校正（★ 2026-09-23 之后可以用了）
    2026-09-22 曾写"真机上不要用它去修里程计 —— 尺度不稳定的话固定系数修不了"。
    **那条已过期**，2026-09-23 实验室 7 组开环标定（1.5m×5 + 2.5m×2）证明：

        线性尺度  k = CAPO/卷尺 = 0.954   （7 次均值）
        CAPO/sport 比值七次标准差 = 0.34%  ⇒ 尺度【是稳定的】
        昨天"两次差 7%" 是【卷尺读数的散布】（±1.8%，r=0.99 证明变动在卷尺）
        ⇒ 结论见 D:\\迷雾通\\尺度标定_20260923.md

    所以真机上 **推荐** 用 scale = 1/0.954 = 1.048 校正。
    闭环里不校正的后果：命令走 L 米 ⇒ 实际走 L/0.954 = 1.048L（多走 4.8%）。

    ⚠️ 绝对精度仍带 ~3% 系统不确定度（卷尺口径 vs sport 锚点的分歧，见文档）。
    """

    def __init__(self, inner: PoseSource, scale: float,
                 name: Optional[str] = None):
        self.inner = inner
        self.scale = scale
        self.name = name or ('%s x%.3f' % (inner.name, scale))
        self._origin: Optional[Pose2D] = None

    def _scaled(self, p: Pose2D) -> Pose2D:
        if self._origin is None:
            self._origin = p
            return p
        o = self._origin
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        dx, dy = p.x - o.x, p.y - o.y
        # 转到原点朝向下，缩放，再转回来
        fx = c * dx + s * dy
        fy = -s * dx + c * dy
        fx *= self.scale
        fy *= self.scale
        return Pose2D(o.x + c * fx - s * fy,
                      o.y + s * fx + c * fy,
                      p.yaw, frame=p.frame)

    def get(self) -> Optional[Timed]:
        t = self.inner.get()
        if t is None:
            return None
        return Timed(recv_t=t.recv_t, stamp=t.stamp, data=self._scaled(t.data))

    def healthy(self, max_age: float = 0.3) -> bool:
        return self.inner.healthy(max_age)

    def wait_ready(self, timeout: float = 20.0) -> bool:
        """透传给内层（OdomSource 用它等第一条消息）。"""
        fn = getattr(self.inner, 'wait_ready', None)
        return bool(fn(timeout)) if fn else True

    def error_count(self) -> int:
        return self.inner.error_count()
