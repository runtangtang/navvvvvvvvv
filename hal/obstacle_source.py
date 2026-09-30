# -*- coding: utf-8 -*-
"""动态障碍点源：ROS 话题接收与纯离线测试源。"""
import threading
import time
from typing import Optional, Protocol

import numpy as np


class ObstacleSource(Protocol):
    """CAPO 系障碍点源；None 表示数据失效，空数组表示确认无障碍。"""

    def get(self) -> Optional[np.ndarray]: ...
    def healthy(self, max_age: float = 1.0) -> bool: ...
    def error_count(self) -> int: ...


class RosObstacleSource:
    """订阅 ``/nav/obstacle_points`` 的 ``Float32MultiArray``。"""

    def __init__(self, node, topic: str = '/nav/obstacle_points',
                 max_age_s: float = 1.0, qos_depth: int = 10):
        from std_msgs.msg import Float32MultiArray

        self.node = node
        self.topic = topic
        self.name = topic
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._pts: Optional[np.ndarray] = None
        self._recv_t = 0.0
        self._errs = 0
        node.create_subscription(Float32MultiArray, topic, self._cb, qos_depth)

    @staticmethod
    def parse_message(msg) -> np.ndarray:
        """解析消息为独立的 ``(N,3) float32`` 数组，供回调和离线测试共用。"""
        flat = np.asarray(msg.data, dtype=np.float32)
        if flat.ndim != 1 or flat.size % 3 != 0:
            raise ValueError('障碍点消息长度必须是 3 的倍数，收到 %d' % flat.size)
        return flat.reshape(-1, 3).copy()

    def _cb(self, msg):
        # ★ 绝不抛异常：否则 executor 会静默死亡，而控制循环仍可能继续发指令。
        try:
            pts = self.parse_message(msg)
            recv_t = time.time()       # ★ 只存到达时刻，不碰消息时间戳
            with self._lock:
                self._pts = pts
                self._recv_t = recv_t
        except Exception:
            with self._lock:
                self._errs += 1

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._pts is None or time.time() - self._recv_t > self.max_age_s:
                return None
            return self._pts.copy()

    def healthy(self, max_age: float = 1.0) -> bool:
        with self._lock:
            return (self._pts is not None
                    and time.time() - self._recv_t <= max_age)

    def error_count(self) -> int:
        with self._lock:
            return self._errs


class MockObstacleSource:
    """测试源；``set(None)`` 注入失效，``set(empty)`` 注入确认无点。"""

    def __init__(self, pts=None, name: str = 'mock-obstacles'):
        self.name = name
        self._lock = threading.Lock()
        self._pts: Optional[np.ndarray] = None
        self._recv_t = 0.0
        self._errs = 0
        if pts is not None:
            self.set(pts)

    def set(self, pts, recv_t: Optional[float] = None):
        if pts is None:
            arr = None
        else:
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError('障碍点必须是 (N,3)，收到 %r' % (arr.shape,))
            arr = arr.copy()
        with self._lock:
            self._pts = arr
            self._recv_t = recv_t if recv_t is not None else time.time()

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._pts is None else self._pts.copy()

    def healthy(self, max_age: float = 1.0) -> bool:
        with self._lock:
            return self._pts is not None

    def error_count(self) -> int:
        with self._lock:
            return self._errs
