# -*- coding: utf-8 -*-
"""
io/cmd_sink.py —— ★ 速度指令的【唯一出口】

为什么要有这个文件
------------------
2026-09-21 的教训：CMU 的 `pathFollower` 会**无条件**往 `/api/sport/request`
发运动指令，`is_real_robot` 只作用于 localPlanner、**管不住它**。
（那天狗抽搐过两次，时间上和我们跑链路测试重合。）

⇒ 架构上的规矩：**上层不许绕过下层。控制层想发指令，只能走这里。**
  安全开关只装在这一道门上，不散落在各处。

三种出口
--------
  DryRunSink   记录每一条指令，★ 一条都不发。用于验证链路、调试控制律。
  SportSink    真的发 /api/sport/request。
               ★ 真狗和仿真用的是【同一个】—— sim_dog 也在听这个话题。
                 所以"真机还是仿真"不是代码的区别，是【现场有没有狗】的区别。
                 正因如此，构造 SportSink 时必须显式表态。

安全约束（写在代码里，不靠人记）
--------------------------------
  · 每次构造 SportSink 都要显式传 acknowledge_motion=True
  · 硬速度上限（超过直接拒绝，不是截断 —— 截断会掩盖上游的 bug）
  · 心跳：超过 heartbeat_timeout 没收到指令 → 自动 settle()
"""
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from core.types import Velocity, ZERO_VEL

API_MOVE, API_STOPMOVE, API_BALANCESTAND = 1008, 1003, 1002

# 硬上限：不管上层传什么，超过这个值直接拒绝执行
HARD_VX_MAX = 1.0      # m/s
HARD_WZ_MAX = 1.5      # rad/s


class CommandRejected(Exception):
    """上层发了超出允许范围的速度。★ 这是 bug，不要 try 掉。"""


class CommandSink(Protocol):
    """控制层看到的唯一接口。"""
    name: str

    def move(self, v: Velocity) -> None: ...
    def stop(self) -> None: ...
    def settle(self) -> None: ...          # StopMove + BalanceStand
    @property
    def is_dry_run(self) -> bool: ...
    def history(self) -> List[tuple]: ...


# ================================================================ 基类
class _BaseSink:
    name = 'base'

    def __init__(self):
        self._hist: List[tuple] = []      # (t, kind, vx, vy, wz)
        self._lock = threading.Lock()
        self._last_move_t: Optional[float] = None
        self._n_move = 0
        self._n_stop = 0

    @property
    def is_dry_run(self) -> bool:
        return True

    def history(self) -> List[tuple]:
        with self._lock:
            return list(self._hist)

    def _record(self, kind, v: Velocity = ZERO_VEL):
        with self._lock:
            self._hist.append((time.time(), kind, v.vx, v.vy, v.wz))

    # ---- 统一的速度合法性检查 ----
    def _validate(self, v: Velocity):
        if not all(math.isfinite(x) for x in (v.vx, v.vy, v.wz)):
            raise CommandRejected('速度里有 NaN/Inf: %r' % (v,))
        if abs(v.vx) > HARD_VX_MAX or abs(v.vy) > HARD_VX_MAX or abs(v.wz) > HARD_WZ_MAX:
            raise CommandRejected(
                '速度超过硬上限（vx/vy<=%.1f m/s, wz<=%.1f rad/s）: %r'
                % (HARD_VX_MAX, HARD_WZ_MAX, v))

    # ---- 心跳：调 move() 之前先看上次指令多久了 ----
    def _check_heartbeat(self, timeout: float):
        if self._last_move_t is None:
            return
        gap = time.time() - self._last_move_t
        if gap > timeout:
            # 断流：狗会保持最后一条指令 —— 必须显式停
            self.settle()

    def move(self, v: Velocity):
        self._validate(v)
        self._num = getattr(self, '_num', 0) + 1

    def stop(self):
        self._n_stop += 1

    def settle(self):
        self.stop()


# ================================================================ DryRun
class DryRunSink(_BaseSink):
    """★ 一条都不发。用于验证链路、调控制律、仿真前的干跑。

    会大声把每条指令记下来 —— 但【绝不发布任何东西】。
    """
    name = 'dry-run'

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    @property
    def is_dry_run(self) -> bool:
        return True

    def move(self, v: Velocity):
        super().move(v)
        self._record('move', v)
        self._n_move += 1
        self._last_move_t = time.time()
        if self.verbose:
            print('  [dry-run] move %s' % v)

    def stop(self):
        super().stop()
        self._record('stop')
        if self.verbose:
            print('  [dry-run] stop')

    def settle(self):
        self.stop()
        self._record('settle')
        if self.verbose:
            print('  [dry-run] settle (StopMove + BalanceStand)')


# ================================================================ 真发
class SportSink(_BaseSink):
    """真的发 /api/sport/request。

    ★ 真狗和仿真都用这个（sim_dog 听同一个话题）。
      所以"会不会让真狗动"取决于【现场有没有狗】，不是取决于用哪个类
      —— 这就是为什么必须显式传 acknowledge_motion=True。
    """
    name = 'sport'

    def __init__(self, node, *, acknowledge_motion: bool,
                 topic: str = '/api/sport/request',
                 heartbeat_timeout: float = 1.0):
        super().__init__()
        if not acknowledge_motion:
            raise ValueError(
                'SportSink 会让狗真的动起来。\n'
                '  如果你只想验证链路/调控制律 → 用 DryRunSink\n'
                '  确实要给狗发指令 → 显式传 acknowledge_motion=True')
        self.node = node
        self.heartbeat_timeout = heartbeat_timeout
        self._n_move = 0
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy)
        from unitree_api.msg import Request
        qr = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self._pub = node.create_publisher(Request, topic, qr)
        self._Request = Request

    @property
    def is_dry_run(self) -> bool:
        return False

    def _send(self, api_id: int, param: Optional[dict] = None):
        r = self._Request()
        r.header.identity.api_id = int(api_id)
        if param is not None:
            r.parameter = json.dumps(param)
        self._pub.publish(r)

    def move(self, v: Velocity):
        super().move(v)
        self._check_heartbeat(self.heartbeat_timeout)
        self._send(API_MOVE, {'x': float(v.vx), 'y': float(v.vy), 'z': float(v.wz)})
        self._record('move', v)
        self._n_move += 1
        self._last_move_t = time.time()

    def stop(self):
        super().stop()
        for _ in range(6):
            self._send(API_STOPMOVE)
            time.sleep(0.01)
        self._record('stop')

    def settle(self):
        """★ StopMove + BalanceStand。
        BalanceStand 必须有 —— 否则如果刚好在抬腿相停下，腿会冻在半空
        → 三条腿支撑 → 倒下（2026-09-22 出现过三次）。"""
        self.stop()
        for _ in range(6):
            self._send(API_BALANCESTAND)
            time.sleep(0.03)
        self._record('settle')

    def stats(self):
        return {'move': self._n_move, 'stop': self._n_stop}
