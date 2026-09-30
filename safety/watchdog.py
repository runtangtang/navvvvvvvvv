# -*- coding: utf-8 -*-
"""
safety/watchdog.py —— 看门狗

它守的是什么
------------
2026-09-22 真机上发生过的、以及差一点发生的每一件事：

  ① 【回调抛异常 → 整个采集静默停掉，而主线程还在发运动指令】
        那天：`_sp` 回调抛了 → rclpy executor 线程死 → CAPO/Point-LIO 的订阅
        一起停 → 狗白走了 2.7m（用户拿遥控器拉的）
        ★ 这是"带病继续"，最危险的一类

  ② 位姿断流（网络断/节点挂）—— 控制器拿着一帧旧位姿继续算

  ③ 控制循环本身卡住（死锁/异常被吞）—— 没人发指令了，但狗还在动

  ④ 阶段 2 的无进展检测漏了 ⇒ 原地空转 9.5 秒

本文件解决 ①③（②④ 由控制循环自己的失效保护管）
—— 看门狗是【最后一道】，独立于控制循环，谁挂了它都能兜住。

设计要点
--------
· 独立线程：控制循环挂了它不受影响
· 一旦触发 → 立刻 sink.settle()（StopMove + BalanceStand），并且**只做一次**
· 触发后**不再自动恢复** —— 恢复到什么状态不是它该决定的事
"""
import threading
import time
from typing import Callable, Optional


class Watchdog:
    def __init__(self, source, sink, *,
                 max_stale: float = 0.30,
                 max_silence: float = 0.50,
                 period: float = 0.05,
                 log: Optional[Callable[[str], None]] = None):
        """
        max_stale   : 位姿多久没更新就判断流 [s]
        max_silence : 控制循环多久没来"报到"就判它挂了 [s]
        """
        self.source = source
        self.sink = sink
        self.max_stale = max_stale
        self.max_silence = max_silence
        self.period = period
        self.log = log or (lambda s: None)

        self._last_feed = time.time()
        self._stop = threading.Event()
        self._th = None
        self._fired: Optional[str] = None        # 触发原因（None = 没触发）
        self._lock = threading.Lock()
        self._enabled = False                    # ★ 默认不守 —— 显式 armed() 才开始

    # ------------------------------------------------------------
    def armed(self) -> 'Watchdog':
        """开始守护。★ 必须在【真的要发运动指令】之前才调 ——
        否则空闲时它会把狗 StopMove 一遍。"""
        with self._lock:
            self._fired = None
            self._enabled = True
            self._last_feed = time.time()
        if self._th is None or not self._th.is_alive():
            self._stop.clear()
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()
        return self

    def disarm(self):
        with self._lock:
            self._enabled = False

    def feed(self):
        """控制循环每转一圈就调一次 —— 表示"我还活着"。"""
        self._last_feed = time.time()

    def triggered(self) -> Optional[str]:
        with self._lock:
            return self._fired

    def stop(self):
        self._stop.set()
        self.disarm()

    # ------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.period)
            with self._lock:
                if not self._enabled:
                    continue
                if self._fired is not None:
                    continue                      # ★ 只触发一次，不反复发指令
            why = self._check()
            if why:
                with self._lock:
                    if self._fired is not None:
                        continue
                    self._fired = why
                    self._enabled = False         # 触发后不再自动恢复
                self.log('  🐕 看门狗触发：%s —— 立即安全停' % why)
                try:
                    self.sink.settle()
                except Exception as e:
                    self.log('  🐕 看门狗 settle 也失败了：%s' % e)

    def _check(self) -> Optional[str]:
        # ③ 控制循环是不是还活着
        silence = time.time() - self._last_feed
        if silence > self.max_silence:
            return 'control_loop_dead (%.2fs 没报到)' % silence
        # ① 位姿源的回调有没有抛异常
        try:
            if self.source.error_count() > 0:
                return 'source_callback_error x%d' % self.source.error_count()
        except Exception as e:
            return 'source.error_count() 自己抛了：%s' % e
        # ② 位姿是不是断了
        try:
            if not self.source.healthy(self.max_stale):
                return 'pose_stale (>%.2fs 没有新位姿)' % self.max_stale
        except Exception as e:
            return 'source.healthy() 自己抛了：%s' % e
        return None


# ================================================================ 控制权
class Authority:
    """控制权令牌。

    规矩：**同一时刻只能有一个人能给狗发指令。**
        · 导航
        · 遥控器（人工接管）
        · 安全层（急停）

    现在的实现很简单（一个字符串），但它把"谁在控制"这件事
    变成了【显式】的，而不是"看谁的手快"。
    """

    NAV = 'nav'
    TELEOP = 'teleop'
    SAFETY = 'safety'
    IDLE = 'idle'

    def __init__(self, who: str = IDLE, log=None):
        self._who = who
        self._lock = threading.Lock()
        self.log = log or (lambda s: None)

    @property
    def who(self) -> str:
        with self._lock:
            return self._who

    def request(self, who: str) -> bool:
        """申请控制权。★ 遥控器/安全层永远可以抢占导航。"""
        with self._lock:
            cur = self._who
            if cur == who:
                return True
            if cur == self.NAV and who in (self.TELEOP, self.SAFETY):
                self._who = who
                self.log('控制权：%s → %s（抢占）' % (cur, who))
                return True
            if cur in (self.IDLE,):
                self._who = who
                self.log('控制权：%s → %s' % (cur, who))
                return True
            self.log('控制权：%s 想抢 %s 的，被拒' % (who, cur))
            return False

    def release(self, who: str):
        with self._lock:
            if self._who == who:
                self._who = self.IDLE
                self.log('控制权：%s 释放' % who)

    def may_move(self, who: str) -> bool:
        with self._lock:
            return self._who == who
