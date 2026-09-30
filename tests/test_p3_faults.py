# -*- coding: utf-8 -*-
"""
tests/test_p3_faults.py —— P3 验收：故障注入（纯逻辑，不需要 ROS，秒级）

★ 注入的全是 2026-09-22 真机上【真的发生过】或差一点发生的故障：

  ① 位姿断流          消息不来了
  ② 回调抛异常        ★ 那天真发生：订阅静默停止，而主线程还在发运动指令
  ③ 控制循环死        没人发指令了，可狗还在按最后一条指令走
  ④ 无进展            走不动了（撞上/卡住）
  ⑤ 超时              目标到不了
  ⑥ 外部急停          遥控器/安全层接管
  ⑦ 速度越界          上层算出个荒唐的速度

判定标准统一：**sink 最后必须收到 settle()**（StopMove + BalanceStand），
             而且返回的 reason 要说清是哪种失败。
"""
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import (Pose2D, Pose6D, Velocity, ZERO_VEL,   # noqa: E402
                        FRAME_SCENE, FRAME_CAPO)
from hal.cmd_sink import DryRunSink, CommandRejected                    # noqa: E402
from hal.pose_source import MockSource                                  # noqa: E402
from control.fine_approach import ApproachConfig                        # noqa: E402
from mission.api import Navigator, NavigateToConfig                     # noqa: E402
from safety.watchdog import Watchdog, Authority                          # noqa: E402

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


def C(x, y, yaw=0.0):
    """假狗/位姿源用的是 capo 系（T=identity 时数值和 scene 一样）"""
    return Pose2D(x, y, yaw, frame=FRAME_CAPO)


def wait_pose(src, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if src.get() is not None:
            return True
        time.sleep(0.02)
    raise AssertionError('位姿源一直没出数据')


# ================================================================ 测试台
class RecSink:
    """记录指令的出口（不发任何东西）。"""
    name = 'rec'
    is_dry_run = True

    def __init__(self):
        self.hist = []
        self.lock = threading.Lock()

    def move(self, v: Velocity):
        if not all(math.isfinite(x) for x in (v.vx, v.vy, v.wz)):
            raise CommandRejected('NaN speed')
        if abs(v.vx) > 1.0 or abs(v.wz) > 1.5:
            raise CommandRejected('speed out of range: %r' % (v,))
        with self.lock:
            self.hist.append(('move', time.time(), v.vx, v.wz))

    def stop(self):
        with self.lock:
            self.hist.append(('stop', time.time(), 0.0, 0.0))

    def settle(self):
        self.stop()
        with self.lock:
            self.hist.append(('settle', time.time(), 0.0, 0.0))

    def last(self):
        with self.lock:
            return self.hist[-1] if self.hist else None

    def settled(self) -> bool:
        with self.lock:
            return any(k == 'settle' for (k, _, _, _) in self.hist)

    def moves_after_settle(self) -> int:
        with self.lock:
            n = 0
            seen = False
            for (k, _, _, _) in self.hist:
                if k == 'settle':
                    seen = True
                elif seen and k == 'move':
                    n += 1
            return n

    def n_move(self) -> int:
        with self.lock:
            return sum(1 for (k, _, _, _) in self.hist if k == 'move')


class FakeDog:
    """把指令积分成位姿，喂给 MockSource。可以按剧本出故障。"""

    def __init__(self, sink, source, dt=0.02):
        self.sink = sink
        self.source = source
        self.dt = dt
        self.pose = C(0.0, 0.0, 0.0)
        self.vx = self.wz = 0.0
        self.fault = None          # None | 'freeze' | 'die' | 'raise'
        self.t0 = time.time()
        self._stop = threading.Event()
        self.source.feed(self.pose)        # ★ 先同步喂一帧：对齐时必须有数据
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.dt)
            if self.fault == 'die':
                continue                       # 位姿断流：不再喂
            if self.fault != 'freeze':
                last = self.sink.last()
                if last and last[0] == 'move':
                    self.vx, self.wz = last[2], last[3]
                else:
                    self.vx, self.wz = 0.0, 0.0
                c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
                self.pose = C(self.pose.x + self.vx * c * self.dt,
                              self.pose.y + self.vx * s * self.dt,
                              self.pose.yaw + self.wz * self.dt)
            if self.fault == 'raise':
                self.source._errs += 1         # 模拟"回调抛异常"
            if self.fault != 'die':
                self.source.feed(self.pose)

    def set_pose(self, p):
        self.pose = p

    def stop(self):
        self._stop.set()


def make_nav(dog, sink, grid=None, timeout=8.0):
    cfg = NavigateToConfig(pos_tol=0.05, yaw_tol_deg=15.0, timeout=timeout,
                           approach=ApproachConfig(timeout=timeout))
    return Navigator(dog.source, sink, cfg=cfg, grid=grid)


def run_nav(nav, gx, gy):
    """在后台线程跑 navigate_to，返回一个可取结果的句柄"""
    box = {}

    def _run():
        try:
            box['res'] = nav.navigate_to(
                Pose6D(gx, gy, 0.0, 0.0, 0.0, 0.0, frame=FRAME_SCENE))
        except Exception as e:
            box['exc'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return box, t


def wait_turning(sink, timeout=8.0):
    """★ 等跟随器/精调发出【转】的指令（|wz|>0）—— 确定它处于 turn/朝向阶段。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with sink.lock:
            for (k, _x, _vx, wz) in sink.hist:
                if k == 'move' and abs(wz) > 0.01:
                    return True
        time.sleep(0.05)
    return False


def wait_walking(sink, timeout=8.0):
    """★ 等跟随器真的发出【走】的指令（vx>0）—— 这样才确定它处于 walk 状态。

    为什么要这个：blocked/stuck 是【按状态】区分的（走不动=blocked、
    转不动=stuck）。靠 sleep 猜时机不可靠，直接等指令最稳。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        with sink.lock:
            for (k, _x, vx, _wz) in sink.hist:
                if k == 'move' and abs(vx) > 0.01:
                    return True
        time.sleep(0.05)
    return False


# ================================================================ ① 位姿断流
def t_stale_pose():
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    dog.set_pose(C(0.0, 0.0, 0.0))
    wait_pose(src)
    nav = make_nav(dog, sink)
    wait_pose(src)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box, t = run_nav(nav, 3.0, 0.0)
    time.sleep(0.8)
    dog.fault = 'die'                     # ← 注入：位姿断流
    t.join(timeout=15)
    dog.stop()
    res = box.get('res')
    assert res is not None, '没返回'
    assert sink.settled(), '★ 断流后没有 settle —— 狗会一直按最后一条指令走！'
    assert not res.success
    assert res.reason in ('stale', 'source_error', 'aborted', 'stuck'), res.reason
    print('         reason=%s，settle 已发' % res.reason)


# ================================================================ ② 回调抛异常
def t_callback_error():
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    nav = make_nav(dog, sink)
    wait_pose(src)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box, t = run_nav(nav, 3.0, 0.0)
    time.sleep(0.8)
    dog.fault = 'raise'                   # ← 注入：回调抛异常
    t.join(timeout=15)
    dog.stop()
    res = box.get('res')
    assert res is not None
    assert sink.settled(), '★ 回调抛异常后没有 settle（那天就是这样白走了 2.7m）'
    assert res.reason == 'source_error', res.reason
    print('         reason=%s，settle 已发' % res.reason)


# ================================================================ ③ 控制循环死
def t_watchdog_control_loop_dead():
    """★ 看门狗的核心场景：控制循环挂了，没人发指令，但狗还在动。"""
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    src.feed(C(0.0, 0.0, 0.0))            # 位姿源是好的
    wd = Watchdog(src, sink, max_stale=0.3, max_silence=0.3).armed()
    # 模拟：控制循环先正常跑一会
    for _ in range(5):
        wd.feed()
        time.sleep(0.05)
    assert wd.triggered() is None, '不该触发'
    time.sleep(0.6)                        # ← 注入：控制循环不再报到
    assert wd.triggered() is not None, '★ 控制循环死了，看门狗没发现'
    assert 'control_loop_dead' in wd.triggered(), wd.triggered()
    assert sink.settled(), '★ 看门狗触发了但没 settle'
    print('         %s，settle 已发' % wd.triggered())
    wd.stop()
    dog.stop()


def t_watchdog_stale_and_error():
    sink, src = RecSink(), MockSource()
    src.feed(C(0.0, 0.0, 0.0))
    wd = Watchdog(src, sink, max_stale=0.2, max_silence=10.0).armed()
    wd.feed()
    time.sleep(0.5)                        # ← 注入：不再有新位姿
    assert wd.triggered() and 'pose_stale' in wd.triggered(), wd.triggered()
    assert sink.settled()
    wd.stop()

    sink2, src2 = RecSink(), MockSource()
    src2.feed(C(0.0, 0.0, 0.0))
    src2._errs = 3                         # ← 注入：回调已经抛了 3 次
    wd2 = Watchdog(src2, sink2, max_stale=10.0, max_silence=10.0).armed()
    wd2.feed()
    time.sleep(0.3)
    assert wd2.triggered() and 'callback_error' in wd2.triggered(), wd2.triggered()
    assert sink2.settled()
    wd2.stop()


def t_watchdog_only_fires_once():
    """触发后不许反复发指令。"""
    sink, src = RecSink(), MockSource()
    wd = Watchdog(src, sink, max_stale=0.1, max_silence=0.1).armed()
    time.sleep(0.5)
    n1 = len(sink.hist)
    time.sleep(0.5)
    n2 = len(sink.hist)
    assert n1 == n2, '看门狗在反复发指令（%d → %d）' % (n1, n2)
    assert sink.settled()
    assert sink.moves_after_settle() == 0, 'settle 之后还有 move'
    wd.stop()


# ================================================================ ④ 无进展
def t_no_progress():
    """★★ 2026-09-28 改：走不动 ⇒ 必须报 blocked（不是 stuck）

    为什么这条测试现在要卡死 reason：
      组2 的 SPAV 状态机【只在 reason=='blocked' 时】走"绕行"分支，
      拿到别的会直接安全熔断退出整个任务。
      真机上"走不动"= 前面有东西挡着 ⇒ 语义上就是 blocked。
      （之前我们把它报成 stuck，而 blocked 【从来没被产生过】）
    """
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    nav = make_nav(dog, sink)
    wait_pose(src)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box, t = run_nav(nav, 3.0, 0.0)
    assert wait_walking(sink), '没等到"走"的指令 —— 用例本身有问题'
    dog.fault = 'freeze'                   # ← 注入：狗动不了了（前面被挡住）
    t.join(timeout=20)
    dog.stop()
    res = box.get('res')
    assert res is not None
    assert sink.settled(), '没 settle'
    assert not res.success
    assert res.reason == 'blocked', \
        '★ 走不动应报 blocked（组2 靠它触发绕行），实际 %r' % res.reason
    print('         reason=%s ✅（走不动 ⇒ blocked）' % res.reason)


def t_stuck_when_turning():
    """★ 2026-09-28 新增：转不动 ⇒ 必须报 stuck（和 blocked 分开）

    区分依据：走的状态下不动 = 前方有东西（blocked，等一等/绕行能过）；
              转的状态下不动 = 姿态卡死/打滑（stuck，要调整姿态）。
    做法：位置就在原地（距离 0），但【目标朝向差 90°】⇒ 一上来就在转。
    """
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    nav = make_nav(dog, sink)
    wait_pose(src)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box = {}
    tgt = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, math.radians(90), frame=FRAME_SCENE)

    def _run():
        try:
            box['res'] = nav.navigate_to(tgt)
        except Exception as e:
            box['exc'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert wait_turning(sink), '没等到"转"的指令 —— 用例本身有问题'
    dog.fault = 'freeze'                   # ← 注入：转不动了
    t.join(timeout=20)
    dog.stop()
    res = box.get('res')
    assert res is not None, 'navigate_to 没返回'
    assert sink.settled(), '没 settle'
    assert not res.success, '转不动却报成功了？'
    assert res.reason == 'stuck', \
        '★ 转不动应报 stuck，实际 %r' % res.reason
    print('         reason=%s ✅（转不动 ⇒ stuck）' % res.reason)


# ================================================================ ⑤ 超时
def t_timeout():
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    # 目标放得很远，8 秒走不到
    nav = make_nav(dog, sink, timeout=3.0)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box, t = run_nav(nav, 50.0, 0.0)
    t.join(timeout=25)
    dog.stop()
    res = box.get('res')
    assert res is not None
    assert sink.settled(), '没 settle'
    assert not res.success
    assert res.reason in ('timeout', 'aborted'), res.reason
    print('         reason=%s，settle 已发' % res.reason)


# ================================================================ ⑥ 外部急停
def t_abort():
    sink, src = RecSink(), MockSource()
    dog = FakeDog(sink, src)
    nav = make_nav(dog, sink, timeout=30.0)
    nav.align(Pose6D(0, 0, 0, 0, 0, 0, frame=FRAME_SCENE))
    box, t = run_nav(nav, 10.0, 0.0)
    time.sleep(0.8)
    nav.abort()                            # ← 注入：遥控器/安全层接管
    t.join(timeout=15)
    dog.stop()
    res = box.get('res')
    assert res is not None
    assert sink.settled(), '★ 急停后没有 settle —— 腿可能冻在抬腿相然后倒下'
    assert not res.success and res.reason == 'aborted', res.reason
    print('         reason=%s，settle 已发' % res.reason)


# ================================================================ ⑦ 速度越界
def t_speed_guard():
    """出口这一道门必须拦住荒唐的速度（不是截断 —— 截断会掩盖上游 bug）。"""
    sink = DryRunSink(verbose=False)
    try:
        sink.move(Velocity(vx=5.0))
        raise AssertionError('没拦住 5 m/s')
    except CommandRejected:
        pass
    try:
        sink.move(Velocity(vx=float('nan')))
        raise AssertionError('没拦住 NaN')
    except CommandRejected:
        pass
    sink.move(Velocity(vx=0.5, wz=0.4))    # 正常值应该放行
    assert sink._n_move >= 1


# ================================================================ ⑧ 控制权
def t_authority():
    a = Authority()
    assert a.request(Authority.NAV)
    assert a.may_move(Authority.NAV)
    assert not a.may_move(Authority.TELEOP)
    # ★ 遥控器永远可以抢占导航
    assert a.request(Authority.TELEOP), '遥控器抢不动导航的控制权！'
    assert a.may_move(Authority.TELEOP)
    assert not a.may_move(Authority.NAV)
    a.release(Authority.TELEOP)
    assert a.who == Authority.IDLE
    # 安全层也能抢
    a.request(Authority.NAV)
    assert a.request(Authority.SAFETY)
    assert a.may_move(Authority.SAFETY)


def main():
    print('=' * 74)
    print('  P3 验收：故障注入（每个都对应一次真实事故或险情）')
    print('=' * 74)
    groups = [
        ('① 位姿断流', [('断流后必须 settle', t_stale_pose)]),
        ('② 回调抛异常', [('★ 那天真发生：必须 settle', t_callback_error)]),
        ('③ 控制循环死（看门狗）',
         [('控制循环不报到了 → 触发', t_watchdog_control_loop_dead),
          ('位姿断流 / 回调异常也能触发', t_watchdog_stale_and_error),
          ('触发后只发一次，不再乱动', t_watchdog_only_fires_once)]),
        ('④ 无进展 —— ★ 走不动=blocked / 转不动=stuck（2026-09-28 分开）',
         [('走不动 → blocked（组2 靠它绕行）', t_no_progress),
          ('转不动 → stuck（要调整姿态）', t_stuck_when_turning)]),
        ('⑤ 超时', [('到不了 → 停', t_timeout)]),
        ('⑥ 外部急停', [('遥控器接管 → 立即 settle', t_abort)]),
        ('⑦ 速度越界', [('荒唐的速度必须拒绝（不是截断）', t_speed_guard)]),
        ('⑧ 控制权', [('遥控器/安全层能抢占导航', t_authority)]),
    ]
    for title, fns in groups:
        print()
        print('-- %s' % title)
        for name, fn in fns:
            check(name, fn)

    print()
    print('=' * 74)
    print('  通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for n, e in FAIL:
            print('    ✗ %s' % n)
        print('=' * 74)
        return 1
    print('  ✅ P3 故障注入全部通过 —— 每一种故障都能安全停')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
