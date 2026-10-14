# -*- coding: utf-8 -*-
"""
control/path_follower.py —— 沿路径走（★ 足式：转向/前进解耦）

为什么是这个范式
----------------
2026-09-22 我抄了【轮式车】的纯跟踪（pure pursuit）来跟路径。后果：
  · 出发时偏 64° 还全速走 → 斜着冲，画弧甩出 1.5m → 撞障碍
  · 拐弯时抄近道 → 切进障碍（34 个采样点落在障碍格里）
  · 路径跟踪偏差 ~20cm → 避障安全裕度被吃掉一半

根因不是参数，是**范式选错了**：
    纯跟踪是给【不能原地转】的轮式车设计的 —— "边走边转"对它是刚需。
    ★ 我们的狗能原地转、还能横移。白扔掉自由度。

★ 足式的做法（也是 UniGoal `execute_turn_then_forward` 在用的）：
    「转一个角度 + 走一段距离」—— 迈步本来就是离散的，这天然是常态

怎么做的
--------
    对路径上的每一个路点:
        ① 【转】  |朝向误差| > align_tol  →  vx = 0, wz = ±max(k|err|, wz_min)
        ② 【停】  对准后停 settle_s 秒（★ 抄 UniGoal：转完先停稳再走）
        ③ 【走】  vx = v_cruise, wz = 0（不修 —— 见下），
                  直到【到达或越过】当前路点 → 切下一个
        ④ 距终点 < fine_range 就交给末端精调

★★ 2026-09-23 真机事故后改掉的两处（都写在 FollowerConfig 的注释里）
   ① 走阶段的退出判据：方位角 ang → 沿路段进度 along + 横向偏移 lat
      为什么：ang 是到【当前路点】的张角，离路点越近越敏感
        （离 0.50m 横偏 4cm → 4.6°；离 0.05m 横偏 4cm → 38.7°）
      ⇒ 每次快走到路点都被误判"偏太多"→ 拉回去重转 → 而这时【还没越过路点】
      ⇒ i_wp 不推进 ⇒ 死循环 ⇒ 真机上走 1.41m 净进 0.17m、净转 +73°
   ② settle_s 0.25 → 0.80
      实测【转向】发零速后要 ~1.4s 才停稳（平移只要 0.6s），
      原来 0.25s 太短 ⇒ 狗还在转就切直走 ⇒ 带着转弯余速边走边转

★ 为什么走的时候 wz = 0
    因为我们有死区：wz < 0.2 根本不抬脚。"轻轻修一点"做不到 ——
    要么不给（会漂），要么给够（0.4，那就走成蛇形了）。
    ⇒ 解法不是连续修正，而是【把路点排密】：
      路点间距 ~0.5m，狗每米天然左偏 ~5cm ⇒ 每段只漂 2.5cm，
      下一个路点重新对准就纠回来了。

★ 转向为什么不能照抄"线性增益"
    UniGoal 用【标定表】而不是增益：
      15°→0.361s  30°→0.670s  45°→0.970s  60°→1.280s  90°→1.900s
    因为转向不线性（有死区、有启动）。我们做闭环所以不需要表，
    但 k_yaw 和 wz_min 必须真机标定，现在给的是初值。
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from core.types import Pose2D, Velocity, ZERO_VEL, wrap
from planning.global_planner import Plan
from planning.local_avoid import (AvoidConfig, corridor_polyline,
                                  should_stop)


# ================================================================ 配置
@dataclass(frozen=True)
class FollowerConfig:
    # ---- 运动 ----
    # ★ 标记含义见记忆 magic-numbers-rule：✅实测 / ⚠️部分依据 / ★真机调 / ★拍★
    v_cruise: float = 0.50          # ✅实测：真机 0.5 m/s 跑过，±5%
    # ★ 转到 ≤ 这个角就不转了。
    #   ✅实测依据（2026-09-26 真机）：5.0 太松 —— 狗是从一边逼近的，每次都停在
    #   【容差边界】而不是中心，所以是【系统性偏 5°】而不是随机误差：
    #       跑 (1,1,1.57)：整条轨迹都以 40° 在走（目标 45°），末端净转 85°（目标 90°）
    #   而 4×90° 闭环转向实验（容差 3°）实际残差是 −1.23/−3.46/−2.25/−0.15°
    #   ⇒ 狗本身能对到 1~3°，2° 是够得着的
    #   代价：每段多转一点时间（2026-09-26 实测 2m 只花 8.8s，吃得下）
    align_tol_deg: float = 2.0
    # ---- ⚠️ reenter_deg 已【废弃不用】（2026-09-23 真机事故后改掉）----
    #   它曾经是"走的时候方位角超过这个值就回去重转"的滞回判据。
    #   ★ 真机发现它是错的：方位角 ang 是【到当前路点】的张角，
    #     离路点越近越敏感 —— 离 0.05m 时横向偏 4cm 就是 39°，
    #     于是每次快走到路点时都被误判"偏太多"拉回去转，
    #     而这时【还没越过路点】⇒ i_wp 不推进 ⇒ 死循环 ⇒ 狗绕圈。
    #     （2026-09-23 真机：走 1.41m 净进 0.17m、净转 +73°、i_wp 全程停在 1）
    #   ⇒ 已换成【横向偏移 max_lateral】，见下。
    reenter_deg: float = 30.0       # 【已废弃，保留字段只为兼容旧配置】
    # ★★ 新的走阶段偏离保护：狗离【当前路段】的垂直距离超过它就回去重转。
    #   为什么用它而不是方位角：横向偏移【不会】被短距离放大，量纲稳定。
    #   定值依据：规划膨胀 0.35m（路径本身就贴着这个边界走）
    #            狗自然走偏 ~3.6°/m × 0.5m 路点 = 2.3cm
    #            ⇒ 0.30 留了 10 倍余量，又小于规划膨胀
    max_lateral: float = 0.30       # 【仅诊断用，不参与控制】见下面 guard_far_m
    # ★★ 守卫的【生效距离】：只有离当前路点比这个还远时，才用 ang 判"偏太多"。
    #   靠近路点后 ang 会被几何放大（离 0.05m 横偏 4cm = 38.7°），是假警报。
    #   定值依据：路点间距 0.5m（densify 的 max_seg）⇒ 0.15 大约是它的 1/3，
    #             够远到 ang 还可信，又近到能覆盖"最后一段"。
    guard_far_m: float = 0.15
    # ★★ 2026-09-26：把"对准"的瞄准点从【当前路点】改成【再往前 lookahead_wp 个路点】。
    #   为什么必须改：判据 ang 是"到瞄准点的方位角"，它同时混了【朝向误差】和
    #   【位置误差】。瞄准点越近，位置误差被放大得越狠：
    #       横偏 4cm，瞄准点 0.05m 远 → 方位角差 38.7°  ← 位置误差完全盖过朝向
    #       横偏 4cm，瞄准点 0.40m 远 → 5.7°
    #       横偏 4cm，瞄准点 0.80m 远 → 2.9°
    #   ⇒ 真机实测到的病：日志说"偏差 −7°（以为对准了）"，狗实际歪 64°
    #     ⇒ 跟随器带着 64° 的歪朝向往前走 = 用户看到的"横着走"
    #   ⇒ 瞄准点往后挪一个路点（0.5m 间距 ⇒ 至少 0.4m 远），噪声被压一个数量级
    #   ★ 为什么不做"近了才挪"那种条件式：跨阈值时瞄准点会跳变，ang 跟着跳。
    #     一直瞄下一个则连续、无突变。
    #   ★ 瞄准点仍然是【路径上的点】⇒ 有横偏时会自然指回路径，收敛不丢。
    #
    # ★★ 2026-09-26 晚【再次修正】：上面那个"瞄 i_wp+1 那个路点"的做法
    #    在【拐角处会抄近道】—— 实测窄口场景里，狗在奔路点5 时被提前 33° 拉向路点6，
    #    然后到路点6 又要转 40° ⇒ 在 0.77m 的窄缝里转了 60° ⇒ 机身尾部扫到墙 ⇒ 卡死。
    #    （真机：终点卡在缝里 (1.62,1.21)，3 秒只挪 1.3cm）
    #    ⇒ 改成【沿路径往前滑 fixed 距离】取瞄准点：
    #        把狗投影到当前路段 → 从投影点【沿路径】走 lookahead_m → 那个点就是瞄准点
    #        （走过拐角就沿下一段继续走）
    #      这样：① 瞄准点永远在路径上，拐角处不会跳变
    #            ② 拐弯提前 lookahead_m 开始、而且连续 ⇒ 不会在窄处突然转大角度
    #            ③ 距离仍 ≥ ~lookahead_m ⇒ 位置噪声仍被压住
    lookahead_m: float = 0.50
    lookahead_wp: int = 1           # 【已废弃，保留字段只为兼容旧配置】
    stop_tol: float = 0.05          # ★真机调：到路点多近算到达（用户定 5cm，宽松点）
    fine_range: float = 0.20        # ★真机调：距终点多近交给末端精调
                                    # ★★ 2026-10-15：0.50 → 0.20（用户定）
                                    #   为什么：末端精调是【直着朝目标走】、不沿路径
                                    #   ⇒ 交接得越早，"抄近道"的窗口越大：
                                    #     最后 0.5m 里如果路径拐了弯（比如绕直角桌），
                                    #     直走就会从内侧切过去 ⇒ 可能蹭到
                                    #   ⇒ 0.2m 时窗口小得多（目标点本身是空的，
                                    #     0.2m 内直着走几乎不可能出事）
                                    #   代价：跟随器多走 0.3m（逐路点转走，慢一点）
                                    #   ★ 同时它决定"预瞄点 d 的下限"：
                                    #     d 必须 > fine_range + stop_tol ⇒ 建议 ≥0.3
    # ★ 转完停多久。2026-09-23 真机实测【转向】发零速后要 ~1.4s 才停稳
    #   （还有一次过冲 −4.9°/s）；平移只要 0.6s。
    #   原来 0.25s（抄 UniGoal）⇒ 狗还在转就切"直走"⇒ 带着转弯余速边走边转。
    settle_s: float = 0.80
    # ---- 转向 ----
    # ★★ 2026-10-08 真机实测后上调（原来 1.2 / 0.40 / 0.80）
    #
    #   证据（当天做的对照实验）：
    #     · 遥控器原地转 90° ⇒ CAPO 准确测到 91.3°，角速度 ~0.75 rad/s
    #       ⇒ 狗本身能转得快，问题不在狗、也不在 CAPO
    #     · 我们发 wz=0.40 时，狗【一跳一跳地几乎没转】，
    #       3 秒只转 9.9°（≈0.06 rad/s，只有遥控的 1/12）
    #       ⇒ 用户现场描述"一下一下的，有点跳跳的感觉"
    #     · 而实测效率表（2026-09-23）：0.4→25%、0.6→52%、1.0→76%
    #   ⇒ 0.40 这个"死区之上"其实【远不够】——它在死区边上，狗只能蹦不能转
    #
    #   ★ 历史：9-28 联调时"转了 3.1 秒只转 10°"也是同一个毛病，
    #     只是那次偏差小、没致命；这次遇到 22.8° 的偏差就判"姿态卡死"了。
    # ★★ 2026-10-08 当天第二次调整：从 (2.0/0.60/1.00) 退到 (1.5/0.60/0.80)
    #   为什么退：激进版虽然"转得快 6 倍"，但【离线回归 P1 精度从 0.0145 翻到 0.0280】，
    #   真机上更是走歪（转身把位置挪偏后，一路叠着偏）。
    #   现在：wz_min 保留 0.60（这是"能转起来"的下限，实测 0.4 只有 25% 效率），
    #         但 wz_max 和 k_yaw 退回来，别冲太快。
    k_yaw: float = 1.5              # ★真机调：角度误差 → 角速度
    wz_min: float = 0.60            # ✅实测：0.4→25% 效率（几乎转不动）、0.6→52%
    wz_max: float = 0.80            # ★ 退回来：冲太快会把位置带偏
    # ---- 控制周期 ----
    control_hz: float = 50.0        # ★要实测：不假设 time.sleep 准（Python 有抖动）
    # ---- 失效保护 ----
    timeout: float = 90.0           # ★拍★ 分段上限（任务级预算在 NavigateToConfig）
    no_progress_s: float = 3.0      # ★真机调
    # ★ 卡死检测：走和转【各用各的指标】（★ 2026-10-15 拆开）
    #   · 走（state='walk'）⇒ 只看【位置】：3 秒挪不到 3cm ⇒ blocked
    #     （角度不参与 —— 推着东西时身体会晃，别把 blocked 晃没了）
    #   · 转（state='turn'）⇒ 只看【角度】：3 秒转不到 3° ⇒ stuck
    #     ★ 原来是 10°（统一判据，2026-09-22 定）——
    #       会把"转得慢"误判成卡死：实测 wz=0.4 时 3 秒转 9.9° < 10° ⇒ 冤判
    #     ⇒ 2026-10-15 用户定：转向判据 10 → 3（"几乎没动"才叫卡死）
    no_progress_min_m: float = 0.03
    no_progress_min_deg: float = 3.0

    # ★★ 2026-10-08 加：避障等待的【上限】
    #   为什么需要：等待期间会【重置】无进展窗口（设计如此 —— 见主循环注释，
    #   目的是让"停着等人走过"不被误判）。但后果是【障碍一直不清】时会
    #   一路等到总超时（90s）⇒ 报 timeout
    #   ⇒ 组2 的 SPAV 【只在 reason=='blocked' 时】绕行，拿到 timeout
    #     会走"安全熔断"直接终止整个任务
    #   ⇒ 所以：等够这么久还堵着 ⇒ 报 blocked，交给上层绕行
    #   ★ 7.0 是用户 2026-10-08 定的（给人走过去留够时间，又别让组2 等太久）
    avoid_blocked_s: float = 7.0
    stale_s: float = 0.30           # ★拍★


DEFAULT_FOLLOWER = FollowerConfig()


@dataclass
class FollowResult:
    success: bool
    reason: str            # handed_off / timeout / stale / blocked / stuck / aborted / off_path
    final_pose: Optional[Pose2D]
    elapsed: float
    n_cycles: int
    traveled: float
    n_wp: int = 0          # 走过了几个路点


# ================================================================ 主类
class PathFollower:
    def __init__(self, cfg: FollowerConfig, source, sink,
                 log: Optional[Callable[[str], None]] = None,
                 should_abort: Optional[Callable[[], bool]] = None,
                 on_cycle: Optional[Callable[[], None]] = None,
                 grid=None, inflate_radius: float = 0.35,
                 safety_margin: float = 0.20,
                 avoid: Optional[AvoidConfig] = None,
                 obstacle_source=None):
        self.cfg = cfg
        self.source = source
        self.sink = sink
        self.log = log or (lambda s: None)
        self.should_abort = should_abort or (lambda: False)
        self.on_cycle = on_cycle or (lambda: None)   # ★ 给看门狗报到
        self.grid = grid                # 可选：用来发现"快撞上了"
        # ★★ 两个半径是【两回事】，不能混：
        #   inflate_radius = 0.35  规划时留给【跟踪误差】的余量
        #                          ⇒ 路径会【贴着这个边界走】（平滑走切线）
        #   safety_margin  = 0.20  ★⚠️ 这是【半宽】量级（GO2 半宽约 0.16m）
        #                          ⇒ 是"你开始偏了"的报警线，不是"要撞了"
        #                          ⇒ 【不是】最坏情况的扫掠半径 ——
        #                            车身横过来时扫掠半径是【半长~0.35m】
        #                            （用户 2026-09-22 提醒："半长还是会长一点的"）
        #                          ⇒ 真正的碰撞保护靠规划膨胀 0.35m 兜底
        #   混用会怎样：路径贴 0.35 边界 + 狗偏 1.6cm ⇒ 一出发就误报
        #   （2026-09-22 解耦版跟随器第一次跑就撞上这个）
        self.inflate_radius = inflate_radius
        self.safety_margin = safety_margin
        self._gi = None
        self.avoid = avoid or AvoidConfig()
        self.obstacle_source = obstacle_source
        self._avoid_hold_until = 0.0
        self._avoid_waiting = False
        self._avoid_reason = None
        self._avoid_since = None      # ★ 避障等待开始时刻（用于 avoid_blocked_s）

        # ★★ 守门：走廊半宽【必须小于】规划膨胀半径
        #   否则贴着路径的【静态】障碍（规划只保证 ≥inflate_radius）会被当成
        #   "新出现的东西" ⇒ 一出发就误停。
        #   2026-09-28 实测：corridor 0.40 vs 膨胀 0.35 ⇒ 墙在 0.30~0.36 时全误停。
        #   （tools/check_corridor_vs_inflate.py 是这个结论的实验）
        if self.avoid.enabled and self.avoid.corridor_m >= self.inflate_radius:
            self.log('⚠️⚠️ 动态避障的 corridor_m(%.2f) ≥ 规划膨胀(%.2f) —— '
                     '会把贴着路径的【静态】障碍误判成动态障碍，一出发就停！'
                     % (self.avoid.corridor_m, self.inflate_radius))

    # ------------------------------------------------------------
    @staticmethod
    def _aim_point(pose, wps, goal, i_wp, L):
        """★ 沿路径取瞄准点：狗投影到当前路段 → 从投影点沿路径走 L → 那个点。

        为什么要"沿路径走"而不是"瞄下一个路点"：
          瞄下一个路点在【拐角处会抄近道】—— 实测窄口场景里狗被提前 33° 拉向
          下一段，到拐角又要转 40°，两次叠加在 0.77m 的缝里转了 60° ⇒ 尾部扫墙卡死。
          沿路径滑动的瞄准点则【连续、不跳变】，拐弯提前 L 开始、缓缓转过去。

        ★ 最多只看【一段半】（当前段 + 下一段），够用且好懂。
        """
        n = len(wps)
        a = wps[max(0, i_wp - 1)]
        b = goal if i_wp >= n - 1 else wps[i_wp]
        seg = math.hypot(b.x - a.x, b.y - a.y)
        if seg < 1e-6:
            return b.x, b.y
        ux, uy = (b.x - a.x) / seg, (b.y - a.y) / seg
        # 狗在本段上的投影弧长（夹在 [0, seg]）
        s = (pose.x - a.x) * ux + (pose.y - a.y) * uy
        s = min(max(s, 0.0), seg)
        L = max(L, 0.05)
        rem = seg - s
        if L <= rem:
            return a.x + (s + L) * ux, a.y + (s + L) * uy
        # ---- 跨到下一段 ----
        if i_wp >= n - 1:
            return b.x, b.y                     # 已是最后一段，就到本段终点
        c = goal if i_wp + 1 >= n - 1 else wps[i_wp + 1]
        seg2 = math.hypot(c.x - b.x, c.y - b.y)
        if seg2 < 1e-6:
            return c.x, c.y
        ux2, uy2 = (c.x - b.x) / seg2, (c.y - b.y) / seg2
        over = min(L - rem, seg2)
        return b.x + over * ux2, b.y + over * uy2

    def _emit(self, v: Velocity, state: str):
        """★ 发指令 + 【指令变化时打一行日志】。

        2026-09-26 加：真机跑 2m 时，狗在【走】状态（指令 wz=0）下自己转了 33°，
        但每秒一行的日志里看不到中间发了什么 —— 分不清是狗自己的问题还是
        状态机在快速切换。所以：只在【指令发生变化】时打一行，
        既不刷屏，又能拿到完整的时间线。
        """
        key = (round(v.vx, 3), round(v.vy, 3), round(v.wz, 3))
        if key != self._last_cmd:
            self._last_cmd = key
            self.log('  ⌘ t=%5.2f %s  vx=%+.2f vy=%+.2f wz=%+.2f'
                     % (time.time() - self._t0,
                        '走' if state == 'walk' else '转',
                        v.vx, v.vy, v.wz))
        self.sink.move(v)

    def _emit_with_avoid(self, v: Velocity, state: str, pose: Pose2D,
                         wps, i_wp: int, goal: Pose2D, now: float) -> bool:
        "在唯一的发送点前让动态层覆盖命令；返回本周期是否停车。"
        if not self.avoid.enabled:
            self._emit(v, state)
            return False

        polyline = corridor_polyline(wps, pose, i_wp, goal,
                                     self.avoid.lookahead_m)
        try:
            healthy = (self.obstacle_source is not None
                       and self.obstacle_source.healthy(max_age=1.0))
            pts = self.obstacle_source.get() if healthy else None
        except Exception:
            healthy, pts = False, None
        sensor_failed = not healthy or pts is None
        stop_now, count = should_stop(pts, polyline, self.avoid)
        if stop_now:
            self._avoid_hold_until = max(self._avoid_hold_until,
                                         now + max(0.0, self.avoid.hold_s))
        waiting = stop_now or now < self._avoid_hold_until
        if waiting:
            if not self._avoid_waiting:
                self._avoid_since = now      # ★ 刚开始等，记下时刻
            if sensor_failed:
                if self._avoid_reason != 'sensor_failed':
                    self.log('  SENSOR FAILURE: obstacle data unavailable; stop and wait')
            elif not self._avoid_waiting:
                self.log('  🚧 路径前方受阻（走廊内 %d 点），停止等待' % count)
            self._avoid_waiting = True
            self._avoid_reason = (
                'sensor_failed' if sensor_failed else
                ('obstacle' if stop_now else self._avoid_reason))
            self._emit(ZERO_VEL, state)
            return True

        if self._avoid_waiting:
            self.log('  ✅ 路径前方已清空，继续行走')
        self._avoid_waiting = False
        self._avoid_reason = None
        self._avoid_since = None          # ★ 等待结束，清计时
        self._emit(v, state)
        return False

    def follow(self, plan: Plan, goal: Pose2D) -> FollowResult:
        cfg = self.cfg
        t0 = time.time()
        self._t0 = t0
        self._last_cmd = None
        self._avoid_hold_until = 0.0
        self._avoid_waiting = False
        self._avoid_reason = None
        self._avoid_since = None
        tp = self.source.get()
        if tp is None:
            return FollowResult(False, 'no_pose', None, 0.0, 0, 0.0)
        start = tp.data

        wps = list(plan.waypoints)
        if not wps:
            return FollowResult(False, 'no_path', start, 0.0, 0, 0.0)

        traveled = 0.0
        last = start
        n = 0
        n_wp = 0
        reason = 'unknown'
        i_wp = 1                       # 路点 0 是起点，从下一个开始
        state = 'turn'                 # 'turn' | 'walk'
        settle_until = 0.0
        # ★ 卡死检测：走和转各用各的指标（★ 2026-10-15 拆开，见 FollowerConfig）
        #   ★ 进展窗口【跨状态切换时重置】—— 否则会冤判：
        #     "走了 3 秒、刚切到转" ⇒ 窗口里 yaw 几乎没变（走路本来就不修朝向）
        #     ⇒ 一进 turn 就被误报"转不动"
        #     （10-15 全量回归里真踩到：走廊用例被误报 stuck）
        chk_t = t0
        chk_pose = start
        chk_state = state

        try:
            while True:
                now = time.time()
                n += 1
                self.on_cycle()

                # ---- 失效保护 ----
                if self.should_abort():
                    reason = 'aborted'
                    break
                if self.source.error_count() > 0:
                    reason = 'source_callback_error'
                    break
                tp = self.source.get()
                if tp is None or (now - tp.recv_t) > cfg.stale_s:
                    reason = 'stale_pose'
                    break
                pose = tp.data
                traveled += math.hypot(pose.x - last.x, pose.y - last.y)
                last = pose

                if now - t0 > cfg.timeout:
                    reason = 'timeout'
                    break

                d_goal = math.hypot(goal.x - pose.x, goal.y - pose.y)
                if d_goal <= cfg.fine_range:
                    reason = 'handed_off'
                    break

                # 动态停车不属于“走不动”。等待期间重置原有无进展窗口，
                # 否则障碍停留 3 秒会让 follow() 退出，违背“停着等”的语义。
                if self._avoid_waiting:
                    chk_t, chk_pose = now, pose
                    # ★★ 2026-10-08：但【不能无限等】。
                    #   等够 avoid_blocked_s 还堵着 ⇒ 报 blocked，交给上层绕行。
                    #   为什么必须报 blocked（而不是让它耗到总超时）：
                    #     组2 的 SPAV 【只在 reason=='blocked' 时】走绕行分支，
                    #     拿到 timeout 会走"安全熔断"直接终止整个任务
                    #     （2026-10-08 真机实测：等 90s 报 timeout ⇒ 他们的自愈没触发）
                    if (self._avoid_since is not None
                            and now - self._avoid_since > cfg.avoid_blocked_s):
                        reason = 'blocked'
                        self.log('  🚧 避障等待 %.1fs、障碍一直没清'
                                 ' ⇒ blocked（交给上层绕行）' % cfg.avoid_blocked_s)
                        break

                # ★ 状态一切换（走↔转），进展窗口重新计（见上面 chk_state 的注释）
                if state != chk_state:
                    chk_t, chk_pose, chk_state = now, pose, state

                # ---- 无进展：走和转【各用各的指标】（★ 2026-10-15 拆开）----
                if not self._avoid_waiting and now - chk_t > cfg.no_progress_s:
                    dm = math.hypot(pose.x - chk_pose.x, pose.y - chk_pose.y)
                    da = abs(math.degrees(wrap(pose.yaw - chk_pose.yaw)))
                    walking = (state == 'walk')
                    no_prog = (dm < cfg.no_progress_min_m if walking
                               else da < cfg.no_progress_min_deg)
                    if no_prog:
                        # ★★ 2026-09-28：区分 blocked / stuck —— 下游处理【完全不同】
                        #   · 走的状态下走不动 ⇒ 前方【有东西挡着】⇒ blocked
                        #     真机上"走不动"绝大多数就是前面有障碍（人/椅子/箱子）
                        #     处理方式：等它挪开、或绕一下 ⇒ 是"等一等能过"
                        #   · 转的状态下转不动 ⇒ 【姿态卡死/打滑】⇒ stuck
                        #     处理方式：调整姿态或换目标点 ⇒ 不是"等一等"能解决的
                        #   ★ 为什么必须分清：组2 的 SPAV 状态机【只在 reason=='blocked'
                        #     时】走"绕行"分支，拿到别的会直接安全熔断退出整个任务
                        #     （他们的核心演示"推椅子→自动绕行"就靠这一条）
                        if walking:
                            reason = 'blocked'
                            self.log('  🚧 前方受阻：%.1fs 内走不动（只挪 %.3fm）'
                                     ' ⇒ blocked（等一等 / 绕行可过）'
                                     % (cfg.no_progress_s, dm))
                        else:
                            reason = 'stuck'
                            self.log('  ⚠️ 姿态卡死：%.1fs 内转不动（只转 %.1f°）'
                                     ' ⇒ stuck（要调整姿态，不是等一等的事）'
                                     % (cfg.no_progress_s, da))
                        break
                    chk_t, chk_pose = now, pose

                # ---- 跑出安全走廊了（跟踪偏差太大）----
                if self.grid is not None and not self._on_path(pose):
                    reason = 'off_path'
                    self.log('  🚨 已跑出安全走廊（跟踪偏差）')
                    break

                # ---- 选当前目标路点（走过的/已经过头的 都跳过）----
                # ★ 到达判定有两条：
                #   ① 离得够近
                #   ② 已经【走过去了】（沿路段方向投影超过路点）
                #   只判 ①会死循环：狗沿机头方向走（可能偏 10°），
                #   会从路点旁边 8.7cm 处擦过去，永远不满足"<3cm"，
                #   于是越走越远、方位角甩开、再回去转 —— 卡在阈值上振。
                #   （2026-09-22 解耦版第一次跑就被这个咬住）
                while i_wp < len(wps) - 1:
                    wp_c = wps[i_wp]
                    d_c = math.hypot(pose.x - wp_c.x, pose.y - wp_c.y)
                    prev = wps[max(0, i_wp - 1)]
                    dx, dy = wp_c.x - prev.x, wp_c.y - prev.y
                    seg = math.hypot(dx, dy)
                    if seg > 1e-6:
                        ux, uy = dx / seg, dy / seg
                        along = (pose.x - wp_c.x) * ux + (pose.y - wp_c.y) * uy
                    else:
                        along = 0.0
                    if d_c < cfg.stop_tol or along > 0.0:
                        i_wp += 1
                        n_wp += 1
                    else:
                        break
                tgt = goal if i_wp >= len(wps) - 1 else wps[i_wp]

                # ---- 到达判据用【当前路点】tgt ----
                ex, ey = tgt.x - pose.x, tgt.y - pose.y
                dist = math.hypot(ex, ey)
                c, s = math.cos(pose.yaw), math.sin(pose.yaw)

                # ---- ★★ 朝向判据 ang 用【沿路径滑动的瞄准点】（见 _aim_point）----
                #   为什么不是"瞄下一个路点"：那样在拐角处会抄近道，
                #   实测窄口场景里被提前 33° 拉偏 ⇒ 到拐角又转 40° ⇒ 在缝里转 60° ⇒ 卡死。
                ax, ay = self._aim_point(pose, wps, goal, i_wp, cfg.lookahead_m)
                aex, aey = ax - pose.x, ay - pose.y
                afwd = aex * c + aey * s
                aleft = -aex * s + aey * c
                ang = math.atan2(aleft, afwd)
                aim_dist = math.hypot(aex, aey)

                # ★★ 沿路径的【进度】和【横向偏移】—— 走阶段用它俩当判据，不用 ang。
                #    ang 是到【当前路点】的张角：离路点越近，同样的横向偏移张角越大
                #      （离 0.50m 偏 4cm → 4.6°；离 0.05m 偏 4cm → 38.7°）
                #    这就是 2026-09-23 真机绕圈的根因：快走到路点时被误判"偏太多"，
                #    拉回去重转，而这时【还没越过路点】⇒ i_wp 不推进 ⇒ 死循环。
                #    下面这两个量【不会被短距离放大】，量纲稳定。
                prev_wp = wps[max(0, i_wp - 1)]
                sdx, sdy = tgt.x - prev_wp.x, tgt.y - prev_wp.y
                sseg = math.hypot(sdx, sdy)
                if sseg > 1e-6:
                    sux, suy = sdx / sseg, sdy / sseg
                    rel_x, rel_y = pose.x - tgt.x, pose.y - tgt.y
                    along = rel_x * sux + rel_y * suy      # >0 = 已越过当前路点
                    lat = abs(-rel_x * suy + rel_y * sux)  # 离路段的垂直距离
                else:
                    along, lat = 0.0, 0.0

                # ================= 两态：转 / 走 =================
                if state == 'turn':
                    if abs(math.degrees(ang)) <= cfg.align_tol_deg:
                        state = 'walk'
                        settle_until = now + cfg.settle_s
                        self._emit_with_avoid(ZERO_VEL, state, pose, wps,
                                              i_wp, goal, now)  # ★ 转完先停稳
                        self.log('    对准（路点 %d/%d）→ 停 %.2fs 再走'
                                 % (i_wp, max(1, len(wps) - 1), cfg.settle_s))
                    else:
                        wz = max(-cfg.wz_max, min(cfg.wz_max, cfg.k_yaw * ang))
                        if abs(wz) < cfg.wz_min:
                            wz = math.copysign(cfg.wz_min, ang)   # 死区之上
                        self._emit_with_avoid(Velocity(0.0, 0.0, wz), state,
                                              pose, wps, i_wp, goal, now)
                else:  # walk
                    # ★★ 走阶段的退出判据【不用方位角 ang】—— 见上面 along/lat 的说明，
                    #    以及 FollowerConfig 里 reenter_deg 的废弃说明。
                    #    用 along（沿路段进度）+ lat（横向偏移），两个都不会被短距离放大。
                    # ★★ 注意：【没有】"横偏太大就回去重转"这一条 ——
                    #    2026-09-23 试过 max_lateral 守卫，结果死锁：
                    #      横偏 0.307m > 阈值 0.30 ⇒ 转 → 对准(朝向对了但位置还偏) →
                    #      走 → 横偏还是 0.307 ⇒ 又转 …… 无限乒乓，狗原地不动
                    #    ★ 根因：【转向只能改朝向，改不了位置】。
                    #    自纠机制是【路点排密到 0.5m + 每到一个路点重新对准】——
                    #    只要一直走到越过路点，下一个路点自然把它拉回来。
                    #    近障碍的保护由 _on_path()（栅格）负责，不在这一层。
                    # ★★ 守卫只在【离路点还远】时管用（cfg.guard_far_m 之外）。
                    #    为什么：ang 是到【当前路点】的张角，离路点越近越敏感 ——
                    #      离 0.50m 横偏 4cm → 4.6°（真偏）
                    #      离 0.05m 横偏 4cm → 38.7°（假警报，其实马上就要走到了）
                    #    ⇒ 原代码就是被这个假警报反复拉回去重转，
                    #      而那时【还没越过路点】⇒ i_wp 不推进 ⇒ 真机绕圈。
                    #    ⇒ 靠近路点后就别再管 ang 了，让这一段走完。
                    if now < settle_until:
                        self._emit_with_avoid(ZERO_VEL, state, pose, wps,
                                              i_wp, goal, now)
                    elif dist <= cfg.stop_tol or along > 0.0:
                        i_wp += 1                       # ★ 到达 或 已越过 → 推进路点
                        n_wp += 1
                        state = 'turn'                  # 到下一个路点重新对准
                        self._emit_with_avoid(ZERO_VEL, state, pose, wps,
                                              i_wp, goal, now)
                    elif (dist > cfg.guard_far_m
                          and abs(math.degrees(ang)) > cfg.reenter_deg):
                        state = 'turn'                  # 远处就偏这么多 → 停下来重对准
                        self._emit_with_avoid(ZERO_VEL, state, pose, wps,
                                              i_wp, goal, now)
                    else:
                        # ★ wz = 0：死区让它没法"轻修"，靠"路点排密+每点重对准"
                        self._emit_with_avoid(
                            Velocity(vx=cfg.v_cruise, vy=0.0, wz=0.0), state,
                            pose, wps, i_wp, goal, now)

                # ★ 2026-09-26：周期日志加密到 0.5s，并加上【狗的实际朝向】。
                #   为什么：真机那次日志说"偏差从 -28° 收到 -8°"，但狗的朝向
                #   实际已经转到 -70° —— 两者脱节，光看偏差【看不出狗转了多少】。
                #   有了 yaw 就能一眼对上："日志以为的" vs "狗实际的"。
                if n % int(cfg.control_hz / 2) == 1:
                    self.log('    %s t=%5.1f 路点%d/%d 距终点%5.2fm '
                             '偏差%+6.1f° 瞄%4.2fm 横偏%5.3fm 朝向%+7.1f°'
                             % ('转' if state == 'turn' else '走',
                                now - t0, i_wp, max(1, len(wps) - 1), d_goal,
                                math.degrees(ang), aim_dist, lat,
                                math.degrees(pose.yaw)))
                time.sleep(1.0 / cfg.control_hz)
        except Exception as e:
            reason = 'exception: %s' % e
        finally:
            self.sink.move(ZERO_VEL)
            self.on_cycle()               # ★ 收尾期间照样报到（不是在"死"）
            time.sleep(0.05)
            self.sink.stop()

        ok = reason in ('handed_off',)
        return FollowResult(ok, reason, last, time.time() - t0, n, traveled, n_wp)

    # ------------------------------------------------------------
    def _g_safety(self):
        """报警用的栅格 —— 按【车体半宽】膨胀，不是按规划余量"""
        if self._gi is None:
            self._gi = self.grid.inflated(self.safety_margin)
        return self._gi

    def _on_path(self, pose: Pose2D) -> bool:
        """是不是离障碍太近了（按车体半宽判）。

        ★ 不是"有没有走在规划走廊里"—— 路径本来就贴着走廊边界，
          那个判据会一出发就误报。这里只问"再近就要撞上了吗"。
        """
        try:
            return self._g_safety().is_free_xy(pose.x, pose.y)
        except Exception:
            return True     # 查不了就不拦 —— 别让安全网自己变成故障源
