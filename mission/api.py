# -*- coding: utf-8 -*-
"""
mission/api.py —— ★ 唯一对外接口

组2 要的就是这个文件里的两个东西：
    navigate_to(...)  →  NavResult

★ 这个文件【不依赖 ROS】—— 纯逻辑。
   将来要改成 ROS2 Action，只需要在外面包一层（mission/ros_action.py），
   仿真、单测、真机共用同一个实现。

⚠️ 必须跟组2说清楚的一件事
--------------------------
我们最后一步是闭环的，但闭环是在【我们自己的里程计坐标系】里收敛的
    ⇒ 我们报的 final_error 恒等于 0（这是恒等式，不是测量）
    ⇒ **真实误差我们无从知道**，除非末端用场景图的物体做对准
所以 NavResult 里返回的是 uncertainty（估计），不是真实误差。
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.types import (Timed, Pose2D, Pose6D, Transform2D, wrap,
                        FRAME_SCENE, FRAME_CAPO)
from core.frames import build_T_scene_capo, require_frame, to_frame
from control.fine_approach import FineApproach, ApproachConfig, DEFAULT_CFG
from control.path_follower import PathFollower, FollowerConfig, DEFAULT_FOLLOWER
from planning.global_planner import plan_path
from planning.local_avoid import AvoidConfig
from maps.grid import OccupancyGrid
from mission.mission import Reason, Stage, is_retryable
from safety.watchdog import Watchdog, Authority


# ================================================================ 对外数据结构
@dataclass
class NavigateToConfig:
    pos_tol: float = 0.03            # 位置容差 [m]
    yaw_tol_deg: float = 3.0         # 朝向容差 [deg]
    timeout: float = 120.0           # ★组2 给的【总预算】[s]
    # ★ 分段上限（内部安全）：两个概念不能互相覆盖。
    #   之前 Navigator 拿 timeout 把子配置的 timeout 全覆盖了——
    #   那让子配置的默认值成了死代码（2026-09-22 用户发现）
    follow_timeout: float = 90.0
    approach_timeout: float = 30.0
    max_speed: Optional[float] = None  # 限速（比如端着东西时）
    approach: ApproachConfig = field(default_factory=lambda: DEFAULT_CFG)
    follower: FollowerConfig = field(default_factory=lambda: DEFAULT_FOLLOWER)
    dynamic_avoid: bool = False       # ★ 默认关；关时保持原有行为
    avoid: AvoidConfig = field(default_factory=AvoidConfig)
    # ★★ 预瞄点（turn_shift 根治，2026-10-15）：非空 = 开。
    #    在终点背后 d 米放预瞄点，让【最后一段路的方向 = 目标朝向】⇒ 不用"到了再大转"。
    #    可给阶梯（按顺序试），如 (0.8, 0.6)；全部不可行 ⇒ 退回普通规划。
    #    ★ d 必须 > follower.fine_range(0.2m) + stop_tol(0.05m) ⇒ 建议 ≥0.3，
    #      否则会被末端精调截胡。
    #    ★ 默认 () = 关 —— 先在狗上验证过再默认开（用户定的）。
    final_approach_lengths: tuple = ()
    inflate_radius: float = 0.35     # ★ 车体半径，必须 ≥ 车体半宽才留得出余量
    # 漂移估计用的假设（★ 只是估计，不是测量）
    drift_per_meter: float = 0.02    # 每米漂 2%


@dataclass
class NavResult:
    success: bool
    final_pose: Optional[Pose6D]     # frame='scene'（我们坐标系里的）
    reason: str                      # 见 mission.Reason
    elapsed: float
    uncertainty: float               # ⚠️ 估计，不是真实误差
    odometry_source: str
    internal_error: float            # 在我们自己坐标系里的到位误差（≈0，恒等式）
    stages: List[Tuple[str, float]] = field(default_factory=list)
    # ★ 2026-10-15：给组2 报"障碍在哪"（reason=blocked 时才有意义）
    blocked_distance: float = -1.0   # 最近障碍【相对机体】距离 [m]；<0 = 没有信息
    blocked_bearing: float = 0.0     # 最近障碍【相对机体】方位角 [rad]（逆时针为正）

    def summary(self) -> str:
        pose = (('(%.3f, %.3f, %.1f°)' % (self.final_pose.x, self.final_pose.y,
                                          math.degrees(self.final_pose.yaw)))
                if self.final_pose else 'None')
        return ('%s  reason=%-12s  用时 %.1fs  内部误差 %.4f m  '
                '不确定度 ±%.3f m  末位姿 %s'
                % ('✅ 成功' if self.success else '❌ 失败', self.reason,
                   self.elapsed, self.internal_error, self.uncertainty, pose))


# ================================================================ 内部：坐标系适配
class _ScenePoseSource:
    """把 capo 系的位姿源，包成 scene 系的位姿源（用固定的 T，之后不再变）。

    ★ T 只在开机对齐时建立一次。这里是它唯一的用武之地。
    """

    def __init__(self, inner, T: Transform2D):
        self.inner = inner
        self.T = T
        self.name = '%s→scene' % inner.name

    def get(self) -> Optional[Timed]:
        t = self.inner.get()
        if t is None:
            return None
        return Timed(recv_t=t.recv_t, stamp=t.stamp, data=self.T @ t.data)

    def healthy(self, max_age: float = 0.3) -> bool:
        return self.inner.healthy(max_age)

    def error_count(self) -> int:
        return self.inner.error_count()


class _SceneObstacleSource:
    """Transform CAPO-frame obstacle points into the fixed scene frame."""

    def __init__(self, inner, T: Transform2D):
        self.inner = inner
        self.T = T
        self.name = '%s->scene' % getattr(inner, 'name', 'obstacles')

    def get(self) -> Optional[np.ndarray]:
        p = self.inner.get()
        if p is None:
            return None
        p = np.asarray(p, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('obstacle points must have shape (N,3), got %r'
                             % (p.shape,))
        c, s = math.cos(self.T.yaw), math.sin(self.T.yaw)
        return np.stack([
            c * p[:, 0] - s * p[:, 1] + self.T.x,
            s * p[:, 0] + c * p[:, 1] + self.T.y,
            p[:, 2],
        ], axis=1)

    def healthy(self, max_age: float = 1.0) -> bool:
        return self.inner.healthy(max_age)

    def error_count(self) -> int:
        return self.inner.error_count()


# ================================================================ 导航器
class Navigator:
    """导航器。持有：位姿源（capo 系）、指令出口、坐标系对齐 T。"""

    def __init__(self, source, sink, cfg: Optional[NavigateToConfig] = None,
                 grid: Optional[OccupancyGrid] = None,
                 log: Optional[Callable[[str], None]] = None,
                 obstacle_source=None):
        self.source = source
        self.sink = sink
        self.cfg = cfg or NavigateToConfig()
        self.grid = grid              # None ⇒ 不做全局规划，直接直线过去
        self.log = log or (lambda s: None)
        self.obstacle_source = obstacle_source
        if self.cfg.dynamic_avoid and obstacle_source is None:
            self.log('SENSOR FAILURE: dynamic avoidance has no obstacle source; motion will stop')
        self.T: Optional[Transform2D] = None
        self._align_capo0: Optional[Pose2D] = None
        self._align_start: Optional[Pose2D] = None   # ★ 对齐时用的"起点（场景系）"
        self._n_align = 0                            # ★ 对齐了几次（>1 = 中途重对齐过）
        self._should_abort = False
        self._abort_cause = None      # ★ 停下来的【真实原因】（可能来自看门狗）
        self.last_plan = None         # 诊断用
        # ★ 2026-09-26：实时阶段 —— 给 feedback 用（原来 stage 恒为 navigating）
        #   ★ 动作/话题两个桥都读它；FineApproach 也会回调更新（HEADING_ALIGN 是它内部的）
        self.current_stage = Stage.IDLE.value
        # ★ 看门狗：独立线程，控制循环挂了它兜底（回调异常/断流/循环死）
        self.watchdog = Watchdog(source, sink, log=self.log)
        self.authority = Authority(Authority.IDLE, log=self.log)

    # ------------------------------------------------------------ 开机对齐
    def align(self, start_scene: Pose6D) -> Transform2D:
        """★ 全系统最容易错、也最致命的一步。

        前提：狗【真的】停在组2给的起点位姿上。
              错多少，全程错多少 —— 而且我们无从察觉。
        """
        t = self.source.get()
        if t is None:
            raise RuntimeError('对齐时拿不到位姿 —— 位姿源还没出数据')
        p_capo = t.data
        p_scene = start_scene.planar
        require_frame(p_scene, FRAME_SCENE, 'align(start_scene)')
        require_frame(p_capo, FRAME_CAPO, 'align(capo reading)')
        self.T = build_T_scene_capo(p_scene, p_capo)
        self._align_capo0 = p_capo
        self._align_start = p_scene
        self._n_align += 1
        self.log('开机对齐：%s' % self.T)
        return self.T

    # ------------------------------------------------------------ ★★ 对齐（从 goal 来）
    def ensure_aligned(self, start_scene: Pose6D, tol_m: float = 0.20,
                       tol_yaw_deg: float = 15.0,
                       realign_on_mismatch: bool = True) -> str:
        """★★ 2026-09-27：用【组2 给在 goal 里的起点位姿】对齐（第一次），
        或者校验它（后续）。

        为什么要有这个方法
        ------------------
        2026-09-27 之前，T 只能通过命令行 --start 建立，而 goal 里的 s_* 【读都不读】。
        ⇒ 文档（action.md §5.1、联调手册 §二）写着"给我起点位姿"，
          但代码根本不认——那条线是断的。
        这个方法把那条线接上：**起点位姿从 goal 来，不问命令行要**。

        返回四种状态（调用方拿去打日志/上报）：
            'aligned_first'   第一次 ⇒ 用 s_* 建立 T
            'consistent'      s_* 和我们当前对狗位置的认识一致 ⇒ 沿用现有 T
            'realigned'       s_* 和我们认识的【差太多】⇒ 重新对齐（★ 会抹掉累积偏差）
            'mismatch_kept'   差太多、但调用方要求不重对齐 ⇒ 保留现有 T + 报警

        ★★ tol_m（默认 0.20 m）不是"容差"，是【"组2 说的起点"和"我们以为狗在哪"
           是不是同一件事】的判据】
            · 正常多段：第二段的 s_* = 第一段的 g_*，而我们到位的误差是
              pos_tol + 滑行 ≈ 8cm ⇒ 差 3~8cm，【远小于 0.20】⇒ 不重对齐 ✅
              （所以不会变成 action.md §四 担心的那种"每段重对齐 = 自欺欺人"）
            · 超过 0.20 m ⇒ 真出事了（组2 换了起点 / 狗被挪过 / 我们漂了很多）
              ⇒ 默认重新对齐 + 大声报警，让组2 的说法覆盖我们的
        来源：★ 论证 —— 能力上限 8cm + 里程计 2%/m 漂移的余量。
        """
        p_scene = start_scene.planar
        require_frame(p_scene, FRAME_SCENE, 'ensure_aligned(start_scene)')
        t = self.source.get()
        if t is None:
            raise RuntimeError('对齐时拿不到位姿 —— 位姿源还没出数据')
        p_capo = t.data
        require_frame(p_capo, FRAME_CAPO, 'ensure_aligned(capo reading)')

        # ---- 第一次：用组2 给的起点建立 T ----
        if self.T is None:
            self.T = build_T_scene_capo(p_scene, p_capo)
            self._align_capo0 = p_capo
            self._align_start = p_scene
            self._n_align += 1
            self.log('★★ 首次对齐（用 goal 里的 s_*）')
            self.log('   ├ 组2 报的起点(场景系): (%.3f, %.3f, %.2f° = %.4f rad)'
                     % (p_scene.x, p_scene.y,
                        math.degrees(p_scene.yaw), p_scene.yaw))
            self.log('   ├ CAPO 当时读数      : (%.3f, %.3f, %.2f°)'
                     % (p_capo.x, p_capo.y, math.degrees(p_capo.yaw)))
            self.log('   └ ⇒ T = %s' % self.T)
            return 'aligned_first'

        # ---- 后续：比对"组2 说的"和"我们以为的" ----
        cur = self.T @ p_capo                 # 当前 T 下，我们认为狗在哪
        d_pos = math.hypot(cur.x - p_scene.x, cur.y - p_scene.y)
        d_yaw = abs(wrap(p_scene.yaw - cur.yaw))
        detail = ('组2 说狗在 (%.3f, %.3f, %.1f°)，我们以为在 (%.3f, %.3f, %.1f°)'
                  ' ⇒ 差 %.3f m / %.1f°'
                  % (p_scene.x, p_scene.y, math.degrees(p_scene.yaw),
                     cur.x, cur.y, math.degrees(cur.yaw),
                     d_pos, math.degrees(d_yaw)))

        if d_pos <= tol_m and d_yaw <= math.radians(tol_yaw_deg):
            self.log('  起点校验 ✅ 一致（阈值 %.2f m / %.1f°）—— 沿用现有 T'
                     % (tol_m, tol_yaw_deg))
            self.log('   └ %s' % detail)
            return 'consistent'

        if not realign_on_mismatch:
            self.log('  ⚠️⚠️ 起点和我们的认识【差太多】，但配置要求【不重对齐】')
            self.log('   └ %s' % detail)
            self.log('   ⇒ 保留现有 T（不让组2 的说法覆盖我们累积的偏差）')
            return 'mismatch_kept'

        self.log('  ⚠️⚠️⚠️ 起点和我们的认识【差太多】⇒ ★ 重新对齐')
        self.log('   └ %s' % detail)
        self.log('   ⚠️ 这会把我们累积的偏差【抹掉】，之后以组2 的说法为准')
        self.log('   ├ CAPO 当时读数: (%.3f, %.3f, %.2f°)'
                 % (p_capo.x, p_capo.y, math.degrees(p_capo.yaw)))
        self.T = build_T_scene_capo(p_scene, p_capo)
        self._align_capo0 = p_capo
        self._align_start = p_scene
        self._n_align += 1
        self.log('   └ ⇒ 新 T = %s' % self.T)
        return 'realigned'

    def _set_stage(self, st: str):
        """★ 改阶段 + 记时间线（两个桥都靠它拿实时 stage）"""
        self.current_stage = st

    def abort(self):
        """外部急停钩子（遥控器/安全层调用）。"""
        self._should_abort = True

    def _abort_now(self) -> bool:
        """控制循环每个周期问一次：该不该停？

        ★★ 看门狗一旦触发，控制循环必须【立即中止】。
           否则：看门狗把狗安全停了 → 控制循环接着发指令 → 安全停被推翻。
           （2026-09-22 端到端演示撞上：看门狗报了 control_loop_dead，
             可紧接着转向阶段又发了一堆 move）
        """
        if self._should_abort:
            return True
        t = self.watchdog.triggered()
        if t:
            self._should_abort = True
            if self._abort_cause is None:
                self._abort_cause = t        # ★ 记住真实原因，别丢
            self.log('  ⚠️ 看门狗已触发（%s）—— 控制循环中止' % t)
            return True
        return False

    def _map_reason(self, r: str) -> str:
        """把内部的控制层 reason 映射成对外的 Reason。

        ★ blocked（被人挡住，等一等能重试）和 no_path（地图上过不去，要换方案）
          下游的处理完全不同，不能混。
        """
        # ★ 看门狗先发现的情况：它是【谁】，比笼统的 aborted 有用得多。
        #   否则"回调抛异常"会被报成 aborted，下游就分不清是哪种失败了。
        if r == 'aborted' and self._abort_cause:
            c = self._abort_cause
            if 'callback_error' in c:
                return Reason.SOURCE_ERROR.value
            if 'pose_stale' in c:
                return Reason.STALE.value
            if 'control_loop_dead' in c:
                return Reason.INTERNAL_ERROR.value
        return {
            'reached': Reason.ARRIVED.value,
            'aligned': Reason.ARRIVED.value,
            # ★★ 2026-09-28：blocked 终于真的能产生了 ——
            #   跟随器在【走的状态下走不动】判 blocked（前方有东西挡着）
            #   在【转的状态下转不动】才判 stuck（姿态卡死）
            #   之前这里【根本没有 blocked 这一项】⇒ 上游永远收不到 blocked，
            #   而组2 的 SPAV 只在 blocked 时才绕行 ⇒ 他们的核心演示演不出来
            'blocked': Reason.BLOCKED.value,
            'no_progress': Reason.STUCK.value,
            'timeout': Reason.TIMEOUT.value,
            'aborted': Reason.ABORTED.value,
            'stale_pose': Reason.STALE.value,
            'source_callback_error': Reason.SOURCE_ERROR.value,
            'no_pose': Reason.NO_POSE.value,
            'position_out_of_tol': Reason.STUCK.value,
            # ★ 2026-09-26：转身把位置挪偏了 —— 单独一类，不要混进 STUCK
            #   （STUCK 是"走不动"，这个"到了又歪了"，下游处理完全不同）
            'turn_shift': Reason.TURN_SHIFT.value,
            'no_lookahead': Reason.PLANNING_FAILED.value,
            'off_path': Reason.STUCK.value,
            'travel_cap': Reason.ABORTED.value,
        }.get(r, r if r in {e.value for e in Reason} else Reason.INTERNAL_ERROR.value)

    def _scene_source(self):
        if self.T is None:
            raise RuntimeError('还没做开机对齐 —— 先调 align(start_scene)')
        return _ScenePoseSource(self.source, self.T)

    # ------------------------------------------------------------ 主接口
    def navigate_to(self, goal: Pose6D) -> NavResult:
        """走到场景系里的一个 6D 位姿。

        ★ 纯逻辑，不依赖 ROS。仿真和真机共用这一份。
        """
        t_start = time.time()
        goal_p = goal.planar
        require_frame(goal_p, FRAME_SCENE, 'navigate_to(goal)')
        stages: List[Tuple[str, float]] = []

        try:
            src = self._scene_source()
            obstacle_scene = (
                _SceneObstacleSource(self.obstacle_source, self.T)
                if self.cfg.dynamic_avoid and self.obstacle_source is not None
                else None)
        except RuntimeError as e:
            return NavResult(False, None, Reason.NO_POSE.value, 0.0, float('nan'),
                             self.source.name, float('nan'),
                             [(Stage.IDLE.value, 0.0)]); self._set_stage(Stage.IDLE.value)

        # ---- 起点 ----
        t0 = src.get()
        if t0 is None:
            return NavResult(False, None, Reason.NO_POSE.value, 0.0, float('nan'),
                             self.source.name, float('nan'), stages)
        start_scene = t0.data

        self.log('目标 %s' % goal_p)
        self.log('起点 %s' % start_scene)

        # ---- PLANNING：有栅格就规划一条路径 ----
        self._should_abort = False
        self._abort_cause = None
        plan = None
        if self.grid is not None:
            stages.append((Stage.PLANNING.value, time.time())); self._set_stage(Stage.PLANNING.value)
            plan = plan_path(self.grid, start_scene, goal_p,
                             inflate_radius=self.cfg.inflate_radius,
                             final_approach_lengths=self.cfg.final_approach_lengths)
            self.last_plan = plan
            if not plan.ok:
                self.log('规划失败：%s' % plan.reason)
                self.sink.settle()
                return NavResult(False, None, Reason.NO_PATH.value,
                                 time.time() - t_start, float('nan'),
                                 self.source.name, float('nan'), stages)
            self.log('规划出 %d 个路点，长 %.2f m' % (len(plan.waypoints), plan.length))
            if plan.final_approach_d > 0:
                a_, b_ = plan.waypoints[-2], plan.waypoints[-1]
                self.log('★ 预瞄点生效：离终点 %.2f m，最后一段沿目标朝向（%.0f°）'
                         % (plan.final_approach_d,
                            math.degrees(math.atan2(b_.y - a_.y, b_.x - a_.x))))
            elif self.cfg.final_approach_lengths:
                self.log('★ 预瞄点没启用（pre 被占/不通/距离太近）⇒ 按普通规划')
        else:
            self.log('没给栅格 —— 直接直线过去（不做避障）')

        # ---- 组配置 ----
        acfg = self.cfg.approach
        if self.cfg.max_speed is not None:
            acfg = ApproachConfig(**{**acfg.__dict__,
                                     'v_cruise': min(acfg.v_cruise, self.cfg.max_speed)})
        acfg = ApproachConfig(**{**acfg.__dict__,
                                 'pos_tol': self.cfg.pos_tol,
                                 'yaw_tol_deg': self.cfg.yaw_tol_deg,
                                 # ★ 分段上限和总预算【取小】，不互相覆盖
                                 'timeout': min(self.cfg.approach_timeout,
                                                self.cfg.timeout)})

        self.watchdog.armed()          # ★ 真的要发指令了，才开始守
        approach = FineApproach(acfg, src, self.sink, log=self.log,
                                 on_stage=self._set_stage,
                                should_abort=self._abort_now,
                                on_cycle=self.watchdog.feed)

        # ---- 走 ----
        # ★ 从这一刻起到函数结束，无论怎么退出（正常/失败/异常）都要 disarm，
        #   否则看门狗会一直守着，控制循环一停它就误触发
        #   （2026-09-22 端到端演示撞上：任务3 失败返回后，看门狗 0.5s 后误报）
        try:
            return self._run_motion(plan, goal_p, src, stages, t_start,
                                    approach, acfg, start_scene,
                                    obstacle_scene)
        finally:
            self.watchdog.disarm()

    def _run_motion(self, plan, goal_p, src, stages, t_start,
                    approach, acfg, start_scene, obstacle_scene):
        if plan is not None:
            fcfg = FollowerConfig(**{**self.cfg.follower.__dict__,
                                     'timeout': min(self.cfg.follow_timeout,
                                                    self.cfg.timeout)})
            fol = PathFollower(fcfg, src, self.sink, log=self.log,
                               should_abort=self._abort_now,
                               on_cycle=self.watchdog.feed,
                               grid=self.grid,
                               inflate_radius=self.cfg.inflate_radius,
                               avoid=AvoidConfig(**{
                                   **self.cfg.avoid.__dict__,
                                   'enabled': self.cfg.dynamic_avoid,
                               }),
                               obstacle_source=obstacle_scene)
            stages.append((Stage.NAVIGATING.value, time.time())); self._set_stage(Stage.NAVIGATING.value)
            fr = fol.follow(plan, goal_p)
            self.log('跟随结束：%s（走了 %.2f m）' % (fr.reason, fr.traveled))
            if not fr.success:
                self.sink.settle()
                return NavResult(False, Pose6D.from_planar(fr.final_pose)
                                 if fr.final_pose else None,
                                 self._map_reason(fr.reason),
                                 time.time() - t_start,
                                 fr.traveled * self.cfg.drift_per_meter,
                                 self.source.name,
                                 math.hypot(goal_p.x - fr.final_pose.x,
                                            goal_p.y - fr.final_pose.y)
                                 if fr.final_pose else float('nan'),
                                 stages,
                                 blocked_distance=getattr(fr, 'blocked_distance', -1.0),
                                 blocked_bearing=getattr(fr, 'blocked_bearing', 0.0))
            # 交接：剩最后一段交给末端精调
            stages.append((Stage.FINE_APPROACH.value, time.time())); self._set_stage(Stage.FINE_APPROACH.value)
            r = approach.go_to(goal_p, target_yaw=goal_p.yaw)
        else:
            stages.append((Stage.NAVIGATING.value, time.time())); self._set_stage(Stage.NAVIGATING.value)
            r = approach.go_to(goal_p, target_yaw=goal_p.yaw)
            stages.append((Stage.FINE_APPROACH.value, time.time())); self._set_stage(Stage.FINE_APPROACH.value)

        # ---- 收尾 ----（watchdog 由 navigate_to 的 finally 统一 disarm）
        self.sink.settle()

        final = r.final_pose
        internal_err = r.final_error
        if not math.isfinite(internal_err) and final is not None:
            internal_err = math.hypot(goal_p.x - final.x, goal_p.y - final.y)

        # ---- 漂移估计（★ 是估计，不是测量）----
        traveled = (math.hypot(final.x - start_scene.x, final.y - start_scene.y)
                    if final is not None else 0.0)
        uncertainty = traveled * self.cfg.drift_per_meter

        reason = self._map_reason(r.reason)
        ok = r.success and reason == Reason.ARRIVED.value

        return NavResult(
            success=ok,
            final_pose=Pose6D.from_planar(final) if final is not None else None,
            reason=reason,
            elapsed=time.time() - t_start,
            uncertainty=uncertainty,
            odometry_source=self.source.name,
            internal_error=internal_err,
            stages=stages,
        )


# ================================================================ 便捷函数
def navigate_to(navigator: Navigator, goal_position, goal_yaw: float = 0.0) -> NavResult:
    """给组2 用的写法：navigate_to(nav, (x, y, z), yaw)"""
    x, y, z = goal_position
    return navigator.navigate_to(Pose6D(x, y, z, 0.0, 0.0, goal_yaw, frame=FRAME_SCENE))
