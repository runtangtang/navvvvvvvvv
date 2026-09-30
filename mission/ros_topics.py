# -*- coding: utf-8 -*-
"""
mission/ros_topics.py —— 把纯 Python 的 Navigator 接到 ROS2 上

★ 两份包装共用【同一个】Navigator 实例：
      RosActionServer    （将来加，标准、有取消）
      RosTopicBridge     （现在这个，联调起步快）
   所以"用 Action 还是 Topic"对我们内部没有区别。

对外话题
--------
发:   /nav/navigate_to   (msg，字段同 goal)
回:   /nav/result        (msg，字段同 result)
中途: /nav/feedback      (msg，字段同 feedback，10Hz)
额外: /nav/planned_path  (nav_msgs/Path，任务开始时发一次，给可视化)

消息定义用 Python dataclass 先顶上（不依赖自定义 ROS 消息包就能跑通）。
★ 正式对接时把这些换成真正的 .msg / .action —— 只是搬运字段，逻辑不动。

安全
----
· 任务串行：同一时刻只跑一个（来新目标会拒绝，除非旧任务已结束）
· 任何退出路径都走 settle()
"""
import dataclasses
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from core.types import Pose6D, Pose2D, FRAME_SCENE
from mission.api import Navigator, NavResult, NavigateToConfig


# ================================================================ 对外消息
@dataclass
class NavigateToGoal:
    """★ 和 .action 的 goal 段【字段一一对应】"""
    s_x: float = 0.0
    s_y: float = 0.0
    s_z: float = 0.0
    s_roll: float = 0.0
    s_pitch: float = 0.0
    s_yaw: float = 0.0
    g_x: float = 0.0
    g_y: float = 0.0
    g_z: float = 0.0
    g_roll: float = 0.0
    g_pitch: float = 0.0
    g_yaw: float = 0.0
    pos_tol: float = 0.03
    yaw_tol: float = 0.05
    timeout: float = 120.0
    max_speed: float = 0.0


@dataclass
class NavFeedback:
    stage: str = 'idle'
    remaining_dist: float = 0.0
    remaining_yaw: float = 0.0
    cur_x: float = 0.0
    cur_y: float = 0.0
    cur_yaw: float = 0.0


@dataclass
class NavResultMsg:
    success: bool = False
    final_x: float = 0.0
    final_y: float = 0.0
    final_yaw: float = 0.0
    internal_error: float = float('nan')
    uncertainty: float = float('nan')
    reason: str = ''
    elapsed: float = 0.0


# ================================================================ 桥
class RosTopicBridge:
    """订阅 /nav/navigate_to，跑完发 /nav/result，跑的过程发 /nav/feedback。

    用法（在狗上的主程序里）：
        nav = Navigator(capo_source, sink, cfg, grid=grid, log=print)
        nav.align(start_pose)                       # 开机对齐一次
        bridge = RosTopicBridge(node, nav)
        rclpy.spin(node)
    """

    def __init__(self, node, navigator: Navigator,
                 goal_topic: str = '/nav/navigate_to',
                 result_topic: str = '/nav/result',
                 feedback_topic: str = '/nav/feedback',
                 path_topic: str = '/nav/planned_path',
                 feedback_hz: float = 10.0,
                 align_tol_m: float = 0.20,
                 align_tol_yaw_deg: float = 15.0,
                 realign_on_mismatch: bool = True,
                 log=None):
        self.node = node
        self.nav = navigator
        self.log = log or (lambda s: None)
        self.feedback_hz = feedback_hz
        # ★★ 2026-09-27：起点对齐由 goal 里的 s_* 驱动（和 Action 版一致）
        self.align_tol_m = align_tol_m
        self.align_tol_yaw_deg = align_tol_yaw_deg
        self.realign_on_mismatch = realign_on_mismatch
        self._lock = threading.Lock()
        self._busy = False
        self._goal_xy = None      # ★ 当前目标（算 feedback 的 remaining_dist 用）
        self._last: Optional[Tuple[float, Pose6D]] = None   # (t, pose) 供 feedback 用

        # ---- 用 std_msgs/String 装 JSON 先跑通，正式对接换真消息 ----
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qr = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self._pub_result = node.create_publisher(String, result_topic, qr)
        self._pub_fb = node.create_publisher(String, feedback_topic, qr)
        self._pub_path = node.create_publisher(String, path_topic, qr)
        node.create_subscription(String, goal_topic, self._on_goal, qr)
        self._String = String
        self.log('RosTopicBridge 就绪：%s → %s / %s'
                 % (goal_topic, result_topic, feedback_topic))

    # ------------------------------------------------------------
    def _on_goal(self, msg):
        import json
        try:
            d = json.loads(msg.data)
            goal = NavigateToGoal(**{k: float(v) if not isinstance(v, str) else v
                                     for k, v in d.items()})
        except Exception as e:
            self.log('✗ 目标解析失败：%s' % e)
            return
        with self._lock:
            if self._busy:
                self.log('✗ 已有任务在跑 —— 拒绝新目标（同一时刻只跑一个）')
                return
            self._busy = True
        threading.Thread(target=self._run, args=(goal,), daemon=True).start()

    # ------------------------------------------------------------
    def _run(self, goal: NavigateToGoal):
        t0 = time.time()
        # 反馈线程
        stop_fb = threading.Event()
        fb = threading.Thread(target=self._feedback_loop, args=(stop_fb,),
                              daemon=True)
        fb.start()
        phase = 'align'
        try:
            # ★ 路径是 navigate_to 内部算的，所以【不能】在这之前发。
            #   由 _feedback_loop 盯着 last_plan，一变就发（见下）
            self._last_plan_id = None

            # ★★ 2026-09-27：用 goal 里的 s_* 对齐（第一次）/ 校验（后续）
            self.nav.ensure_aligned(
                Pose6D(goal.s_x, goal.s_y, goal.s_z,
                       goal.s_roll, goal.s_pitch, goal.s_yaw,
                       frame=FRAME_SCENE),
                tol_m=self.align_tol_m,
                tol_yaw_deg=self.align_tol_yaw_deg,
                realign_on_mismatch=self.realign_on_mismatch)
            phase = 'run'

            g = Pose6D(goal.g_x, goal.g_y, goal.g_z,
                       goal.g_roll, goal.g_pitch, goal.g_yaw, frame=FRAME_SCENE)

            # ★★ 2026-09-23 修：goal 里的容差/超时/限速原来【被收了但没用】——
            #    navigate_to 用的是【构造时】的 NavigateToConfig，组2 传什么都不生效。
            #    现在按每个目标临场生成配置。任务串行（_busy），所以改 cfg 是安全的。
            #    ⚠️ 单位：JSON 里的 g_yaw / yaw_tol 都是【弧度】（见 action.md），
            #            而内部配置用【度】—— 必须换算，这个坑之前踩过。
            old_cfg = self.nav.cfg
            try:
                new_cfg = dataclasses.replace(
                    old_cfg,
                    pos_tol=float(goal.pos_tol),
                    yaw_tol_deg=math.degrees(float(goal.yaw_tol)),
                    timeout=float(goal.timeout),
                    max_speed=(float(goal.max_speed)
                               if goal.max_speed and goal.max_speed > 0
                               else None))
                self.nav.cfg = new_cfg
                self.log('  按目标设定：pos_tol %.3f m  yaw_tol %.3f rad (%.1f°)  '
                         'timeout %.0fs  max_speed %s'
                         % (new_cfg.pos_tol, goal.yaw_tol, new_cfg.yaw_tol_deg,
                            new_cfg.timeout, new_cfg.max_speed))
            except Exception as e:
                self.log('  ⚠️ 目标里的容差无效（%s），沿用默认配置' % e)

            with self._lock:
                self._goal_xy = (float(goal.g_x), float(goal.g_y))
            res = self.nav.navigate_to(g)
            self._publish_result(res, time.time() - t0)
        except Exception as e:
            if phase == 'align':
                self.log('✗ 用 goal 的 s_* 对齐失败：%s' % e)
                r = NavResultMsg(success=False, reason='no_pose',
                                 elapsed=time.time() - t0)
            else:
                self.log('✗ 任务异常：%s' % e)
                r = NavResultMsg(success=False, reason='internal_error',
                                 elapsed=time.time() - t0)
            self._publish(self._pub_result, asdict(r))
        finally:
            stop_fb.set()
            # ★ 还原配置（下一次任务会再按它的目标重设）
            try:
                self.nav.cfg = old_cfg
            except Exception:
                pass
            with self._lock:
                self._busy = False
                self._goal_xy = None

    # ------------------------------------------------------------
    def _feedback_loop(self, stop):
        import json
        while not stop.is_set():
            try:
                # 从 navigator 的位姿源拿当前位置
                # ★ 规划路径一变就发（navigate_to 内部算完会设 last_plan）
                pl = self.nav.last_plan
                if pl is not None and id(pl) != getattr(self, '_last_plan_id', None):
                    self._last_plan_id = id(pl)
                    if pl.ok:
                        self._publish(self._pub_path,
                                      {'waypoints': [[w.x, w.y, w.yaw]
                                                     for w in pl.waypoints],
                                       'length': pl.length})
                        self.log('已发规划路径（%d 个路点，%.2f m）'
                                 % (len(pl.waypoints), pl.length))

                t = self.nav.source.get()
                if t is not None:
                    p_scene = (self.nav.T @ t.data) if self.nav.T else t.data
                    self._last = (time.time(), p_scene)
                    # ★★ 2026-09-23 修：remaining_dist / remaining_yaw 原来是【恒为 0】
                    #    （dataclass 默认值从没被填过）。现在按"当前位置→目标"实时算，
                    #    组2 能直接拿它画进度条。
                    rd = float('nan')
                    ryaw = float('nan')
                    stage = 'navigating'
                    with self._lock:
                        g = self._goal_xy
                    if g is not None:
                        dx, dy = g[0] - p_scene.x, g[1] - p_scene.y
                        rd = math.hypot(dx, dy)
                        if rd > 1e-6:
                            bearing = math.atan2(dy, dx)
                            ryaw = abs(math.atan2(math.sin(bearing - p_scene.yaw),
                                                  math.cos(bearing - p_scene.yaw)))
                        # 粗分阶段：进末端精调范围就报 fine_approach
                        # ★ 用跟随器真正的阈值 fine_range（默认 0.5m），别另造一个数
                        #   ⚠️ 这只是给组2 看的粗略标识，精确阶段机在 mission/mission.py
                        try:
                            if rd <= self.nav.cfg.follower.fine_range:
                                stage = 'fine_approach'
                        except Exception:
                            pass
                    fb = NavFeedback(
                        stage=getattr(self.nav, 'current_stage', stage) or stage,
                        remaining_dist=rd,
                        remaining_yaw=ryaw,
                        cur_x=p_scene.x, cur_y=p_scene.y, cur_yaw=p_scene.yaw)
                    self._publish(self._pub_fb, asdict(fb))
            except Exception:
                pass
            time.sleep(1.0 / self.feedback_hz)

    def _publish_path(self):
        import json
        try:
            pl = self.nav.last_plan
            if pl is None or not pl.ok:
                return
            self._publish(self._pub_path,
                          {'waypoints': [[w.x, w.y, w.yaw] for w in pl.waypoints],
                           'length': pl.length})
        except Exception:
            pass

    def _publish_result(self, res: NavResult, elapsed: float):
        fp = res.final_pose
        r = NavResultMsg(
            success=bool(res.success),
            final_x=fp.x if fp else float('nan'),
            final_y=fp.y if fp else float('nan'),
            final_yaw=fp.yaw if fp else float('nan'),
            internal_error=res.internal_error,
            uncertainty=res.uncertainty,
            reason=res.reason,
            elapsed=elapsed)
        self._publish(self._pub_result, asdict(r))

    def _publish(self, pub, d):
        import json
        m = self._String()
        m.data = json.dumps(d)
        pub.publish(m)


# ================================================================ 独立节点
def make_nav_node(source, sink, cfg: NavigateToConfig, grid,
                  start_scene: Pose6D, log=None):
    """起一个只有 bridge 的节点（狗上主程序的最小形态）"""
    import rclpy
    from rclpy.node import Node
    node = Node('nav_bridge')
    nav = Navigator(source, sink, cfg=cfg, grid=grid, log=log)
    nav.align(start_scene)
    bridge = RosTopicBridge(node, nav, log=log)
    return node, nav, bridge
