# -*- coding: utf-8 -*-
"""
mission/ros_action.py —— 把纯 Python 的 Navigator 接到 ROS2 Action 上（正式版接口）

★ 和 ros_topics.py 是【同一层的两个载体】：
      RosTopicBridge  —— std_msgs/String 装 JSON（v0，联调起步用）
      RosActionServer —— 真 .action（v1，正式版）★ 本文件
  Navigator（纯 Python 层）【一行不改】—— 这就是当初分层的价值。

消息定义：nav_interfaces/action/NavigateTo.action

★★ 用之前必须知道的两件事
------------------------
① ★【必须用 MultiThreadedExecutor】
   Action 的 execute_callback 会一直阻塞到任务结束。单线程 executor 下，
   它会把所有其他回调（尤其是 500Hz 的位姿订阅）一起堵死 ⇒ 导航直接废掉。
   ⇒ 节点侧这么写：
        from rclpy.executors import MultiThreadedExecutor
        ex = MultiThreadedExecutor()
        ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()

② ★【一个任务一次】—— 任务进行中来新目标会被【拒绝】（goal_handle 返回 REJECT）。
   要中断当前任务，用 Action 的 cancel（会转成我们的 abort）。

安全
----
· 任何退出路径都走 settle()
· 取消 / 异常 / 断流都要安全停
· 用 Navigator 自带的 watchdog（cancel 只是另一条 abort 路径）
"""
import math
import threading
import time

from core.types import Pose6D, FRAME_SCENE
from mission.api import Navigator


class RosActionServer:
    """订阅 /nav/navigate_to (Action)，跑完回 result，跑的过程发 feedback(10Hz)。

    用法（狗上主程序）：
        nav = Navigator(capo_source, sink, cfg, grid=grid, log=print)
        nav.align(start_pose)                       # 开机对齐一次
        srv = RosActionServer(node, nav, log=print)
        ex = MultiThreadedExecutor()                # ★ 必须多线程
        ex.add_node(node)
        ex.spin()
    """

    def __init__(self, node, navigator: Navigator,
                 action_name: str = '/nav/navigate_to',
                 feedback_hz: float = 10.0,
                 align_tol_m: float = 0.20,
                 align_tol_yaw_deg: float = 15.0,
                 realign_on_mismatch: bool = True,
                 publish_topics: bool = True,
                 log=None):
        # 延迟 import —— 没装 nav_interfaces 也能 import 本模块
        from nav_interfaces.action import NavigateTo
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup

        self.node = node
        self.nav = navigator
        self.log = log or (lambda s: None)
        self.feedback_hz = feedback_hz
        # ★★ 2026-09-27：起点对齐（由 goal 里的 s_* 驱动，不再靠命令行 --start）
        self.align_tol_m = align_tol_m
        self.align_tol_yaw_deg = align_tol_yaw_deg
        self.realign_on_mismatch = realign_on_mismatch
        self._lock = threading.Lock()
        self._busy = False
        self._cancel = threading.Event()
        self._last_plan_id = None

        # ★ ReentrantCallbackGroup：允许 execute 和其他回调并发
        self._cbg = ReentrantCallbackGroup()

        self._srv = ActionServer(
            node, NavigateTo, action_name,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._cbg)

        self.log('  ActionServer 就绪：%s  （★ 要用 MultiThreadedExecutor）'
                 % action_name)

        # ★★ 2026-09-28：除了 Action 协议，【再发一份 JSON 话题】
        #    为什么：可视化 tools/live_view.py 订阅的是 /nav/feedback、
        #    /nav/planned_path、/nav/result —— 那是 RosTopicBridge(JSON版) 发的，
        #    ActionServer 原来【一个都不发】⇒ 一用 --action 可视化就全空
        #    （2026-09-28 真机踩到："可视化啥都没有"）
        #    ⇒ 这里把同样三个话题也发出去，字段格式【照抄 ros_topics.py】，
        #      两边共用同一套 dataclass，格式不会漂。
        self.publish_topics = publish_topics
        self._pub_fb = self._pub_path = self._pub_result = None
        self._String = None
        if publish_topics:
            try:
                from std_msgs.msg import String
                from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                                       HistoryPolicy)
                qr = QoSProfile(depth=50,
                                reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST)
                self._pub_fb = node.create_publisher(String, '/nav/feedback', qr)
                self._pub_path = node.create_publisher(String, '/nav/planned_path', qr)
                self._pub_result = node.create_publisher(String, '/nav/result', qr)
                self._String = String
                self.log('  附带发 JSON 话题（给可视化用）：'
                         '/nav/feedback + /nav/planned_path + /nav/result')
            except Exception as e:
                self.log('  ⚠️ JSON 话题发不出去（%s）—— 可视化会看不到东西' % e)

    # ------------------------------------------------------------ 回调
    def _on_goal(self, goal_request):
        from rclpy.action import GoalResponse
        with self._lock:
            if self._busy:
                self.log('✗ 已有任务在跑 —— 拒绝新目标（一次只跑一个）')
                return GoalResponse.REJECT
            self._busy = True
        self.log('  收到目标  起点 s=(%.3f, %.3f, %.4f rad)  →  '
                 '终点 g=(%.3f, %.3f, %.4f rad)'
                 % (goal_request.s_x, goal_request.s_y, goal_request.s_yaw,
                    goal_request.g_x, goal_request.g_y, goal_request.g_yaw))
        self.log('             pos_tol %.3f m  yaw_tol %.4f rad (%.1f°)  '
                 'timeout %.0fs  max_speed %.2f'
                 % (goal_request.pos_tol, goal_request.yaw_tol,
                    math.degrees(goal_request.yaw_tol),
                    goal_request.timeout, goal_request.max_speed))
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        from rclpy.action import CancelResponse
        self.log('  ★ 收到取消请求 —— 会安全停下来')
        self._cancel.set()
        # ★ 走 Navigator 现成的 abort 通道（它内部会把 should_abort 置位，
        #   跟随器/精调循环每周期都会查 ⇒ 会走 settle() 安全停）
        try:
            self.nav.abort()
        except Exception:
            pass
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """★ 这个回调会【阻塞】到导航结束 —— 必须跑在多线程 executor 里"""
        from nav_interfaces.action import NavigateTo
        # ★ 别 import GoalStatus —— 它不在 rclpy.action 里（Humble 下会 ImportError），
        #   而且我们用 goal_handle.succeed()/abort()/canceled() 就够了，用不到它。
        import dataclasses

        req = goal_handle.request
        t0 = time.time()
        self._cancel.clear()

        # ---- feedback 线程：从 navigator 拿当前位置，10Hz 发 ----
        stop_fb = threading.Event()
        last = {'p': None}
        fb_thread = threading.Thread(
            target=self._feedback_loop, args=(goal_handle, stop_fb, last),
            daemon=True)
        fb_thread.start()

        old_cfg = self.nav.cfg
        reason = 'internal_error'
        ok = False
        phase = 'align'          # ★ 出错时用来区分：是"对齐"挂了还是"导航"挂了
        try:
            # ★★ 2026-09-27：用【goal 里的 s_*】做开机对齐。
            #    · 第一次       ⇒ 用 s_* 建立 T（之后全程不变）
            #    · 后续每次     ⇒ 只把它和我们"以为狗在哪"比对；差得少就沿用，
            #                     差太多才重对齐 + 大声报警
            #    ★ 为什么不是"每段重对齐"：那会把累积偏差抹掉（自欺欺人，
            #      见 action.md §四 / 架构设计 §3.2）。这里的阈值 0.20m 保证
            #      正常的多段误差（3~8cm）【不会】触发重对齐。
            self.nav.ensure_aligned(
                Pose6D(req.s_x, req.s_y, req.s_z, req.s_roll,
                       req.s_pitch, req.s_yaw, frame=FRAME_SCENE),
                tol_m=self.align_tol_m,
                tol_yaw_deg=self.align_tol_yaw_deg,
                realign_on_mismatch=self.realign_on_mismatch)
            phase = 'run'

            # ★ 按目标设定容差/超时/限速（和 topic 版同一套逻辑）
            #   ⚠️ 单位：action 里 yaw_tol 是【弧度】，内部配置用【度】
            try:
                new_cfg = dataclasses.replace(
                    old_cfg,
                    pos_tol=float(req.pos_tol),
                    yaw_tol_deg=math.degrees(float(req.yaw_tol)),
                    timeout=float(req.timeout),
                    max_speed=(float(req.max_speed) if req.max_speed > 0
                               else None))
                self.nav.cfg = new_cfg
                self.log('  按目标设定：pos_tol %.3f m  yaw_tol %.3f rad (%.1f°)  '
                         'timeout %.0fs  max_speed %s'
                         % (new_cfg.pos_tol, req.yaw_tol, new_cfg.yaw_tol_deg,
                            new_cfg.timeout, new_cfg.max_speed))
            except Exception as e:
                self.log('  ⚠️ 目标里的容差无效（%s），沿用默认配置' % e)

            goal = Pose6D(req.g_x, req.g_y, req.g_z,
                          req.g_roll, req.g_pitch, req.g_yaw, frame=FRAME_SCENE)
            self._goal_xy = (float(req.g_x), float(req.g_y))
            res = self.nav.navigate_to(goal)
            ok = res.success
            reason = res.reason
            final = res.final_pose
            internal_err = res.internal_error
            unc = res.uncertainty
        except Exception as e:
            if phase == 'align':
                self.log('✗ 用 goal 的 s_* 对齐失败：%s' % e)
                reason = 'no_pose'      # ★ 对不齐 ≠ 导航失败，reason 分开报
            else:
                self.log('✗ 任务异常：%s' % e)
            final, internal_err, unc = None, float('nan'), float('nan')
        finally:
            self._goal_xy = None
            stop_fb.set()
            try:
                self.nav.cfg = old_cfg
            except Exception:
                pass
            try:
                self.nav.sink.settle()      # ★ 任何路径都收尾
            except Exception:
                pass
            with self._lock:
                self._busy = False

        # ---- 组装 result ----
        r = NavigateTo.Result()
        r.success = bool(ok)
        r.internal_error = float(internal_err)
        r.uncertainty = float(unc)
        r.reason = str(reason)
        r.elapsed = float(time.time() - t0)
        if final is not None:
            from geometry_msgs.msg import PoseStamped
            ps = PoseStamped()
            ps.header.stamp = self.node.get_clock().now().to_msg()
            ps.header.frame_id = 'scene'
            ps.pose.position.x = float(final.x)
            ps.pose.position.y = float(final.y)
            ps.pose.position.z = float(getattr(final, 'z', 0.0))
            # yaw → 四元数（只绕 z）
            cy, sy = math.cos(final.yaw / 2.0), math.sin(final.yaw / 2.0)
            ps.pose.orientation.z = sy
            ps.pose.orientation.w = cy
            r.final_pose = ps

        # ★ 同一份内容再发一份 JSON —— 给 live_view / 事后分析用
        #   ⚠️ 字段名【照抄 ros_topics.py 的 NavResultMsg】（final_* 拆成三个 float），
        #      不要用 Action 的 final_pose —— live_view 读的是 final_x/final_y/final_yaw
        self._publish(self._pub_result, {
            'success': bool(ok),
            'final_x': float(final.x) if final is not None else float('nan'),
            'final_y': float(final.y) if final is not None else float('nan'),
            'final_yaw': float(final.yaw) if final is not None else float('nan'),
            'internal_error': float(internal_err),
            'uncertainty': float(unc),
            'reason': str(reason),
            'elapsed': float(r.elapsed)})

        # ---- 标记结束（★ 被取消的话要报 canceled，不是 succeeded）----
        if self._cancel.is_set():
            goal_handle.canceled()
        elif ok:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        self.log('  任务结束：success=%s reason=%s elapsed=%.1fs'
                 % (r.success, r.reason, r.elapsed))
        return r

    # ------------------------------------------------------------
    def _publish(self, pub, d):
        """发一条 JSON 到话题（★ 只给可视化用，失败绝不能影响导航）"""
        if pub is None or self._String is None:
            return
        try:
            import json
            m = self._String()
            m.data = json.dumps(d)
            pub.publish(m)
        except Exception:
            pass

    def _feedback_loop(self, goal_handle, stop, last):
        """10Hz 发 feedback。★ 回调里【绝不抛异常】（会杀掉线程）"""
        from nav_interfaces.action import NavigateTo
        from geometry_msgs.msg import PoseStamped
        src = self.nav.source
        while not stop.is_set():
            try:
                # ★ 规划路径一变就发（navigate_to 内部算完会设 last_plan）
                #   —— 和 ros_topics.py 里同一套逻辑
                pl = self.nav.last_plan
                if pl is not None and id(pl) != self._last_plan_id:
                    self._last_plan_id = id(pl)
                    if getattr(pl, 'ok', False):
                        self._publish(self._pub_path,
                                      {'waypoints': [[w.x, w.y, w.yaw]
                                                     for w in pl.waypoints],
                                       'length': pl.length})
                        self.log('  已发规划路径（%d 个路点，%.2f m）'
                                 % (len(pl.waypoints), pl.length))

                t = src.get()
                if t is not None:
                    p = (self.nav.T @ t.data) if self.nav.T else t.data
                    stage = (getattr(self.nav, 'current_stage', 'navigating')
                             or 'navigating')
                    g = getattr(self, '_goal_xy', None)
                    rd = float('nan')
                    ryaw = float('nan')
                    if g is not None:
                        dx, dy = g[0] - p.x, g[1] - p.y
                        rd = float(math.hypot(dx, dy))
                        if rd > 1e-6:
                            b = math.atan2(dy, dx)
                            ryaw = float(abs(math.atan2(
                                math.sin(b - p.yaw), math.cos(b - p.yaw))))

                    # ① Action 协议的 feedback
                    fb = NavigateTo.Feedback()
                    fb.stage = stage
                    fb.remaining_dist = rd
                    fb.remaining_yaw = ryaw
                    ps = PoseStamped()
                    ps.header.stamp = self.node.get_clock().now().to_msg()
                    ps.header.frame_id = 'scene'
                    ps.pose.position.x = float(p.x)
                    ps.pose.position.y = float(p.y)
                    cy, sy = math.cos(p.yaw / 2.0), math.sin(p.yaw / 2.0)
                    ps.pose.orientation.z = sy
                    ps.pose.orientation.w = cy
                    fb.current_pose = ps
                    goal_handle.publish_feedback(fb)

                    # ② ★ 同一份内容再发一份 JSON —— 给 live_view 用
                    self._publish(self._pub_fb, {
                        'stage': stage,
                        'remaining_dist': rd, 'remaining_yaw': ryaw,
                        'cur_x': float(p.x), 'cur_y': float(p.y),
                        'cur_yaw': float(p.yaw)})
            except Exception:
                pass
            time.sleep(1.0 / max(self.feedback_hz, 1.0))
