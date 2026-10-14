# -*- coding: utf-8 -*-
"""
demo/nav_node_real.py —— 【狗上跑】真机版主导航节点

★★ 危险：这个节点【真的会驱动狗】★★
    启动必须显式加 --i-know-it-moves，否则拒绝执行。

它和 nav_node_sim.py 的唯一区别：
    位姿源  仿真狗的 /SMX/Odom   →   真 CAPO 的 /SMX/Odom   （话题一样，代码一样）
    出口    SportSink            →   同一个 SportSink        （真的发给狗了）
★ 这就是架构里说的"一份代码三种后端"—— 换后端不改逻辑。

今天（2026-09-23）的用法（没有组1的图、没有组2的位姿）:
    ① 把【狗当前位置】当场景系原点（T = 单位阵）
    ② 图用空图（不避障），或者给一张简单图
    ③ 通过 /nav/navigate_to 发相对目标
       ⇒ 例: "往前走 2 米" = g=(2, 0, 0)

跑法（狗上）:
    bash ~/nav_ws/dogenv_run.sh ~/nav_ws/nav/demo/nav_node_real.py \
         --i-know-it-moves [--grid xxx.npz] [--dry-run]

安全:
    · 没 --i-know-it-moves  → 拒绝启动
    · --dry-run             → 位姿照收、规划照算，但指令【只打印不发出】
    · 任何退出路径都 settle()
"""
import argparse
import math
import os
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import Pose6D, FRAME_CAPO, FRAME_SCENE    # noqa: E402
from hal.pose_source import OdomSource, ScaledSource       # noqa: E402
from hal.cmd_sink import SportSink, DryRunSink             # noqa: E402
from hal.obstacle_source import RosObstacleSource          # noqa: E402
from maps.grid import OccupancyGrid                        # noqa: E402
from mission.api import Navigator, NavigateToConfig        # noqa: E402
from mission.ros_topics import RosTopicBridge              # noqa: E402
from control.fine_approach import ApproachConfig           # noqa: E402
from control.path_follower import FollowerConfig           # noqa: E402


def empty_grid() -> OccupancyGrid:
    """空图：没有静态障碍。用于"今天先验证链路，不做避障"。

    尺寸给足（±20m），免得目标点落到图外。
    """
    res = 0.10
    W = H = 400                       # 40m × 40m
    occ = np.zeros((H, W), dtype=bool)
    return OccupancyGrid(occ, (-20.0, -20.0), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--i-know-it-moves', action='store_true',
                    help='★ 不加这个就拒绝启动（这个节点真的会驱动狗）')
    ap.add_argument('--dry-run', action='store_true',
                    help='只算不发：位姿照收、规划照算，指令只打印')
    ap.add_argument('--grid', default=None,
                    help='栅格 npz；不给就用空图（不避障）')
    ap.add_argument('--start', nargs=3, type=float, default=None,
                    help='★ 可选快捷方式：启动时就用这个起点对齐 x y yaw（弧度）。'
                         '★ 不给（推荐）= 【等第一个 goal 的 s_* 来对齐】，'
                         '那时 goal 的 s_* 才是对齐的输入；'
                         '给了 = 启动即对齐，之后 goal 的 s_* 只做一致性校验')
    ap.add_argument('--action', action='store_true',
                    help='★ 用正式 Action 接口（nav_interfaces）而不是 JSON/topic')
    ap.add_argument('--dynamic-avoid', action='store_true',
                    help='★ 开动态避障（默认关）。需要障碍点源才有意义')
    ap.add_argument('--final-approach', default=None, metavar='D[,D2]',
                    help='★ 预瞄点（turn_shift 根治，默认【关】）：在终点【背后】d 米'
                         '（沿目标朝向的反方向）放一个点，让【最后一段路的方向=目标朝向】'
                         '⇒ 走到终点时已经朝对，不用"到了再大转"（大转会挪偏 ~10cm）。'
                         '可给阶梯（逗号分隔，按顺序试），如 0.8,0.6；都不可行则退回普通规划。'
                         '★ d 必须 > 0.5（跟随器在 0.5m 处交给末端精调），否则被截胡')
    ap.add_argument('--capo-scale', type=float, default=1.0,
                    help='★ CAPO 尺度校正：把 CAPO 报的位移【乘】这个数。'
                         '2026-09-23 实测 k = CAPO/卷尺 = 0.954（7 次，稳）'
                         '⇒ 填 1/0.954 = 1.048。默认 1.0 = 不校正。'
                         '不校正的后果：命令走 L 米，实际走 1.048L（多 4.8%%）')
    a, _ = ap.parse_known_args()

    if not a.i_know_it_moves:
        sys.exit(
            '\n★ 拒绝启动：这个节点会【真的驱动狗】。\n'
            '  确认狗在空旷处、遥控器在手、你已经同意之后，再加 --i-know-it-moves\n'
            '  如果只想验证链路不想让狗动，加 --dry-run\n')

    rclpy.init()
    node = Node('nav_main_real')
    log = lambda s: print('  [nav] %s' % s, flush=True)
    print('=' * 66)
    print('  导航节点（真机）' + ('   ★★ DRY-RUN：不会发指令 ★★' if a.dry_run else '   ★★ 会真的驱动狗 ★★'))
    print('=' * 66)

    # ---- 图 ----
    if a.grid:
        grid = OccupancyGrid.from_file(a.grid)
        log('用栅格: %s' % grid)
    else:
        grid = empty_grid()
        log('用空图（40m×40m，不避障）—— 今天先验链路')

    # ---- 位姿源 + 出口（★ 唯一改动的两个地方，逻辑一行没变）----
    capo_raw = OdomSource(node, '/SMX/Odom', frame=FRAME_CAPO, name='CAPO')
    if abs(a.capo_scale - 1.0) > 1e-9:
        capo = ScaledSource(capo_raw, a.capo_scale,
                            name='CAPO x%.3f' % a.capo_scale)
        log('位姿源: CAPO × %.3f  ★ 尺度校正（k=0.954 ⇒ 1/0.954=1.048）'
            % a.capo_scale)
    else:
        capo = capo_raw
        log('位姿源: CAPO  ★ 未做尺度校正（命令走 L 米实际会走 1.048L）')
    if a.dry_run:
        sink = DryRunSink(verbose=False)
        log('出口: DryRunSink（不发任何东西）')
    else:
        sink = SportSink(node, acknowledge_motion=True)
        log('出口: SportSink ★ 真的发给狗')

    # ★ 必须先把 spin 起起来，否则回调永远不触发
    # ★★★ Action 模式【必须用 MultiThreadedExecutor】：
    #     execute_callback 会阻塞到任务结束；单线程会把 500Hz 的 CAPO 订阅一起堵死。
    if a.action:
        from rclpy.executors import MultiThreadedExecutor
        _ex = MultiThreadedExecutor()
        _ex.add_node(node)
        threading.Thread(target=_ex.spin, daemon=True).start()
    else:
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cfg = NavigateToConfig(
        pos_tol=0.05, yaw_tol_deg=10.0,
        timeout=120.0, follow_timeout=90.0, approach_timeout=40.0,
        inflate_radius=0.35,
        approach=ApproachConfig(timeout=40.0),
        follower=FollowerConfig(timeout=90.0),
        dynamic_avoid=a.dynamic_avoid,
        final_approach_lengths=(tuple(float(v) for v in a.final_approach.split(','))
                                if a.final_approach else ()),
    )
    obstacles = RosObstacleSource(node) if a.dynamic_avoid else None
    nav = Navigator(capo, sink, cfg=cfg, grid=grid, log=log,
                    obstacle_source=obstacles)

    if not capo.wait_ready(20):
        print('✗ 20 秒拿不到 /SMX/Odom —— CAPO 起了吗？（bash ~/nav_ws/start_capo.sh）')
        return 2
    p0 = capo.get().data
    log('CAPO 就绪，当前读数 (%.3f, %.3f, %.1f°)'
        % (p0.x, p0.y, math.degrees(p0.yaw)))

    # ---- ★★ 开机对齐 ----
    # ★ 2026-09-27 改：对齐【默认不在启动时做】，改成等第一个 goal 的 s_* 来做
    #   （Navigator.ensure_aligned）。理由：文档一直写着"起点位姿从 goal 来"
    #   （action.md §5.1 / 联调手册 §二），但代码原来只认命令行 --start，
    #   goal 里的 s_* 【读都不读】—— 那条线是断的。现在接上了。
    #   ⇒ --start 退化成"我已经知道狗在哪"的快捷方式。
    if a.start is not None:
        nav.align(Pose6D(a.start[0], a.start[1], 0.0, 0.0, 0.0,
                         a.start[2], frame=FRAME_SCENE))
        log('★ 用 --start 显式对齐 @ 场景系 (%.3f, %.3f, %.1f° = %.4f rad)'
            % (a.start[0], a.start[1], math.degrees(a.start[2]), a.start[2]))
        log('  ⇒ 之后 goal 里的 s_* 只做【一致性校验】，不会重建 T')
    else:
        log('★ 未做开机对齐 —— 等【第一个 goal 的 s_*】来对齐（联调推荐这个）')
    log('等待 /nav/navigate_to ...')

    if a.action:
        from mission.ros_action import RosActionServer
        bridge = RosActionServer(node, nav, log=log)
    else:
        bridge = RosTopicBridge(node, nav, log=log)
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sink.settle()
        except Exception:
            pass
        os._exit(0)


if __name__ == '__main__':
    main()
