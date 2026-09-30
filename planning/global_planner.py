# -*- coding: utf-8 -*-
"""
planning/global_planner.py —— 全局路径规划

算法
----
① 从终点向外做【Dijkstra 距离场】（8 连通）    ★ 不是 FMM，但效果等价
② 从起点沿距离场梯度下降 → 格子路径
③ ★ 路径后处理：拉直（line-of-sight 捷径）+ 抽稀
      （Dijkstra 的 8 连通路径是锯齿状的，直接跟会抖、还会贴墙）

★ 和 CMU 那套的区别
-------------------
CMU 的 localPlanner 是【从 343 条预计算运动基元里选一条】——那是【局部避障】。
我们这里是【全局规划】：直接算出一条到终点的路径。
两层是不同的东西，不冲突（将来避障加在 planning/local_avoid.py）。

★ 为什么用 Dijkstra 不用 FMM
---------------------------
两者效果高度相似（都产出"到目标的距离场"，沿它走）。
Dijkstra 是栅格上的 8 连通距离，略锯齿；FMM 解 Eikonal 方程，梯度更平滑。
但我们取的是【沿路径的前视点】，不是直接跟梯度 ⇒ 差别不大。
将来发现路径太贴墙/太生硬，再换 FMM（UniGoal 里有现成的）。
"""
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.types import Pose2D, FRAME_SCENE
from maps.grid import NB, OccupancyGrid


@dataclass
class Plan:
    ok: bool
    waypoints: List[Pose2D] = field(default_factory=list)   # 场景系
    length: float = 0.0
    reason: str = ''
    grid_len: int = 0          # 平滑前的格子路径长度（诊断用）


# ================================================================ 距离场
def dijkstra_free(free: np.ndarray, goal_cell: Tuple[int, int]) -> np.ndarray:
    """从终点向外做 Dijkstra，返回距离场（单位：格）。free 是 bool 的"可走"掩码。"""
    H, W = free.shape
    dist = np.full((H, W), np.inf, dtype=np.float64)
    gy, gx = goal_cell
    if not free[gy, gx]:
        return dist
    dist[gy, gx] = 0.0
    pq = [(0.0, gy, gx)]
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy, dx, c in NB:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx]:
                nd = d + c
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, ny, nx))
    return dist


def nearest_free(free: np.ndarray, cell: Tuple[int, int],
                 max_r: int = 200) -> Optional[Tuple[int, int]]:
    """格子落在障碍/膨胀区时，找最近的自由格（环形 BFS）"""
    H, W = free.shape
    y0, x0 = cell
    if 0 <= y0 < H and 0 <= x0 < W and free[y0, x0]:
        return cell
    for r in range(1, max_r):
        best, bd = None, 1e18
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue
                y, x = y0 + dy, x0 + dx
                if 0 <= y < H and 0 <= x < W and free[y, x]:
                    d = dy * dy + dx * dx
                    if d < bd:
                        bd, best = d, (y, x)
        if best:
            return best
    return None


def extract_cells(dist: np.ndarray, free: np.ndarray,
                  start_cell: Tuple[int, int], max_steps: int = 20000):
    """从起点沿距离场梯度下降 → 格子路径"""
    y, x = start_cell
    path = [(y, x)]
    for _ in range(max_steps):
        bd, best = dist[y, x], None
        for dy, dx, _ in NB:
            ny, nx = y + dy, x + dx
            if (0 <= ny < dist.shape[0] and 0 <= nx < dist.shape[1]
                    and free[ny, nx] and dist[ny, nx] < bd):
                bd, best = dist[ny, nx], (ny, nx)
        if best is None:
            break
        path.append(best)
        y, x = best
        if bd <= 0.0:
            break
    return path


# ================================================================ ★ 路径平滑
def smooth_path(cells: List[Tuple[int, int]], grid: OccupancyGrid,
                min_seg_m: float = 0.15) -> List[Tuple[int, int]]:
    """★ 拉直 + 抽稀。

    Dijkstra 的 8 连通路径是锯齿状的（一格格拐），直接当路径跟会抖、还会贴墙。
    做法：贪心地往前找【最远的、视线可达的】那个点，保留它，跳过中间的。
    """
    if len(cells) <= 2:
        return list(cells)

    def world(c):
        return grid.cell_to_world(*c)

    out = [cells[0]]
    i = 0
    n = len(cells)
    while i < n - 1:
        j = min(i + 1, n - 1)
        # 往前探，找最远可见的点
        best = j
        k = j
        while k < n:
            if grid.line_clear(world(cells[i]), world(cells[k])):
                best = k
                k += 1
            else:
                break
        out.append(cells[best])
        i = best
    # 抽稀：去掉挨得太近的中间点
    if len(out) > 2:
        keep = [out[0]]
        for p in out[1:-1]:
            if math.dist(world(p), world(keep[-1])) >= min_seg_m:
                keep.append(p)
        keep.append(out[-1])
        out = keep
    return out


def densify(waypoints, max_seg: float = 0.50):
    """把路点之间插值加密，让相邻点距离 ≤ max_seg。

    ★ 为什么需要：
      新的跟随器是【逐路点：转 → 直走】的，走的时候 wz = 0（不修方向）。
      而狗走路天然往左偏 ~5cm/米 ⇒ 段越长漂得越多。
      段长 0.5m → 每段只漂 2.5cm，下一个路点重新对准就纠回来了。
      （★ 这是"用路点密度换跟踪精度"，代价是稍微慢一点）
    """
    if len(waypoints) < 2:
        return list(waypoints)
    out = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        d = math.dist((a.x, a.y), (b.x, b.y))
        n = max(1, int(math.ceil(d / max_seg)))
        for k in range(1, n + 1):
            t = k / n
            out.append(Pose2D(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                              math.atan2(b.y - a.y, b.x - a.x),
                              frame=FRAME_SCENE))
    out[-1] = Pose2D(waypoints[-1].x, waypoints[-1].y, waypoints[-1].yaw,
                     frame=FRAME_SCENE)
    return out


# ================================================================ 主入口
def plan_path(grid: OccupancyGrid, start: Pose2D, goal: Pose2D,
              inflate_radius: float = 0.35,
              smooth: bool = True,
              max_seg: float = 0.50) -> Plan:
    """场景系里，从 start 规划到 goal。

    inflate_radius: 车体半径。★ 必须 ≥ 车体半宽（GO2 是 0.35m），否则会蹭墙。
    """
    if start.frame != FRAME_SCENE or goal.frame != FRAME_SCENE:
        from core.types import FrameMismatch
        raise FrameMismatch('plan_path 只接受场景系（scene）的位姿：%s / %s'
                            % (start.frame, goal.frame))

    g = grid.inflated(inflate_radius)
    free = g.free_mask()

    sc = g.world_to_cell(start.x, start.y)
    gc = g.world_to_cell(goal.x, goal.y)
    sc = nearest_free(free, sc)
    gc = nearest_free(free, gc)
    if sc is None:
        return Plan(False, reason='start_blocked')
    if gc is None:
        return Plan(False, reason='goal_blocked')

    dist = dijkstra_free(free, gc)
    if not np.isfinite(dist[sc]):
        return Plan(False, reason='no_path')

    cells = extract_cells(dist, free, sc)
    raw_len = len(cells)
    if smooth:
        cells = smooth_path(cells, g)

    wp = []
    for (r, c) in cells:
        x, y = g.cell_to_world(r, c)
        wp.append(Pose2D(x, y, 0.0, frame=FRAME_SCENE))
    # 朝向：沿路径方向算，最后一个点用 goal 的朝向
    for i in range(len(wp) - 1):
        dx, dy = wp[i + 1].x - wp[i].x, wp[i + 1].y - wp[i].y
        yaw = math.atan2(dy, dx)
        wp[i] = Pose2D(wp[i].x, wp[i].y, yaw, frame=FRAME_SCENE)
    if wp:
        wp[-1] = Pose2D(wp[-1].x, wp[-1].y, goal.yaw, frame=FRAME_SCENE)
    # 端点用真实的起点/终点（不是格心），避免几个 cm 的系统偏差
    if wp:
        wp[0] = Pose2D(start.x, start.y, wp[0].yaw, frame=FRAME_SCENE)
        wp[-1] = Pose2D(goal.x, goal.y, goal.yaw, frame=FRAME_SCENE)

    # ★ 加密：跟随器是"逐路点转→直走"，段太长会漂
    wp = densify(wp, max_seg)

    L = sum(math.dist((wp[i].x, wp[i].y), (wp[i + 1].x, wp[i + 1].y))
            for i in range(len(wp) - 1))
    return Plan(True, wp, L, 'ok', raw_len)


def path_to_waypoint(plan: Plan, robot: Pose2D, lookahead: float = 0.8) -> Optional[Pose2D]:
    """沿路径取【前视点】—— ★ "滚动目标"就发生在这里（内部实现，不是对外接口）

    做法：找到路径上离机器人最近的点，再往前推 lookahead 米。
    """
    if not plan.ok or len(plan.waypoints) == 0:
        return None
    best_i, best_d = 0, 1e18
    for i, p in enumerate(plan.waypoints):
        d = (p.x - robot.x) ** 2 + (p.y - robot.y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    acc = 0.0
    i = best_i
    while i < len(plan.waypoints) - 1:
        seg = math.dist((plan.waypoints[i].x, plan.waypoints[i].y),
                        (plan.waypoints[i + 1].x, plan.waypoints[i + 1].y))
        if acc + seg >= lookahead:
            return plan.waypoints[i + 1]
        acc += seg
        i += 1
    return plan.waypoints[-1]
