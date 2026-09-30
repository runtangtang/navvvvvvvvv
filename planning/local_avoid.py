# -*- coding: utf-8 -*-
"""沿已规划路径的局部动态障碍停车层（纯几何，不依赖 ROS）。"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from core.types import FRAME_SCENE, FrameMismatch, Pose2D


@dataclass(frozen=True)
class AvoidConfig:
    enabled: bool = False
    lookahead_m: float = 2.0
    # ★★ corridor_m 必须【小于】规划膨胀半径 inflate_radius（默认 0.35）★
    #
    #   为什么（2026-09-28 实测，见 tools/check_corridor_vs_inflate.py）：
    #     规划按 0.35 膨胀 ⇒ 路径上任意点离最近【静态】障碍 ≥0.35m，
    #     而且路径常常【贴着这个边界走】（见 FollowerConfig 里 inflate_radius 的注释）。
    #     如果 corridor_m ≥ 0.35 ⇒ 那些静态障碍点会被判"在走廊里" ⇒ ★ 一出发就误停。
    #     实测（墙离路径 0.35m，20 个点）：
    #         corridor_m=0.40 ⇒ 命中 20 点 ⇒ 停   ← ✗ 误触发
    #         corridor_m=0.25 ⇒ 命中  0 点 ⇒ 不停  ← ✓
    #     墙在 0.30~0.36 时，0.40 全都会误停。
    #
    #   ★ 取 0.25 的道理：只对"比规划承诺的更近的东西"报警。
    #     真机有点云噪声/抖动时会偏保守 —— 那是安全方向，可接受；现场可调大试。
    corridor_m: float = 0.25
    min_points: int = 2
    hold_s: float = 1.0


class ObstacleSource(Protocol):
    """场景系障碍点源；None 表示失效，空数组表示确认无障碍。"""

    def get(self) -> Optional[np.ndarray]:
        ...

    def healthy(self, max_age: float = 1.0) -> bool:
        ...

    def error_count(self) -> int:
        ...


def _require_scene(pose: Pose2D, where: str) -> None:
    if pose.frame != FRAME_SCENE:
        raise FrameMismatch('%s 只接受场景系位姿，收到 frame=%r'
                            % (where, pose.frame))


def corridor_polyline(wps: Sequence[Pose2D], pose: Pose2D, i_wp: int,
                      goal: Pose2D, lookahead_m: float) -> List[Tuple[float, float]]:
    """从当前路段投影点沿路径向前截取 ``lookahead_m`` 米的折线。"""
    _require_scene(pose, 'corridor_polyline(pose)')
    _require_scene(goal, 'corridor_polyline(goal)')
    if not wps:
        return []
    for wp in wps:
        _require_scene(wp, 'corridor_polyline(wps)')

    n = len(wps)
    i_wp = min(max(int(i_wp), 1), n - 1) if n > 1 else 0
    a = wps[max(0, i_wp - 1)]
    b = goal if i_wp >= n - 1 else wps[i_wp]
    dx, dy = b.x - a.x, b.y - a.y
    seg = float(np.hypot(dx, dy))
    if seg > 1e-12:
        ux, uy = dx / seg, dy / seg
        s = (pose.x - a.x) * ux + (pose.y - a.y) * uy
        s = min(max(s, 0.0), seg)
        start = (a.x + s * ux, a.y + s * uy)
    else:
        start = (b.x, b.y)

    out = [(float(start[0]), float(start[1]))]
    remaining = max(float(lookahead_m), 0.0)
    if remaining <= 0.0:
        return out

    # 当前段终点之后，依次走剩余路点；goal 作为唯一的最终端点。
    future = [(float(b.x), float(b.y))]
    if i_wp < n - 1:
        future.extend((float(wps[j].x), float(wps[j].y))
                      for j in range(i_wp + 1, n - 1))
        future.append((float(goal.x), float(goal.y)))

    cur = out[0]
    for end in future:
        sx, sy = end[0] - cur[0], end[1] - cur[1]
        length = float(np.hypot(sx, sy))
        if length <= 1e-12:
            continue
        if length <= remaining:
            out.append(end)
            remaining -= length
            cur = end
            if remaining <= 1e-12:
                break
        else:
            ratio = remaining / length
            out.append((cur[0] + sx * ratio, cur[1] + sy * ratio))
            break
    return out


def points_in_corridor(pts: np.ndarray,
                       polyline: Sequence[Tuple[float, float]],
                       corridor_m: float) -> np.ndarray:
    """返回到折线任一有限线段距离小于 ``corridor_m`` 的点。"""
    if pts is None:
        raise ValueError('pts=None means obstacle sensing failed')
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError('障碍点必须是 (N, 3)，收到 %r' % (arr.shape,))
    if len(arr) == 0 or len(polyline) < 2 or corridor_m <= 0.0:
        return arr[:0]

    xy = arr[:, :2]
    inside = np.zeros(len(arr), dtype=bool)
    finite = np.isfinite(arr).all(axis=1)
    for p0, p1 in zip(polyline[:-1], polyline[1:]):
        a = np.asarray(p0, dtype=float)
        d = np.asarray(p1, dtype=float) - a
        denom = float(np.dot(d, d))
        if denom <= 1e-24:
            continue
        t = ((xy - a) @ d) / denom
        on_segment = (t >= 0.0) & (t <= 1.0)
        nearest = a + t[:, None] * d
        dist2 = np.sum((xy - nearest) ** 2, axis=1)
        inside |= finite & on_segment & (dist2 < float(corridor_m) ** 2)
    return arr[inside]


def should_stop(pts: Optional[np.ndarray],
                polyline: Sequence[Tuple[float, float]],
                cfg: AvoidConfig) -> Tuple[bool, int]:
    """感知失效或走廊内有效点达到阈值时请求停车。"""
    if pts is None:
        return True, 0
    hit = points_in_corridor(pts, polyline, cfg.corridor_m)
    count = int(len(hit))
    return count >= max(1, int(cfg.min_points)), count
