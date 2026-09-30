# -*- coding: utf-8 -*-
"""
maps/grid.py —— 占据栅格

★ 来源可换，上层不动
----------------------
    OccupancyGrid.from_file(...)      读 npz（现在用这个）
    OccupancyGrid.from_points(...)    Point-LIO 世界系点云 + 高度过滤
    OccupancyGrid.from_boxes(...)     组1 的物体包围盒（★ 留空，等接口）
    手写一个测试栅格也行

    ⇒ 换来源只改这里，planning / control 一行不动。

坐标
----
栅格在【场景系】里。origin 是格子 (0,0) 的【左下角】在场景系的坐标。
    x = origin[0] + col * res
    y = origin[1] + row * res
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.types import Pose2D, FRAME_SCENE

# 8 邻域: (drow, dcol, 步长)
NB = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356),
      (0, -1, 1.0),                       (0, 1, 1.0),
      (1, -1, 1.41421356),  (1, 0, 1.0),  (1, 1, 1.41421356)]


@dataclass
class OccupancyGrid:
    occ: np.ndarray                      # (H, W) bool, True = 占据
    origin: Tuple[float, float]          # 场景系坐标
    res: float                           # 米/格
    frame: str = FRAME_SCENE

    # ------------------------------------------------------------ 基本
    @property
    def shape(self):
        return self.occ.shape

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(math.floor((x - self.origin[0]) / self.res))
        row = int(math.floor((y - self.origin[1]) / self.res))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (self.origin[0] + (col + 0.5) * self.res,
                self.origin[1] + (row + 0.5) * self.res)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.occ.shape[0] and 0 <= col < self.occ.shape[1]

    def is_free_xy(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        return self.in_bounds(r, c) and not self.occ[r, c]

    def free_mask(self) -> np.ndarray:
        return ~self.occ

    # ------------------------------------------------------------ 构造
    @staticmethod
    def from_points(pts: np.ndarray, res: float = 0.05, pad: float = 1.0,
                    zmin: float = 0.05, zmax: float = 1.2,
                    extra_xy: Optional[Sequence[Tuple[float, float]]] = None,
                    bounds: Optional[Tuple[float, float, float, float]] = None,
                    ) -> 'OccupancyGrid':
        """障碍点 (N,3) 【场景系】→ 栅格。只取 z 在 [zmin,zmax) 的点。

        extra_xy: 额外要包进地图的点（起点/终点）——
                  否则地图只按障碍物包围盒算，机器人可能落在栅格外。
        """
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            raise ValueError('点云是空的')
        sel = pts[(pts[:, 2] >= zmin) & (pts[:, 2] < zmax)] if pts.shape[1] >= 3 else pts
        if len(sel) == 0:
            raise ValueError('没有 z∈[%.2f,%.2f) 的点 —— 改 zmin/zmax' % (zmin, zmax))
        x0, y0 = float(sel[:, 0].min()), float(sel[:, 1].min())
        x1, y1 = float(sel[:, 0].max()), float(sel[:, 1].max())
        for ex, ey in (extra_xy or []):
            x0, y0, x1, y1 = min(x0, ex), min(y0, ey), max(x1, ex), max(y1, ey)
        if bounds is not None:
            x0, y0, x1, y1 = bounds
        else:
            x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        W = int(math.ceil((x1 - x0) / res))
        H = int(math.ceil((y1 - y0) / res))
        occ = np.zeros((H, W), dtype=bool)
        ix = ((sel[:, 0] - x0) / res).astype(int).clip(0, W - 1)
        iy = ((sel[:, 1] - y0) / res).astype(int).clip(0, H - 1)
        occ[iy, ix] = True
        return OccupancyGrid(occ, (x0, y0), res)

    @staticmethod
    def from_file(path: str) -> 'OccupancyGrid':
        d = np.load(path)
        return OccupancyGrid(d['occ'].astype(bool),
                             tuple(float(v) for v in d['origin']),
                             float(d['res']))

    def save(self, path: str):
        np.savez(path, occ=self.occ, origin=np.array(self.origin), res=self.res)

    # ------------------------------------------------------------ 膨胀
    def inflated(self, radius_m: float) -> 'OccupancyGrid':
        """按车体半径膨胀。★ 半径要 ≥ 车体半宽，否则留不出余量。"""
        r = int(math.ceil(radius_m / self.res))
        if r <= 0:
            return OccupancyGrid(self.occ.copy(), self.origin, self.res, self.frame)
        H, W = self.occ.shape
        out = self.occ.copy()
        offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if dy * dy + dx * dx <= r * r]
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), min(H, H + dy)
            xs0, xs1 = max(0, dx), min(W, W + dx)
            yd0, yd1 = max(0, -dy), min(H, H - dy)
            xd0, xd1 = max(0, -dx), min(W, W - dx)
            out[yd0:yd1, xd0:xd1] |= self.occ[ys0:ys1, xs0:xs1]
        return OccupancyGrid(out, self.origin, self.res, self.frame)

    # ------------------------------------------------------------ 射线
    def line_clear(self, a: Tuple[float, float], b: Tuple[float, float],
                   step: Optional[float] = None) -> bool:
        """场景系两点之间是否无阻挡（按栅格采样，够用）"""
        step = step or (self.res * 0.5)
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        n = max(2, int(d / step) + 1)
        for i in range(n + 1):
            t = i / n
            if not self.is_free_xy(a[0] + dx * t, a[1] + dy * t):
                return False
        return True

    # ------------------------------------------------------------ 可视化
    def plot(self, ax, color_occ='0.75', label=None):
        ny, nx = self.occ.shape
        x0, y0 = self.origin
        ax.imshow(self.occ, origin='lower', cmap='Greys', vmin=0, vmax=1.4,
                  extent=[x0, x0 + nx * self.res, y0, y0 + ny * self.res],
                  alpha=0.9, zorder=1, label=label)
        return ax

    def __repr__(self):
        return 'OccupancyGrid(%dx%d @ %.3fm, origin=(%.2f,%.2f), frame=%s)' % (
            self.occ.shape[1], self.occ.shape[0], self.res,
            self.origin[0], self.origin[1], self.frame)


# ================================================================ 手工造测试场
def make_test_world(kind: str = 'utrap', res: float = 0.05) -> OccupancyGrid:
    """手写测试场景（不依赖任何数据源）。

    'empty'  空地，只有外围墙
    'utrap'  U 形陷阱 —— 复刻 2026-09-21 验收用的那个：直冲进去就出不来
    'corridor' 带缺口的墙 + 柱子
    """
    W, H = 240, 200            # 12m × 10m
    occ = np.zeros((H, W), dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True

    def rect(x0, y0, x1, y1):
        c0, c1 = int(x0 / res), int(x1 / res)
        r0, r1 = int(y0 / res), int(y1 / res)
        occ[r0:r1, c0:c1] = True

    if kind == 'utrap':
        # U 形：底边 + 两侧竖墙，开口朝 +y（上方）
        rect(5.0, 3.0, 7.6, 3.2)      # 底
        rect(5.0, 3.0, 5.2, 6.2)      # 左壁
        rect(7.4, 3.0, 7.6, 6.2)      # 右壁
    elif kind == 'corridor':
        # 一堵带缺口的墙（x=7.5，缺口 y∈[5.65,6.7]）+ 三根柱子
        for y0, y1 in ((2.0, 5.65), (6.7, 9.0)):
            rect(7.5, y0, 7.7, y1)
        for cx, cy in ((3.0, 7.5), (4.5, 2.5), (9.5, 4.0)):
            rect(cx - 0.15, cy - 0.15, cx + 0.15, cy + 0.15)
    return OccupancyGrid(occ, (0.0, 0.0), res)
