# -*- coding: utf-8 -*-
"""
tools/viz_scene.py —— 把场景图 + 当前坐标系画出来，供对照标注

出三张图:
   scene_frame.png        ★ 主图：只画网格 + 坐标系（清楚，适合标注对比）
   scene_with_axes.png    带坐标轴标题的版本
   scene_clean.png        干净底图，随便标

★ 画图一律用英文标签（中文在 matplotlib 里乱码）
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, '/mnt/d/迷雾通/nav')
from maps.grid import OccupancyGrid

D = '/mnt/d/迷雾通/nav'
SCENE = D + '/demo_scene.npz'
d = np.load(SCENE)
occ = d['occ'].astype(bool)
mo, so, res = d['origin'], d['scene_origin'], float(d['res'])
g = OccupancyGrid(occ, (float(mo[0] - so[0]), float(mo[1] - so[1])), res)
H, W = occ.shape
x0, y0 = g.origin
EXT = [x0, x0 + W * res, y0, y0 + H * res]


def draw_frame(ax, arrow_len=1.4, alpha=0.55):
    ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.6,
              extent=EXT, alpha=alpha)
    ax.plot(0, 0, 'o', color='crimson', ms=15, mec='k', zorder=8)
    for v, nm in (((arrow_len, 0), 'x'), ((0, arrow_len), 'y')):
        ax.annotate('', xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='crimson',
                                    lw=3.2, shrinkA=0, shrinkB=0), zorder=9)
        ax.text(v[0] * 1.12, v[1] * 1.12, nm, color='crimson',
                fontsize=22, fontweight='bold', zorder=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(alpha=0.45, zorder=2)
    ax.set_aspect('equal')
    ax.set_xlabel('scene x [m]')
    ax.set_ylabel('scene y [m]')


# ---------------- ① 主图：坐标系总览（用来对照标注） ----------------
fig, ax = plt.subplots(figsize=(12, 10))
draw_frame(ax, arrow_len=1.4, alpha=0.5)
ax.text(0.15, -0.62,
        'SCENE origin (0,0)\nthis is where the dog starts (facing +x)',
        color='crimson', fontsize=10, fontweight='bold', zorder=10,
        bbox=dict(boxstyle='round', fc='#ffecec', ec='crimson'))
ax.set_title('SCENE FRAME  (current)\n'
             'red dot = origin (0,0)      red arrows = x / y axes\n'
             'mark YOUR expected origin & axes on this and compare',
             fontsize=12)
plt.tight_layout()
plt.savefig(D + '/tools/scene_frame.png', dpi=110)
print('saved scene_frame.png')

# ---------------- ② 带障碍的版本 ----------------
fig2, ax2 = plt.subplots(figsize=(12, 10))
draw_frame(ax2, arrow_len=1.2, alpha=1.0)
ax2.set_title('SCENE MAP + CURRENT SCENE FRAME\n'
              'axis numbers = SCENE coordinates (the frame Group 1 defines)\n'
              'the dog always starts at (0,0) facing +x', fontsize=12)
plt.tight_layout()
plt.savefig(D + '/tools/scene_with_axes.png', dpi=110)
print('saved scene_with_axes.png')

# ---------------- ③ 干净底图 ----------------
fig3, ax3 = plt.subplots(figsize=(12, 10))
ax3.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35, extent=EXT)
ax3.set_title('SCENE MAP (clean) - scene coordinates', fontsize=12)
ax3.set_xlabel('scene x [m]')
ax3.set_ylabel('scene y [m]')
ax3.set_aspect('equal')
plt.tight_layout()
plt.savefig(D + '/tools/scene_clean.png', dpi=110)
print('saved scene_clean.png')
