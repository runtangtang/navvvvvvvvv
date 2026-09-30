#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analyze_lio_map.py —— 用【时间维度】分析建图质量

为什么要它（2026-09-26）
------------------------
v1 记录器把所有帧的点混在一起存，图糊了以后【查不下去】：
是全程都糊？还是某一段突然漂了？是狗自己的腿？还是旁边的人/网线？
⇒ v2 记录器多存了 frame_id 和 traj，这个脚本负责把它们用起来。

看什么
------
① 按【帧号】给点云染色 —— 同一块物理表面如果被不同时间段记到了不同位置，
   颜色就会分层 ⇒ 直接看出"哪一段开始漂"
② 画出 【Point-LIO 的轨迹】，按时间染色 —— 看轨迹本身有没有突变
③ 每帧的点数/质心随时间的变化 —— 突变 = 配准出问题的那一刻
④ 沿轨迹的"局部地面高度" —— 如果它一直单调漂，就是 z 漂

用法:
    python3 tools/analyze_lio_map.py <v2记录npz> [--out xxx.png]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('--out', default=None)
ap.add_argument('--sub', type=int, default=4, help='点云抽稀（画图用）')
a = ap.parse_args()

out = a.out or (os.path.splitext(a.npz)[0] + '_analyze.png')
d = np.load(a.npz, allow_pickle=True)
pts = d['pts']
print('=' * 64)
print('  分析: %s' % a.npz)
print('=' * 64)
print('  点 %d' % len(pts))

has_fid = 'frame_id' in d.files
has_traj = 'traj' in d.files and len(d['traj'])

if not has_fid:
    print('  ⚠️ 没有 frame_id —— 这是 v1 记录器存的，做不了时间分析')
    print('     （用 lio_map_recorder.py 重录一次就有了）')
if has_traj:
    traj = np.array(d['traj'])
    tt = np.array(d['traj_t'])
    tt = tt - tt[0]
    print('  轨迹 %d 条，时长 %.1f s' % (len(traj), tt[-1]))
    dx = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))
    print('  轨迹总长 %.2f m ｜ 净位移 %.2f m'
          % (dx.sum(), np.hypot(traj[-1, 0] - traj[0, 0],
                                traj[-1, 1] - traj[0, 1])))

import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402

if has_fid:
    fid = d['frame_id']
    fig, axs = plt.subplots(2, 2, figsize=(17, 13))

    # ① 按帧号染色
    ax = axs[0, 0]
    s = slice(None, None, a.sub)
    sc = ax.scatter(pts[s, 0], pts[s, 1], c=fid[s], s=0.4, cmap='turbo',
                    alpha=0.55)
    plt.colorbar(sc, ax=ax, label='frame index(=time)')
    ax.plot(0, 0, 'o', color='lime', ms=13, mec='k')
    ax.set_title('points colored by TIME  —  分层 = 漂移')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # ② 轨迹按时间染色
    ax = axs[0, 1]
    if has_traj:
        ax.scatter(traj[:, 0], traj[:, 1], c=tt, s=8, cmap='turbo')
        ax.plot(traj[0, 0], traj[0, 1], 's', color='k', ms=10, label='start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'o', color='lime', ms=12,
                mec='k', label='end')
        ax.legend(fontsize=8)
    ax.set_title('Point-LIO trajectory (colored by time)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # ③ 每帧点数随时间
    ax = axs[1, 0]
    nf = int(fid.max()) + 1
    cnt = np.bincount(fid, minlength=nf)
    ax.plot(cnt, lw=0.8)
    ax.set_xlabel('frame index'); ax.set_ylabel('points in frame')
    ax.set_title('points per frame  —  掉到 0 = 配准丢了')
    ax.grid(alpha=0.3)

    # ④ 每帧质心的 z（看 z 有没有单调漂）
    ax = axs[1, 1]
    cz = np.zeros(nf); cx = np.zeros(nf); cy = np.zeros(nf)
    for i in range(nf):
        m = fid == i
        if m.sum():
            cx[i] = pts[m, 0].mean(); cy[i] = pts[m, 1].mean()
            cz[i] = pts[m, 2].mean()
    ax.plot(cz, lw=1.0)
    ax.set_xlabel('frame index'); ax.set_ylabel('mean z of frame [m]')
    ax.set_title('per-frame mean z  —  单调漂 = z 漂移')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=85)
    print('  图已存: %s' % out)

    # ---- 数值：找"突变帧" ----
    print()
    print('  找突变（每帧质心相对上一帧跳 >0.1m 的）:')
    jump = np.hypot(np.diff(cx), np.diff(cy))
    idx = np.where(jump > 0.1)[0]
    if len(idx) == 0:
        print('    没有 >0.1m 的帧间跳跃')
    else:
        for i in idx[:12]:
            print('    frame %d → %d  质心跳 %.3f m' % (i, i + 1, jump[i]))
        if len(idx) > 12:
            print('    ...（共 %d 处）' % len(idx))
    print('=' * 64)
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pts[::a.sub, 0], pts[::a.sub, 1], s=0.4, alpha=0.5)
    ax.plot(0, 0, 'o', color='lime', ms=13, mec='k')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=85)
    print('  图已存: %s' % out)
