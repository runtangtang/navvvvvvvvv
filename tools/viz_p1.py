# -*- coding: utf-8 -*-
"""
tools/viz_p1.py —— P1 结果可视化

图 1  五条真值轨迹叠在一起（看"到没到"和"偏多少"）
图 2  误差归因：干净 / 尺度误差 / 延迟 / 比例控制，各占多少
图 3  ★ 尺度误差【随距离放大】：同一缺陷在 1m 和 3m 下的表现

用英文标签（中文在 matplotlib 里会乱码）。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/mnt/d/迷雾通/nav'
CASES = [
    ('clean-1m',   'no odometry error, 1 m',   'tab:green'),
    ('clean-3m',   'no odometry error, 3 m',   'tab:blue'),
    ('scale-3m',   'odometry scale -6%, 3 m',  'tab:red'),
    ('latency-1m', 'odometry latency 240 ms',  'tab:orange'),
    ('prop',       'proportional law (stalls)','tab:purple'),
]


def load(label):
    p = os.path.join(D, 'p1_%s.npz' % label)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d


def main():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    # ---------------- 图 1：离目标还有多远（随时间）----------------
    # 比画轨迹有用：轨迹都在 y≈0 上，看不出东西；这条曲线能看出
    #   收敛快不快 / 停在哪 / 有没有冲过头
    ax = axes[0]
    got = 0
    for label, title, col in CASES:
        d = load(label)
        if d is None or len(d['traj']) < 5:
            continue
        got += 1
        tr = d['traj']
        goal_d = float(d['meta'][0])
        dist = np.abs(goal_d - tr[:, 1])          # 沿 x 离目标多远
        t = tr[:, 0] - tr[0, 0]
        ax.plot(t, dist, color=col, lw=2.2, label=title)
        ax.plot(t[-1], dist[-1], 'o', color=col, ms=7)
    ax.axhline(0.03, color='k', ls='--', lw=1.2)
    ax.text(0.05, 0.033, '3 cm target', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(5e-4, 5)
    ax.set_title('A. Distance remaining vs time\n'
                 'marker = where it stopped', fontsize=11)
    ax.set_xlabel('time since start of run [s]')
    ax.set_ylabel('|distance to goal| [m]  (log)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3, which='both')
    if got == 0:
        ax.text(0.5, 0.5, 'no data yet', ha='center', transform=ax.transAxes)

    # ---------------- 图 2：误差归因 ----------------
    ax = axes[1]
    names, errs, cols = [], [], []
    for label, title, col in CASES:
        d = load(label)
        if d is None:
            continue
        names.append(title.replace(', ', '\n'))
        errs.append(float(d['meta'][1]))
        cols.append(col)
    if names:
        y = np.arange(len(names))
        ax.barh(y, errs, color=cols, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        for i, e in enumerate(errs):
            ax.text(e + max(errs) * 0.02, i, '%.3f m' % e, va='center', fontsize=9)
        ax.axvline(0.03, color='k', ls='--', lw=1.2)
        ax.text(0.031, len(names) - 0.5, 'our 3 cm target', fontsize=8, rotation=0)
        ax.set_xlabel('FINAL ERROR vs truth [m]')
        ax.set_title('B. Where the error actually comes from\n'
                     '(simulation truth, not our own estimate)', fontsize=11)
        ax.grid(alpha=0.3, axis='x')

    # ---------------- 图 3：尺度误差随距离放大 ----------------
    ax = axes[2]
    dists = np.linspace(0.2, 5.0, 60)
    for scale, col in ((1.00, 'tab:green'), (0.94, 'tab:red'), (0.90, 'tab:orange')):
        # 闭环在我们的坐标系里收敛 ⇒ 真实误差 ≈ (1/scale − 1)·d，滑行补偿是常数
        err = np.abs((1.0 / scale - 1.0) * dists)
        ax.plot(dists, err, color=col, lw=2.4,
                label='odometry scale %.2f (%.0f%%)' % (scale, (1 - scale) * 100))
    ax.axhline(0.03, color='k', ls='--', lw=1.2)
    ax.text(0.25, 0.035, '3 cm target', fontsize=8)
    # 实测点
    for label, mk in (('clean-1m', 'o'), ('clean-3m', 's'),
                      ('scale-3m', '^')):
        d = load(label)
        if d is None:
            continue
        ax.plot(float(d['meta'][0]), abs(float(d['meta'][1])), mk,
                color='k', ms=9, mfc='none', mew=1.8,
                label='measured: %s' % label)
    ax.set_xlabel('distance travelled [m]')
    ax.set_ylabel('final error [m]')
    ax.set_title('C. Scale error grows with distance\n'
                 '(coast compensation is constant, so it hides it at short range)',
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(D, 'tools', 'p1_result.png')
    plt.savefig(out, dpi=110)
    print('saved %s' % out)


if __name__ == '__main__':
    main()
