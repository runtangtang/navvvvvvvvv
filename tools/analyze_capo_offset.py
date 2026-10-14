#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 CAPO 原始日志中的位移方向、yaw 和速度。

输入格式：每行 ``t x y yaw_deg``，其中 t 是从 0 开始的相对时间。
默认把选中的运动窗口按样本数等分成 3 段；可用 ``--segments`` 调整。
"""

import argparse
import math
import sys

import numpy as np


def wrap_deg(angle):
    """把角度归一化到 [-180, 180)。"""
    return (float(angle) + 180.0) % 360.0 - 180.0


def load_log(path):
    try:
        data = np.loadtxt(path, comments='#', dtype=float, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError('无法读取日志 %s：%s' % (path, exc)) from exc
    if data.shape[1] != 4:
        raise ValueError('日志必须恰好有 4 列：t x y yaw_deg（实际 %d 列）'
                         % data.shape[1])
    if len(data) < 2:
        raise ValueError('日志至少需要 2 条样本')
    if not np.isfinite(data).all():
        raise ValueError('日志含 NaN 或无穷值')
    if np.any(np.diff(data[:, 0]) < 0.0):
        raise ValueError('时间列必须单调不减')
    return data


def select_window(data, from_t, to_t):
    if from_t is not None and to_t is not None and from_t > to_t:
        raise ValueError('--from-t 不能大于 --to-t')
    mask = np.ones(len(data), dtype=bool)
    if from_t is not None:
        mask &= data[:, 0] >= from_t
    if to_t is not None:
        mask &= data[:, 0] <= to_t
    selected = data[mask]
    if len(selected) < 2:
        raise ValueError('所选时间窗内至少需要 2 条样本')
    if selected[-1, 0] <= selected[0, 0]:
        raise ValueError('所选时间窗的持续时间必须大于 0')
    return selected


def analyze(data, requested_segments):
    t, x, y, yaw_deg = data.T
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_deg)))

    count = min(requested_segments, len(data) - 1)
    bounds = np.rint(np.linspace(0, len(data) - 1, count + 1)).astype(int)
    bounds = np.unique(bounds)

    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        dt = float(t[b] - t[a])
        if dt <= 0.0:
            continue
        dx = float(x[b] - x[a])
        dy = float(y[b] - y[a])
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            continue
        direction = math.degrees(math.atan2(dy, dx))
        rows.append({
            'a': int(a), 'b': int(b),
            't0': float(t[a]), 't1': float(t[b]),
            'direction': direction,
            'speed': distance / dt,
            'yaw_mean': float(np.mean(yaw_unwrapped[a:b + 1])),
            'distance': distance,
            'duration': dt,
        })

    if not rows:
        raise ValueError('所选时间窗没有可计算方向的非零位移段')

    directions = np.degrees(np.unwrap(np.radians(
        [row['direction'] for row in rows])))
    for row, direction in zip(rows, directions):
        row['direction_unwrapped'] = float(direction)

    spread = float(np.max(directions) - np.min(directions))
    mean_direction = float(np.mean(directions))
    mean_yaw = float(np.mean(yaw_unwrapped))
    offset = wrap_deg(mean_direction - mean_yaw)
    total_distance = sum(row['distance'] for row in rows)
    total_duration = sum(row['duration'] for row in rows)
    mean_speed = total_distance / total_duration
    return rows, spread, mean_direction, mean_yaw, offset, mean_speed


def print_report(rows, spread, mean_direction, mean_yaw, offset,
                 mean_speed, expect_speed):
    print('逐段结果（角度已跨 ±180° unwrap）')
    print('段  区间(s)             位移方向(°)  速度(m/s)  yaw均值(°)')
    for i, row in enumerate(rows, 1):
        print('%-3d %7.3f ~ %-7.3f %12.2f %11.3f %11.2f'
              % (i, row['t0'], row['t1'], row['direction_unwrapped'],
                 row['speed'], row['yaw_mean']))
    print()
    print('位移方向散布 = %.2f°' % spread)
    print('位移方向均值 = %.2f°' % mean_direction)
    print('yaw 均值      = %.2f°' % mean_yaw)
    print('偏移(位移方向-yaw) = %+.2f°' % offset)
    if spread < 10.0:
        print('判据结论：坐标系旋转（散布 < 10°）')
    else:
        print('判据结论：漂移或其它问题（散布 >= 10°）')
    if expect_speed is None:
        print('平均速度 = %.3f m/s' % mean_speed)
    else:
        delta = mean_speed - expect_speed
        pct = 100.0 * delta / expect_speed if expect_speed != 0.0 else float('nan')
        print('平均速度 = %.3f m/s；期望 = %.3f m/s；差值 = %+.3f m/s (%+.1f%%)'
              % (mean_speed, expect_speed, delta, pct))


def save_plot(path, data, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError) as exc:
        print('未生成 PNG：matplotlib 不可用（%s）' % exc, file=sys.stderr)
        return

    t, x, y, yaw_deg = data.T
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_deg)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, y, color='tab:blue', linewidth=1.5, label='CAPO path')
    for i, row in enumerate(rows, 1):
        a, b = row['a'], row['b']
        axes[0].plot([x[a], x[b]], [y[a], y[b]], 'o-', label='segment %d' % i)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_title('CAPO displacement segments')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(t, yaw_unwrapped, color='tab:orange', label='CAPO yaw')
    for row in rows:
        mid = 0.5 * (row['t0'] + row['t1'])
        axes[1].plot(mid, row['direction_unwrapped'], 'o', color='tab:blue')
    axes[1].plot([], [], 'o', color='tab:blue', label='displacement direction')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('angle [deg, unwrapped]')
    axes[1].set_title('Yaw vs displacement direction')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('PNG 已保存：%s' % path)


def main():
    parser = argparse.ArgumentParser(
        description='判断 CAPO 异常更像固定坐标系旋转还是漂移')
    parser.add_argument('log', help='每行 t x y yaw_deg 的 CAPO 原始日志')
    parser.add_argument('--from-t', type=float, help='只分析此相对时间之后（含）的样本')
    parser.add_argument('--to-t', type=float, help='只分析此相对时间之前（含）的样本')
    parser.add_argument('--expect-speed', type=float, help='期望速度，单位 m/s')
    parser.add_argument('--png', help='可选输出图路径')
    parser.add_argument('--segments', type=int, default=3,
                        help='所选窗口等分段数（默认 3）')
    args = parser.parse_args()
    if args.segments < 1:
        parser.error('--segments 必须 >= 1')

    try:
        data = select_window(load_log(args.log), args.from_t, args.to_t)
        result = analyze(data, args.segments)
    except ValueError as exc:
        parser.error(str(exc))

    print_report(*result, args.expect_speed)
    if args.png:
        save_plot(args.png, data, result[0])


if __name__ == '__main__':
    main()
