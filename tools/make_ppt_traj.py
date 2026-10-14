#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑一次仿真 + 抓轨迹/路径 → 出一张干净的 PPT 图（只画图，不放说明文字）

用法（WSL）:
    python3 tools/make_ppt_traj.py <栅格npz> <任务 x,y,yaw> <输出png> <标题>

做法:
  ① 起 sim_dog + nav_node_sim（和 run_cube_sim.sh 一样的环境）
  ② 订阅 /nav/feedback（10Hz，含当前位置）和 /nav/planned_path（一次）
  ③ 跑完把轨迹 + 路径 + 路点画在一张图上
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('grid')
ap.add_argument('task')            # "x,y,yaw"
ap.add_argument('out')
ap.add_argument('title')
a = ap.parse_args()

DOMAIN = '7'
os.environ['ROS_DOMAIN_ID'] = DOMAIN
os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
os.environ.pop('CYCLONEDDS_URI', None)

gx, gy, gyaw = [float(v) for v in a.task.split(',')]

# ---------- 起仿真 ----------
subprocess.run('pkill -f sim_dog; sleep 1', shell=True)
sim = subprocess.Popen(['python3', '-u', 'sim_dog.py', '--seed', '42',
                        '--x0', '0', '--y0', '0', '--yaw0', '0'],
                       stdout=open('/tmp/ppt_simdog.log', 'w'),
                       stderr=subprocess.STDOUT)
time.sleep(4)
node = subprocess.Popen(['python3', '-u', 'demo/nav_node_sim.py',
                         '--i-know-it-moves', '--grid', a.grid,
                         '--start', '0', '0', '0'],
                        stdout=open('/tmp/ppt_node.log', 'w'),
                        stderr=subprocess.STDOUT)
time.sleep(6)

# ---------- 另一个进程发目标 + 收数据 ----------
import rclpy                                             # noqa: E402
from rclpy.node import Node                              # noqa: E402
from std_msgs.msg import String                          # noqa: E402

rclpy.init()
n = Node('ppt_collect')
traj, paths = [], []


def fb_cb(m):
    try:
        d = json.loads(m.data)
        traj.append((float(d['cur_x']), float(d['cur_y']), float(d.get('cur_yaw', 0.0))))
    except Exception:
        pass


def path_cb(m):
    try:
        d = json.loads(m.data)
        pts = [(float(p[0]), float(p[1])) for p in d['waypoints']]
        if pts:
            paths.append(pts)
    except Exception:
        pass


n.create_subscription(String, '/nav/feedback', fb_cb, 50)
n.create_subscription(String, '/nav/planned_path', path_cb, 50)
pub = n.create_publisher(String, '/nav/navigate_to', 10)
DONE = []


def res_cb(m):
    try:
        DONE.append(json.loads(m.data))
    except Exception:
        pass


n.create_subscription(String, '/nav/result', res_cb, 50)
time.sleep(1)

goal = {"s_x": 0.0, "s_y": 0.0, "s_z": 0.0, "s_roll": 0.0, "s_pitch": 0.0, "s_yaw": 0.0,
        "g_x": gx, "g_y": gy, "g_z": 0.0, "g_roll": 0.0, "g_pitch": 0.0, "g_yaw": gyaw,
        "pos_tol": 0.05, "yaw_tol": 0.03, "timeout": 90.0, "max_speed": 0.0}
msg = String()
msg.data = json.dumps(goal)
for _ in range(5):
    pub.publish(msg)
    rclpy.spin_once(n, timeout_sec=0.1)
    time.sleep(0.1)

t0 = time.time()
while time.time() - t0 < 120:
    rclpy.spin_once(n, timeout_sec=0.1)
    # ★ 只有收到 /nav/result 才算跑完（不要用"位置没变"判断 ——
    #   转向阶段狗也是停着的，会被误判）
    if DONE:
        print('  result:', DONE[0].get('reason'), '成功=%s' % DONE[0].get('success'))
        break

subprocess.run('pkill -f sim_dog; pkill -f nav_node_sim', shell=True)
print('  轨迹 %d 点，路径 %d 条' % (len(traj), len(paths)))
if traj:
    print('  终点 (%.3f, %.3f)  离目标 %.3f m'
          % (traj[-1][0], traj[-1][1], math.hypot(traj[-1][0] - gx, traj[-1][1] - gy)))

# ---------- 画图（★ 只画图，不放说明文字） ----------
grid = np.load(a.grid, allow_pickle=True)
occ = grid['occ']
o = grid['origin']
res = float(grid['res'])
ny, nx = occ.shape
ext = (o[0], o[0] + nx * res, o[1], o[1] + ny * res)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.imshow(occ, origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=2, alpha=0.45)

for i, pts in enumerate(paths):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, 'r--', lw=1.6, label='planned path' if i == 0 else None)
    # ★ 路点
    ax.plot(xs, ys, 'r.', ms=7, label='waypoints' if i == 0 else None)

if traj:
    T = np.array(traj)
    ax.plot(T[:, 0], T[:, 1], 'C0-', lw=2.2, label='travelled')
    for k in range(0, len(T), max(1, len(T) // 14)):
        ax.arrow(T[k, 0], T[k, 1], 0, 0, head_width=0.07, head_length=0.07,
                 fc='C0', ec='C0', length_includes_head=False)
    ax.plot(T[0, 0], T[0, 1], 'ks', ms=11, label='start')
    ax.plot(T[-1, 0], T[-1, 1], 'o', color='lime', mec='k', ms=13, label='end')

ax.plot(gx, gy, '*', color='gold', mec='k', ms=18, label='goal')
ax.set_aspect('equal')
ax.set_title(a.title, fontsize=13)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid(alpha=0.25)
ax.legend(loc='best', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(a.out, dpi=130)
print('  图已存:', a.out)
