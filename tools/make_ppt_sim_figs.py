#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect isolated local ROS simulations and draw PPT trajectory comparisons.

Existing navigation/simulator code is executed unchanged. No robot access.
  python tools/make_ppt_sim_figs.py --run
  python tools/make_ppt_sim_figs.py --data-dir tools/ppt_sim_RUN_DIRECTORY
Run with the existing WSL conda ros2 Python. No installation is needed.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np

TOOLS = Path(__file__).resolve().parent
NAV = TOOLS.parent
SCENES = {
    'narrow': {'grid': 'lab_scene.npz', 'title': '窄口（缝 0.8m）',
               'reference_error_m': .0427, 'reference_elapsed_s': 30.0, 'domain': 147,
               'crop': [-.85, 2.30, -.45, 2.55]},
    'arc': {'grid': 'arc_scene.npz', 'title': '弧形走廊',
            'reference_error_m': .0299, 'reference_elapsed_s': 31.5, 'domain': 148,
            'crop': [-.75, 2.30, -.75, 2.75]},
}
GOAL = np.array([.5, 2., 0.])
# Bound both discovery and data to the local loopback; never inherit domain 7.
DDS_XML = ('<CycloneDDS><Domain id="any"><General>'
           '<Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces>'
           '<AllowMulticast>false</AllowMulticast></General><Discovery>'
           '<ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>20</MaxAutoParticipantIndex>'
           '<Peers><Peer Address="127.0.0.1"/></Peers>'
           '</Discovery></Domain></CycloneDDS>')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolated_environment(domain):
    api = Path('/home/runtangtang/ros2_ws/install/unitree_api')
    if not (api / 'lib/python3.10/site-packages/unitree_api/msg').is_dir():
        raise RuntimeError('Existing unitree_api installation not found: ' + str(api))
    env = os.environ.copy()
    # ROS_LOCALHOST_ONLY=1 makes this RMW inject an additional lo interface.
    # The XML is authoritative: exactly lo, no multicast, only 127.0.0.1 peers.
    env.pop('ROS_LOCALHOST_ONLY', None)
    env.update(ROS_DOMAIN_ID=str(domain),
               RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', CYCLONEDDS_URI=DDS_XML,
               PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for key, entries in {
        'PYTHONPATH': [str(api / 'lib/python3.10/site-packages')],
        'LD_LIBRARY_PATH': [str(api / 'lib'), str(Path(sys.prefix) / 'lib')],
        'AMENT_PREFIX_PATH': [str(api), sys.prefix],
    }.items():
        env[key] = ':'.join(entries + ([env[key]] if env.get(key) else []))
    return env


def stop_owned(process):
    """Only signal the new process group belonging to our Popen child."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def collect_scene(name, directory):
    cfg = SCENES[name]
    if (os.environ.get('ROS_DOMAIN_ID') != str(cfg['domain'])
            or os.environ.get('ROS_LOCALHOST_ONLY') is not None
            or os.environ.get('CYCLONEDDS_URI') != DDS_XML
            or os.environ.get('RMW_IMPLEMENTATION') != 'rmw_cyclonedds_cpp'):
        raise RuntimeError('Collector must be started by --run with loopback DDS isolation')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    # Fail before starting any publisher if the existing message ABI is unavailable.
    from unitree_api.msg import Request
    Request.__class__.__import_type_support__()

    class Capture(Node):
        def __init__(self):
            super().__init__('ppt_sim_capture_%d' % os.getpid())
            self.feedback, self.truth, self.odom, self.paths = [], [], [], []
            self.result = None
            self.goal_time = None
            self.goals = []
            self.result_time = None
            qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(String, '/nav/feedback', self.on_feedback, qos)
            self.create_subscription(String, '/nav/planned_path', self.on_path, qos)
            self.create_subscription(String, '/nav/result', self.on_result, qos)
            self.create_subscription(String, '/nav/navigate_to', self.on_goal, qos)
            self.create_subscription(Odometry, '/sim/truth', lambda m: self.on_odom(m, self.truth), qos)
            self.create_subscription(Odometry, '/SMX/Odom', lambda m: self.on_odom(m, self.odom), qos)
            self.goal_pub = self.create_publisher(String, '/nav/navigate_to', qos)

        def on_feedback(self, msg):
            d = json.loads(msg.data)
            self.feedback.append([time.time(), d['cur_x'], d['cur_y'], d['cur_yaw'],
                                  d.get('stage', ''), d.get('remaining_dist', float('nan'))])

        def on_path(self, msg):
            self.paths.append(json.loads(msg.data))

        def on_result(self, msg):
            self.result = json.loads(msg.data)
            self.result_time = time.time()

        def on_goal(self, msg):
            self.goals.append(json.loads(msg.data))
            self.goal_time = time.time()

        def on_odom(self, msg, target):
            p = msg.pose.pose
            q = p.orientation
            yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            target.append([time.time(), stamp, p.position.x, p.position.y, yaw])

    directory.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    rclpy.init()
    capture = Capture()

    def start(script, args, label):
        logfile = (directory / (name + '_' + label + '.log')).open('w', encoding='utf-8')
        logs.append(logfile)
        child = subprocess.Popen([sys.executable, '-u', str(NAV / script)] + args,
                                 cwd=NAV, env=os.environ.copy(), start_new_session=True,
                                 stdout=logfile, stderr=subprocess.STDOUT)
        processes.append(child)
        return child

    def spin_for(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(capture, timeout_sec=.03)
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A simulation process exited unexpectedly; inspect raw logs')

    try:
        # Same simulator, initial pose, seed, navigation config and client as run_cube_sim.sh.
        sim = start('sim_dog.py', ['--seed', '42', '--x0', '0.0', '--y0', '0.0', '--yaw0', '0'], 'simdog')
        spin_for(4)
        if not capture.truth or not capture.odom or sim.poll() is not None:
            raise RuntimeError('Local simulator did not supply truth and odometry')
        nav = start('demo/nav_node_sim.py', ['--start', '0', '0', '0', '--grid',
                                           str(TOOLS / cfg['grid'])], 'navnode')
        spin_for(5)
        allowed = {capture.get_name(), 'sim_dog', 'nav_main'}
        found = {node for node, namespace in capture.get_node_names_and_namespaces()}
        if found != allowed or nav.poll() is not None:
            raise RuntimeError('Unexpected nodes in isolated simulation domain: ' + repr(found))
        # group2_sim sends once after a fixed 3 s. With multicast disabled that
        # can precede endpoint discovery. Send its identical goal only once the
        # capture and navigation subscriptions are both matched; no code changes.
        spin_for(3)
        ready_deadline = time.monotonic() + 15
        while (capture.goal_pub.get_subscription_count() < 2
               or any(capture.count_publishers(topic) != 1 for topic in (
                   '/nav/feedback', '/nav/planned_path', '/nav/result'))):
            if time.monotonic() > ready_deadline:
                raise RuntimeError('Navigation topic discovery did not become ready')
            spin_for(.1)
        goal_dict = dict(s_x=0., s_y=0., s_z=0., s_roll=0., s_pitch=0., s_yaw=0.,
                         g_x=.5, g_y=2., g_z=0., g_roll=0., g_pitch=0., g_yaw=0.,
                         pos_tol=.05, yaw_tol=.03, timeout=90., max_speed=0.)
        goal_msg = String()
        goal_msg.data = json.dumps(goal_dict)
        (directory / (name + '_client.log')).write_text(
            'LOCAL goal sender: endpoints ready; same payload as group2_sim.py\n'
            + json.dumps(goal_dict, indent=2) + '\n', encoding='utf-8')
        capture.goal_pub.publish(goal_msg)
        deadline = time.monotonic() + 135
        while capture.result is None and time.monotonic() < deadline:
            spin_for(.10)
        if capture.result is None:
            raise RuntimeError('No result within 135 seconds')
        spin_for(3)  # Original script waits 3 s after result; also capture settled truth.
        if len(capture.goals) != 1 or len(capture.paths) != 1:
            raise RuntimeError('Expected exactly one goal and one unmodified planned path')
        if not capture.result['success'] or capture.result['reason'] != 'arrived':
            raise RuntimeError('Simulation failed: ' + repr(capture.result))
        fb = [row for row in capture.feedback if row[0] >= capture.goal_time]
        truth = np.array([row for row in capture.truth if row[0] >= capture.goal_time])
        odom = np.array([row for row in capture.odom if row[0] >= capture.goal_time])
        traj = np.array([row[1:4] for row in fb], dtype=float)
        if len(traj) < 30 or not np.isfinite(traj).all():
            raise RuntimeError('Insufficient or nonfinite feedback trajectory')
        grid_path = TOOLS / cfg['grid']
        with np.load(grid_path) as g:
            occ = g['occ'].astype(bool)
            map_origin, scene_origin = g['origin'], g['scene_origin']
            origin = map_origin - scene_origin
            resolution = float(g['res'])
        final = [capture.result['final_x'], capture.result['final_y'], capture.result['final_yaw']]
        metadata = {
            'scene': name, 'grid': str(grid_path), 'grid_sha256': sha256(grid_path),
            'start_scene': [0., 0., 0.], 'goal_scene': GOAL.tolist(),
            'goal_json': capture.goals[0], 'result': capture.result,
            'goal_time': capture.goal_time, 'result_time': capture.result_time,
            'reference_error_m': cfg['reference_error_m'],
            'reference_elapsed_s': cfg['reference_elapsed_s'],
            'ROS_DOMAIN_ID': cfg['domain'], 'DDS_interface': '127.0.0.1',
            'DDS_multicast': False, 'CYCLONEDDS_URI': DDS_XML,
            'raw_log_directory': str(directory),
            'trajectory_source': '/nav/feedback (default 10Hz), unfiltered; final pose from /nav/result',
            'planned_path_source': '/nav/planned_path, original waypoints, not replanned for plotting',
            'truth_source': '/sim/truth (default 100Hz), for consistency and collision checks',
            'simulator_cli': '--seed 42 --x0 0.0 --y0 0.0 --yaw0 0',
            'nav_cli': '--start 0 0 0 --grid ' + str(grid_path),
            'client_cli': 'Equivalent to group2_sim.py --tasks 0.5,2.0,0.0 --yaw-tol 0.03; await topic discovery then publish identical goal',
            'existing_source_sha256': {p: sha256(NAV / p) for p in (
                'run_cube_sim.sh', 'sim_dog.py', 'demo/nav_node_sim.py', 'demo/group2_sim.py',
                'control/path_follower.py', 'control/fine_approach.py', 'mission/ros_topics.py')},
        }
        np.savez_compressed(directory / (name + '.npz'),
                            traj=traj, traj_t=np.array([row[0] for row in fb]),
                            stage_names=np.array([row[4] for row in fb]),
                            stage_remain=np.array([row[5] for row in fb]),
                            path_0=np.array(capture.paths[0]['waypoints']), n_path=1, n_result=1,
                            truth=truth, odom=odom, occ=occ, origin=origin,
                            map_origin=map_origin, scene_origin=scene_origin, res=resolution,
                            result_final=np.array(final), goal=GOAL,
                            metadata_json=json.dumps(metadata, ensure_ascii=True, allow_nan=False))
        print('COLLECTED %s: feedback=%d truth=%d arrived elapsed=%.3fs internal_error=%.6fm' % (
            name, len(traj), len(truth), capture.result['elapsed'], capture.result['internal_error']), flush=True)
    finally:
        # Stop navigation before the simulator so its final settle stays local and is received.
        for child in reversed(processes):
            stop_owned(child)
        capture.destroy_node()
        rclpy.shutdown()
        for logfile in logs:
            logfile.close()


def occupancy_hits(xy, occ, origin, res):
    rc = np.floor((xy - origin) / res).astype(int)
    inside = (rc[:, 0] >= 0) & (rc[:, 0] < occ.shape[1]) & (rc[:, 1] >= 0) & (rc[:, 1] < occ.shape[0])
    if not inside.all():
        raise RuntimeError('Trajectory/path outside stored scene grid; check coordinate alignment')
    return int(np.count_nonzero(occ[rc[:, 1], rc[:, 0]]))


def dense_polyline(xy, step):
    points = []
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b-a) / step)) + 1)
        points.append(a + np.linspace(0, 1, n)[:, None] * (b-a))
    return np.concatenate(points)


def path_distances(xy, path):
    distances = np.full(len(xy), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        delta = b-a
        denom = float(delta @ delta)
        t = np.clip((xy-a) @ delta / max(denom, 1e-20), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(xy - (a + t[:, None]*delta), axis=1))
    return distances


def load_validate(path):
    with np.load(path, allow_pickle=False) as d:
        data = {key: d[key].copy() for key in d.files}
    meta = json.loads(str(data['metadata_json']))
    name, result = meta['scene'], meta['result']
    traj, planned = data['traj'], data['path_0']
    occ, origin, res = data['occ'], data['origin'], float(data['res'])
    truth, odom = data['truth'], data['odom']
    final = data['result_final']
    if (not result['success'] or result['reason'] != 'arrived'
            or len(planned) < 2 or np.linalg.norm(traj[0, :2]) > .02
            or np.linalg.norm(planned[0, :2]) > .02):
        raise RuntimeError('Unexpected status, initial pose or planned path: ' + name)
    error = float(np.linalg.norm(final[:2] - GOAL[:2]))
    if abs(error-result['internal_error']) > 1e-6:
        raise RuntimeError('Result pose disagrees with reported error: ' + name)
    truth_error = float(np.linalg.norm(truth[-1, 2:4] - GOAL[:2]))
    if abs(truth_error - error) > .02:
        raise RuntimeError('Scene trajectory and simulator truth disagree: ' + name)
    checks = {
        'path_collision_samples': occupancy_hits(dense_polyline(planned[:, :2], res/2), occ, origin, res),
        'feedback_collision_samples': occupancy_hits(traj[:, :2], occ, origin, res),
        'truth_collision_samples': occupancy_hits(truth[:, 2:4], occ, origin, res),
    }
    if any(checks.values()):
        raise RuntimeError('Grid/path/trajectory alignment or collision check failed: ' + repr(checks))
    deviations = path_distances(traj[:, :2], planned[:, :2])
    checks.update(
        elapsed_s=float(result['elapsed']), internal_error_m=error,
        settled_truth_error_m=truth_error,
        feedback_points=len(traj), truth_points=len(truth), odom_points=len(odom),
        feedback_frequency_hz=float((len(traj)-1) / (data['traj_t'][-1]-data['traj_t'][0])),
        path_length_m=float(np.linalg.norm(np.diff(planned[:, :2], axis=0), axis=1).sum()),
        actual_length_m=float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()),
        distance_to_planned_median_p95_max_m=np.quantile(deviations, [.5, .95, 1]).tolist(),
    )
    if name == 'arc':
        radii = np.linalg.norm(traj[:, :2] - [.5, 1.], axis=1)
        checks['actual_arc_radius_min_max_m'] = [float(radii.min()), float(radii.max())]
        if radii.min() < .6 or radii.max() > 1.5:
            raise RuntimeError('Arc trajectory is not inside the expected annular corridor')
    else:
        gap_traj = traj[(traj[:, 1] >= .8) & (traj[:, 1] <= 1.2)]
        checks['narrow_crossing_x_min_max_m'] = [float(gap_traj[:, 0].min()), float(gap_traj[:, 0].max())]
        if len(gap_traj) < 3 or gap_traj[:, 0].min() <= 1.2 or gap_traj[:, 0].max() >= 2.:
            raise RuntimeError('Narrow trajectory did not traverse the expected gap')
    checks['reference_error_m'] = SCENES[name]['reference_error_m']
    checks['reference_elapsed_s'] = SCENES[name]['reference_elapsed_s']
    # A noticeably changed result is a blocker, not a reason to relabel the data.
    if (abs(error-checks['reference_error_m']) > .02
            or abs(checks['elapsed_s']-checks['reference_elapsed_s']) > 8.):
        raise RuntimeError('Run differs substantially from user reference: ' + repr(checks))
    meta['checks'] = checks
    (path.parent / (name + '_summary.json')).write_text(
        json.dumps(meta, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('VALIDATED %s: error=%.3fcm elapsed=%.3fs p95-path-distance=%.3fcm collision-samples=0' % (
        name, error*100, checks['elapsed_s'], checks['distance_to_planned_median_p95_max_m'][1]*100), flush=True)
    return data, meta


def plot_scene(data, meta, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, FancyBboxPatch

    font = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if not font.is_file():
        raise RuntimeError('Chinese font not found: ' + str(font))
    font_manager.fontManager.addfont(str(font))
    font_name = font_manager.FontProperties(fname=str(font)).get_name()
    with plt.rc_context({'font.family': [font_name, 'DejaVu Sans'], 'font.size': 17, 'axes.unicode_minus': False}):
        fig = plt.figure(figsize=(12, 9), dpi=200, facecolor='white')
        ax = fig.add_axes([.07, .12, .63, .76])
        name, cfg, checks = meta['scene'], SCENES[meta['scene']], meta['checks']
        fig.text(.07, .947, '仿真｜' + cfg['title'], fontsize=27, color='#163D57', weight='bold')
        fig.text(.07, .898, '规划路径与闭环实际轨迹（原始采样，未平滑）', fontsize=17, color='#536675')
        occ, origin, res = data['occ'], data['origin'], float(data['res'])
        height, width = occ.shape
        ax.imshow(occ, origin='lower', interpolation='nearest',
                  cmap=ListedColormap(['white', '#D4D8DC']), vmin=0, vmax=1,
                  extent=[origin[0], origin[0]+width*res, origin[1], origin[1]+height*res], zorder=0)
        path, traj, final = data['path_0'], data['traj'], data['result_final']
        # Final pose is a raw /nav/result sample, not an artificial goal extension.
        shown = np.vstack([traj, final])
        ax.plot(path[:, 0], path[:, 1], color='#A27034', lw=5.0, ls=(0, (5, 3)), zorder=3)
        ax.plot(shown[:, 0], shown[:, 1], color='#165E8A', lw=2.7, zorder=4)
        indices, travelled = [0], 0.
        for i in range(1, len(traj)):
            travelled += float(np.linalg.norm(traj[i, :2]-traj[i-1, :2]))
            if travelled >= .40:
                indices.append(i)
                travelled = 0.
        arrow_points = traj[indices]
        ax.quiver(arrow_points[:, 0], arrow_points[:, 1], .16*np.cos(arrow_points[:, 2]),
                  .16*np.sin(arrow_points[:, 2]), angles='xy', scale_units='xy', scale=1,
                  color='#165E8A', width=.005, headwidth=4.0, headlength=5.0, zorder=5)
        ax.plot(0, 0, marker='s', color='black', ms=10, zorder=7)
        ax.plot(*GOAL[:2], marker='o', color='#48A06A', mec='#163D57', mew=1.3, ms=12, zorder=7)
        ax.plot(*final[:2], marker='o', color='#165E8A', ms=5, zorder=8)
        ax.set_xlim(cfg['crop'][:2]); ax.set_ylim(cfg['crop'][2:])
        ax.set_aspect('equal'); ax.grid(color='#E8EBEE', lw=.75)
        ax.set_xlabel('场景 x / m', fontsize=18); ax.set_ylabel('场景 y / m', fontsize=18)
        ax.tick_params(labelsize=16)
        for spine in ax.spines.values():
            spine.set_color('#9EAAB4')
        if name == 'narrow':
            ax.annotate('缝 0.8m', xy=(1.56, 1.02), xytext=(.78, 2.28), fontsize=17,
                        color='#536675', arrowprops={'arrowstyle': '->', 'lw': 1.4, 'color': '#76838D'},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 3})
        fig.patches.append(FancyBboxPatch((.728, .135), .240, .72,
                                         boxstyle='round,pad=.008,rounding_size=.008',
                                         transform=fig.transFigure, facecolor='#F4F7FA', edgecolor='#DFE6EC'))
        fig.text(.75, .795, '位置误差（本次）', fontsize=18, color='#536675')
        fig.text(.75, .729, '%.2f cm' % (checks['internal_error_m']*100), fontsize=32, color='#163D57', weight='bold')
        fig.text(.75, .690, '用户参考：%.1f cm' % (cfg['reference_error_m']*100), fontsize=13, color='#536675')
        fig.text(.75, .653, 'arrived · %.1f s' % checks['elapsed_s'], fontsize=19, color='#163D57')
        fig.text(.75, .594, '到规划折线距离 p95', fontsize=15, color='#536675')
        fig.text(.75, .551, '%.1f cm' % (checks['distance_to_planned_median_p95_max_m'][1]*100), fontsize=21, color='#163D57')
        handles = [Patch(facecolor='#D4D8DC', label='占据栅格（障碍）'),
                   Line2D([], [], color='#A27034', lw=4, ls='--', label='规划路径'),
                   Line2D([], [], color='#165E8A', lw=2.7, label='实际轨迹 + 朝向'),
                   Line2D([], [], color='black', marker='s', ls='', ms=9, label='起点'),
                   Line2D([], [], color='#48A06A', marker='o', ls='', ms=10, label='终点目标'),
                   Line2D([], [], color='#165E8A', marker='o', ls='', ms=5, label='实际到位点')]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.739, .494),
                   frameon=False, fontsize=15, labelspacing=1.0, handlelength=2.)
        fig.text(.75, .176, '仅仿真，非真机数据', fontsize=15, color='#536675')
        fig.text(.07, .048, '数据：/nav/feedback（默认 10 Hz）+ /nav/planned_path；所有曲线与栅格均在场景系。',
                 fontsize=13, color='#536675')
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=r'Glyph .* missing from .*font.*')
            fig.canvas.draw()
            fig.savefig(output, dpi=200, facecolor='white')
        plt.close(fig)
        print('PNG ' + str(output), flush=True)


def preview_check(output):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (960, 1130), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(SCENES):
        with Image.open(TOOLS / ('ppt_' + name + '.png')) as image:
            if image.size != (2400, 1800):
                raise RuntimeError('Unexpected final PNG size')
            thumbnail = image.convert('RGB').resize((720, 540), Image.Resampling.LANCZOS)
            canvas.paste(thumbnail, (120, i*565+25))
        draw.text((125, i*565+5), 'ppt_' + name + '.png (fits half a Full-HD slide)', fill='#163D57')
    canvas.save(output)
    print('PASS half-slide preview: ' + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run both original local simulators and collect fresh data')
    parser.add_argument('--data-dir', type=Path, help='Existing data directory for offline replotting')
    parser.add_argument('--preview-check', type=Path, help='Optional temporary half-slide preview montage')
    parser.add_argument('--collect-scene', choices=SCENES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.collect_scene:
        collect_scene(args.collect_scene, args.data_dir)
        return
    if args.run:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        if args.data_dir:
            directory = args.data_dir.resolve()
            if not directory.is_dir():
                parser.error('To resume, --data-dir must be an existing partial collection directory')
        else:
            directory = TOOLS / ('ppt_sim_' + stamp + '_' + str(os.getpid()))
            directory.mkdir(exist_ok=False)
        print('DATA_DIR ' + str(directory), flush=True)
        for name, cfg in SCENES.items():
            if (directory / (name + '.npz')).exists():
                print('KEEP existing raw data: ' + name, flush=True)
                continue
            env = isolated_environment(cfg['domain'])
            print('RUN %s (loopback-only DDS domain=%s)' % (name, cfg['domain']), flush=True)
            attempt = directory / (name + '_attempt_' + stamp + '_' + str(os.getpid()))
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                        '--collect-scene', name, '--data-dir', str(attempt)],
                                       cwd=NAV, env=env, start_new_session=True)
            try:
                rc = process.wait(timeout=165)
            finally:
                stop_owned(process)
            if rc:
                raise RuntimeError('Collection failed for %s; inspect %s' % (name, directory))
            shutil.copy2(attempt / (name + '.npz'), directory / (name + '.npz'))
    else:
        directory = args.data_dir
        if directory is None:
            parser.error('Use --run to collect or --data-dir DIRECTORY to replot without ROS')
    # Validate both scenes before updating either formal PNG.
    records = [load_validate(directory / (name + '.npz')) for name in SCENES]
    for (data, meta), name in zip(records, SCENES):
        plot_scene(data, meta, TOOLS / ('ppt_' + name + '.png'))
    if args.preview_check:
        preview_check(args.preview_check)
    print('PASS both scenes: arrived, zero collision samples, coordinate checks; original code unchanged', flush=True)


if __name__ == '__main__':
    main()
