#!/usr/bin/env python3
"""Capture the unchanged nodes from run_cube_sim.sh and draw their trajectory.

Run in WSL after activating conda ros2 and sourcing ros2_ws/install/setup.bash:
    python3 tools/make_ppt_arc.py

This collector only subscribes. Goals come from demo/group2_sim.py via the
original shell workflow. All ROS participants use an isolated loopback domain.
"""
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if "--data-dir" in sys.argv:
        data_dir = Path(sys.argv[sys.argv.index("--data-dir") + 1]).resolve()
        with np.load(data_dir / "arc.npz") as data:
            draw(data["traj"], [data[f"path_{i}"] for i in range(int(data["n_path"]))],
                 data["occ"], data["origin"], float(data["res"]), root, data_dir)
        return
    if "--group2-client" in sys.argv:
        # Execute the existing client's main and send method unchanged. Only
        # wait for both navigation and capture discovery before its one send;
        # the client's fixed three-second startup can otherwise lose the goal.
        sys.argv = ["demo/group2_sim.py", "--tasks", "0.5,2.0,0.0",
                    "--yaw-tol", "0.03"]
        namespace = runpy.run_path("demo/group2_sim.py", run_name="ppt_group2_client")
        client_class = namespace["Group2"]
        original_send = client_class.send

        def ready_send(client, *goal):
            end = time.monotonic() + 30
            while client.pub.get_subscription_count() < 2:
                if time.monotonic() >= end:
                    raise RuntimeError("Navigation/client DDS discovery timeout")
                namespace["rclpy"].spin_once(client, timeout_sec=0.1)
            return original_send(client, *goal)

        client_class.send = ready_send
        namespace["main"]()
        return
    # Run the shell's exact node commands, without its global pkill or domain
    # reset. Re-exec before ROS imports so the collector and all children have
    # identical loopback discovery and library settings.
    from make_ppt_sim_figs import isolated_environment, DDS_XML
    if "--isolated" not in sys.argv:
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  "--isolated"], isolated_environment(148))
    if (os.environ.get("ROS_DOMAIN_ID") != "148"
            or os.environ.get("CYCLONEDDS_URI") != DDS_XML):
        raise RuntimeError("Missing loopback isolation")

    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    existing = [line for line in processes.splitlines()
                if "sim_dog.py" in line or "nav_node_sim.py" in line]
    if existing:
        raise RuntimeError("Existing simulation processes; refusing shell cleanup: "
                           + "\n".join(existing))

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("ppt_arc_capture")
    trajectory, paths, results = [], [], []

    def feedback(message):
        data = json.loads(message.data)
        trajectory.append((float(data["cur_x"]), float(data["cur_y"]),
                           float(data.get("cur_yaw", 0.0))))

    def planned_path(message):
        data = json.loads(message.data)
        points = np.asarray(data["waypoints"], dtype=float)
        if len(points):
            paths.append(points)

    def result(message):
        results.append(json.loads(message.data))

    node.create_subscription(String, "/nav/feedback", feedback, 100)
    node.create_subscription(String, "/nav/planned_path", planned_path, 100)
    node.create_subscription(String, "/nav/result", result, 100)
    node.create_subscription(String, "/nav/navigate_to", lambda message: None, 100)
    data_dir = root / "tools" / (time.strftime("ppt_arc_%Y%m%d_%H%M%S")
                                + "_" + str(os.getpid()))
    data_dir.mkdir()
    processes = []
    logs = []

    def start(script, args, label):
        log = (data_dir / (label + ".log")).open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([sys.executable, "-u", script, *args],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append(process)
        return process

    def spin_for(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        start("sim_dog.py", ["--seed", "42", "--x0", "0.0", "--y0", "0.0",
                             "--yaw0", "0"], "simdog")
        spin_for(4)
        start("demo/nav_node_sim.py", ["--start", "0", "0", "0", "--grid",
                                       "tools/arc_scene.npz"], "navnode")
        spin_for(5)
        process = start(str(Path(__file__).resolve()), ["--group2-client"], "group2")
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if process.poll() is None:
            raise RuntimeError("Simulation timed out; no plot was overwritten.")
        spin_for(0.5)
    finally:
        for child in reversed(processes):
            try:
                os.killpg(child.pid, signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()

    if not results or not results[-1].get("success"):
        raise RuntimeError("No successful result; no plot was overwritten. " + str(results))
    if results[-1].get("reason") != "arrived" or process.returncode != 0:
        raise RuntimeError("Unexpected result or shell exit: " + str(results[-1]))
    if not paths or len(trajectory) < 2:
        raise RuntimeError("Incomplete planned-path/feedback capture.")

    track = np.asarray(trajectory, dtype=float)
    grid_file = root / "tools" / "arc_scene.npz"
    with np.load(grid_file) as grid:
        occupancy = grid["occ"].copy()
        origin = grid["origin"].copy()
        if "scene_origin" in grid:
            origin = origin - grid["scene_origin"][:2]
        resolution = float(grid["res"])
    for points in [track, *paths]:
        cells = np.floor((points[:, :2] - origin[:2]) / resolution).astype(int)
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < occupancy.shape[1])
                  & (cells[:, 1] >= 0) & (cells[:, 1] < occupancy.shape[0]))
        if not inside.all() or occupancy[cells[:, 1], cells[:, 0]].any():
            raise RuntimeError("Path/grid coordinate mismatch or obstacle intersection.")

    np.savez_compressed(data_dir / "arc.npz", traj=track, occ=occupancy,
                        origin=origin, res=resolution, n_path=len(paths),
                        result_json=json.dumps(results[-1]),
                        **{f"path_{i}": points for i, points in enumerate(paths)})
    (data_dir / "result.json").write_text(
        json.dumps(results[-1], indent=2), encoding="utf-8")
    draw(track, paths, occupancy, origin, resolution, root, data_dir)
    print(json.dumps({"result": results[-1], "feedback_points": len(track),
                      "paths": len(paths), "output": str(root / "tools" / "ppt_arc.png")},
                     indent=2))


def draw(track, paths, occupancy, origin, resolution, root, data_dir):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ny, nx = occupancy.shape
    extent = (origin[0], origin[0] + nx * resolution,
              origin[1], origin[1] + ny * resolution)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(occupancy, origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=2, alpha=0.45)
    for i, points in enumerate(paths):
        ax.plot(points[:, 0], points[:, 1], "r--", lw=1.6,
                label="planned path" if i == 0 else None)
        ax.plot(points[:, 0], points[:, 1], "r.", ms=7,
                label="waypoints" if i == 0 else None)
    ax.plot(track[:, 0], track[:, 1], "C0-", lw=2.2, label="travelled")
    for k in range(0, len(track), max(1, len(track) // 14)):
        x, y, yaw = track[k]
        ax.arrow(x, y, 0.07 * math.cos(yaw), 0.07 * math.sin(yaw),
                 head_width=0.07, head_length=0.07, fc="C0", ec="C0",
                 length_includes_head=True, zorder=4)
    ax.plot(track[0, 0], track[0, 1], "ks", ms=11, label="start")
    ax.plot(track[-1, 0], track[-1, 1], "o", color="lime", mec="k", ms=13,
            label="end")
    ax.plot(0.5, 2.0, "*", color="gold", mec="k", ms=18, label="goal")
    ax.set_aspect("equal")
    ax.set_title("Arc corridor", fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    output = root / "tools" / "ppt_arc.png"
    temporary = data_dir / "ppt_arc.png"
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
