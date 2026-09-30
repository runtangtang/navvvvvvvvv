#!/bin/bash
# ============================================================
#  ★ 验证：Action 模式下 live_view 也能收到数据（2026-09-28）
#
#  背景：live_view 订阅 /nav/feedback + /nav/planned_path + /nav/result，
#        但那三个话题原来是 RosTopicBridge(JSON版) 发的；
#        ActionServer【一个都不发】⇒ 一用 --action 可视化就全空。
#        修法：ActionServer 也发这三个话题（字段照抄 ros_topics.py）。
#
#  判据：live_view 日志里 "轨迹 N 点、路径 M 条、result K 条" 的 N/M/K 都要 > 0
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1
conda activate ros2
source "$HOME/ros2_ws/install/setup.bash" >/dev/null 2>&1
source "$HOME/nav_iface_ws/install/setup.bash" >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7
export MPLBACKEND=Agg          # 别弹窗

pkill -f sim_dog 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
pkill -f live_view.py 2>/dev/null
sleep 1
rm -f /tmp/viz_test.npz /tmp/liveview_test.log

echo "======================================================================"
echo "  Action 模式 + live_view   端到端验证"
echo "======================================================================"

python3 -u sim_dog.py --seed 42 --x0 0.0 --y0 0.0 --yaw0 0 \
    > /tmp/simdog_viz.log 2>&1 &
sleep 4

echo "[1/3] 导航节点（--action，★ 不传 --start）..."
python3 -u demo/nav_node_sim.py --grid tools/empty_grid.npz --action \
    > /tmp/navnode_viz.log 2>&1 &
sleep 5

echo "[2/3] live_view（★ 订阅 JSON 话题）..."
python3 -u tools/live_view.py --grid tools/empty_grid.npz \
    --npz /tmp/viz_test.npz --seconds 22 --save-every 5 \
    > /tmp/liveview_test.log 2>&1 &
sleep 4

echo "[3/3] 发目标（扮演组2）..."
python3 -u demo/group2_action.py --tasks "2.0,0,0" --yaw-tol 0.03
sleep 6

pkill -f nav_node_sim 2>/dev/null
pkill -f sim_dog 2>/dev/null
sleep 2
pkill -f live_view.py 2>/dev/null
sleep 1

echo
echo "===== live_view 收到了什么 ====="
grep -E "数据已落盘|result:" /tmp/liveview_test.log | tail -6

echo
echo "===== 判据检查 ====="
python3 -c "
import numpy as np
try:
    d = np.load('/tmp/viz_test.npz')
    n = len(d['traj']); npath = int(d['n_path']); nres = int(d['n_result'])
    print('  轨迹点数 : %d   %s' % (n, '✅' if n > 0 else '★ 还是 0！'))
    print('  路径条数 : %d   %s' % (npath, '✅' if npath > 0 else '★ 还是 0！'))
    print('  result数 : %d   %s' % (nres, '✅' if nres > 0 else '★ 还是 0！'))
    if n > 0:
        t = d['traj']
        print('  起点 (%.2f, %.2f)  终点 (%.2f, %.2f)  走了 %.2f m'
              % (t[0,0], t[0,1], t[-1,0], t[-1,1],
                 float(np.sum(np.hypot(np.diff(t[:,0]), np.diff(t[:,1]))))))
    ok = (n > 0 and npath > 0 and nres > 0)
    print()
    print('  ⇒ %s' % ('✅ 通过：Action 模式下可视化有数据了' if ok
                       else '★ 未通过'))
except Exception as e:
    print('  读 npz 失败:', e)
"
