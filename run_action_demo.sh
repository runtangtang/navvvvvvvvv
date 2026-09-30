#!/bin/bash
# ============================================================
#  正式 Action 接口的端到端验证（仿真，不用狗）
#
#  链路: 仿真狗 → 导航节点（Action 服务端）→ 扮演组2 的 Action 客户端
#  用法: bash run_action_demo.sh [栅格npz] [任务]
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1
conda activate ros2
source "$HOME/ros2_ws/install/setup.bash" >/dev/null 2>&1
# ★ 先 source 消息包工作区（在 ~/nav_iface_ws 里编的，不能在 nav/ 里编）
source "$HOME/nav_iface_ws/install/setup.bash" >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

GRID="${1:-tools/lab_scene.npz}"
TASKS="${2:-2.0,0.0,0.0}"

pkill -f sim_dog 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
sleep 1

echo "======================================================================"
echo "  Action 接口端到端验证   栅格=$GRID   任务=$TASKS"
echo "======================================================================"

python3 -u sim_dog.py --seed 42 --x0 0.0 --y0 0.0 --yaw0 0 \
    > /tmp/simdog_act.log 2>&1 &
sleep 4

echo "[1/2] 导航节点（★ Action 服务端）..."
python3 -u demo/nav_node_sim.py --start 0 0 0 --grid "$GRID" --action \
    > /tmp/navnode_act.log 2>&1 &
sleep 5

echo "[2/2] 扮演组2（★ Action 客户端）..."
python3 -u demo/group2_action.py --tasks "$TASKS" --yaw-tol 0.03
RC=$?
sleep 3

pkill -f nav_node_sim 2>/dev/null
pkill -f sim_dog 2>/dev/null
sleep 1

echo
echo "--- 节点日志（关键行）---"
grep -E "ActionServer|用栅格|收到目标|规划出|跟随结束|位置阶段|任务结束|按目标" \
    /tmp/navnode_act.log | head -12
exit $RC
