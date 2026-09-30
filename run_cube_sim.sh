#!/bin/bash
# ============================================================
#  仿真验证：带障碍的自定义场景
#
#  用法: bash run_cube_sim.sh [栅格npz] [任务]
#    默认 tools/cube_scene.npz（狗相对坐标：正前方 1m 一个 0.6m 正方体）
#
#  ★ 为什么仿真狗从 (0,0,0) 起：栅格是【狗相对坐标】写的（原点=狗、x=朝向），
#    而 nav_node_sim 用 --start 0 0 0 ⇒ 场景系 = 仿真狗的世界系原点。两者对齐。
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

GRID="${1:-tools/cube_scene.npz}"
TASK="${2:-2.0,0.0,0.0}"

pkill -f sim_dog 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
sleep 1

echo "======================================================================"
echo "  仿真：栅格=$GRID   任务=$TASK"
echo "======================================================================"

python3 -u sim_dog.py --seed 42 --x0 0.0 --y0 0.0 --yaw0 0 \
    > /tmp/simdog_cube.log 2>&1 &
sleep 4

python3 -u demo/nav_node_sim.py --start 0 0 0 --grid "$GRID" \
    > /tmp/navnode_cube.log 2>&1 &
sleep 5

python3 -u demo/group2_sim.py --tasks "$TASK" --yaw-tol 0.03
RC=$?
sleep 3

pkill -f nav_node_sim 2>/dev/null
pkill -f sim_dog 2>/dev/null
sleep 1

echo
echo "--- 节点日志（关键行）---"
grep -E "用栅格|规划出|跟随结束|位置阶段|按目标" /tmp/navnode_cube.log | head -12
exit $RC
