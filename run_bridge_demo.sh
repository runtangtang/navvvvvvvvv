#!/bin/bash
# 完整走一遍【对外接口】：仿真狗 + 我们的导航节点 + 扮演组2的客户端 + 笔记本可视化
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

pkill -f sim_dog 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
pkill -f live_view 2>/dev/null
pkill -f group2_sim 2>/dev/null
sleep 1

echo "[1/4] 仿真狗 ..."
python3 -u sim_dog.py --seed 42 --x0 -1.0 --y0 0.0 --yaw0 0 > /tmp/simdog_b.log 2>&1 &
sleep 4

echo "[2/4] 我们的导航节点 + ROS 桥 ..."
python3 -u demo/nav_node_sim.py --start -1.0 0.0 0 > /tmp/navnode.log 2>&1 &
sleep 4

echo "[3/4] 笔记本可视化（存快照模式）..."
python3 -u tools/live_view.py --seconds 400 --save-every 30 > /tmp/viewer.log 2>&1 &
sleep 2

echo "[4/4] 扮演组2：发目标 ..."
python3 -u demo/group2_sim.py --tasks "3.0,0.0,0.785;1.4,-2.8,1.57;-1.0,0.0,0.0" --gap 2
RC=$?
sleep 6

pkill -f live_view 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
pkill -f sim_dog 2>/dev/null
sleep 1
echo
echo "--- 我们的节点日志（尾）---"
grep -vE "^\s+跟随|^\s+对准" /tmp/navnode.log | tail -12
echo
echo "--- 可视化日志 ---"
tail -4 /tmp/viewer.log
exit $RC
