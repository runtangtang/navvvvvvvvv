#!/bin/bash
# 端到端任务演示：起仿真狗在组1 的场景图里，跑组2 给的任务序列
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

pkill -f sim_dog 2>/dev/null; sleep 1
# 狗停在【组2 给的起点】= 场景系 (0,0,0) 朝 +x
python3 -u sim_dog.py --seed 42 --x0 0 --y0 0 --yaw0 0 > /tmp/simdog_demo.log 2>&1 &
sleep 4
python3 -u demo/mission_demo.py
RC=$?
pkill -f sim_dog 2>/dev/null; sleep 1
exit $RC
