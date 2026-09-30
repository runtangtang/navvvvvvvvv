#!/bin/bash
# 起仿真狗 -> 跑自检 -> 停仿真狗
# 注意: pkill 的模式 "sim_dog" 不出现在本脚本被调用时的父命令行里（父命令是 bash run_test.sh）
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7          # 和真机/别的实验隔离

python3 -u sim_dog.py --seed 42 > /tmp/simdog.log 2>&1 &
SIM=$!
sleep 4
python3 -u test_sim_dog.py
kill $SIM 2>/dev/null
sleep 1
echo
echo "--- sim_dog 日志尾部 ---"
tail -3 /tmp/simdog.log
