#!/bin/bash
# ============================================================
#  多段任务 + cancel 的验证（Action 接口，仿真）
#  ★ 任务写死在脚本里 —— 从命令行传 ";" 会被 wsl.exe 吞掉
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1
conda activate ros2
source "$HOME/ros2_ws/install/setup.bash" >/dev/null 2>&1
source "$HOME/nav_iface_ws/install/setup.bash" >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

GRID=tools/lab_scene.npz

start_nav () {
    pkill -f sim_dog 2>/dev/null; pkill -f nav_node_sim 2>/dev/null; sleep 1
    python3 -u sim_dog.py --seed 42 --x0 0.0 --y0 0.0 --yaw0 0 \
        > /tmp/sd_m.log 2>&1 &
    sleep 4
    python3 -u demo/nav_node_sim.py --start 0 0 0 --grid "$GRID" --action \
        > /tmp/nn_m.log 2>&1 &
    sleep 5
}
stop_all () {
    pkill -f nav_node_sim 2>/dev/null; pkill -f sim_dog 2>/dev/null; sleep 1
}

echo "################################################################"
echo "#  ① 多段任务：去 (0.5,2.0) 再回来 (0.5,0.5)  —— 中间【不重启节点】"
echo "################################################################"
start_nav
python3 -u demo/group2_action.py --tasks "0.5,2.0,0.0;0.5,0.5,0.0" --gap 3 --yaw-tol 0.03
RC1=$?
stop_all

echo
echo "################################################################"
echo "#  ② cancel 测试：发目标后 6 秒主动取消"
echo "################################################################"
start_nav
python3 -u demo/group2_action.py --tasks "0.5,2.0,0.0" --cancel-after 6
RC2=$?
stop_all

echo
echo "==================== 汇总 ===================="
echo "  ① 多段:   $( [ $RC1 -eq 0 ] && echo 通过 || echo 未通过 )"
echo "  ② cancel: $( [ $RC2 -eq 0 ] && echo 见上面输出 || echo 见上面输出 )"
