#!/bin/bash
# P2 验收（闭环 + 障碍）: 起仿真狗在场地里，让它绕开缺口的墙走到对面
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

run_one () {
    local world="$1"; local sx="$2"; local sy="$3"; local gx="$4"; local gy="$5"
    echo
    echo "################################################################"
    echo "#  world=$world   起点($sx,$sy) → 终点($gx,$gy)"
    echo "################################################################"
    pkill -f sim_dog 2>/dev/null; sleep 1
    python3 -u sim_dog.py --seed 42 --x0 "$sx" --y0 "$sy" --yaw0 0 \
        > /tmp/simdog_p2.log 2>&1 &
    sleep 4
    python3 -u tests/test_p2_sim.py --world "$world" --start "$sx" "$sy" \
        --goal "$gx" "$gy" --label "$world"
    local rc=$?
    pkill -f sim_dog 2>/dev/null; sleep 1
    return $rc
}

RC=0
run_one corridor 1.5 2.5 11.0 7.5 || RC=1
run_one utrap    1.5 8.5  6.3 4.2 || RC=1

pkill -f sim_dog 2>/dev/null
echo
[ $RC -eq 0 ] && echo "P2 全部通过" || echo "P2 有未通过"
exit $RC
