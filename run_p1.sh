#!/bin/bash
# P1 验收：5 个用例，每个都重起一次仿真狗
#
#   ① 干净 × 1m          期望：真值误差 < 3cm
#   ② 干净 × 3m          期望：真值误差仍 < 3cm   ← 误差【不随距离增长】
#   ③ 尺度 0.94 × 3m     观察：尺度误差【随距离暴露】
#                        （短距离时滑行补偿会把它掩盖，3m 才看得出来）
#   ④ 延迟 240ms × 1m    观察：是否失稳
#   ⑤ 比例控制 × 1m      必须：卡死（真机停在 7.6cm，仿真应停在同量级）
#
# 仿真里没有场地限制，所以敢走 3m —— 这正是仿真的价值。
# ★ pkill 的模式 sim_dog 不出现在本脚本被调用时的父命令行里

cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

RC=0

start_sim () {
    pkill -f sim_dog 2>/dev/null
    sleep 1
    # shellcheck disable=SC2086
    python3 -u sim_dog.py --seed 42 $1 > /tmp/simdog_p1.log 2>&1 &
    sleep 4
}

run_case () {
    local title="$1"; local simargs="$2"; shift 2
    echo
    echo "################################################################"
    echo "#  $title"
    echo "################################################################"
    start_sim "$simargs"
    python3 -u tests/test_p1_sim.py "$@" || RC=1
    pkill -f sim_dog 2>/dev/null
    sleep 1
}

run_case "① 干净 x 1m"            ""                                        --goal-dist 1.0 --label "clean-1m"     --expect reach
run_case "② 干净 x 3m"            ""                                        --goal-dist 3.0 --label "clean-3m"     --expect reach
run_case "③ 尺度 -6% x 3m"        "--capo-scale 0.94"                       --goal-dist 3.0 --capo-scale 0.94 --label "scale-3m" --expect any
run_case "④ 延迟 240ms x 1m"      "--capo-latency 0.24 --capo-noise 0.007"  --goal-dist 1.0 --label "latency-1m"  --expect any
run_case "⑤ 比例控制（反面教材）"  ""                                        --goal-dist 1.0 --vx-mode prop --label "prop" --expect stall

pkill -f sim_dog 2>/dev/null
echo
echo "================================================================"
if [ $RC -eq 0 ]; then echo "  P1 全部用例通过"; else echo "  P1 有用例未通过（看上面）"; fi
echo "================================================================"
exit $RC
