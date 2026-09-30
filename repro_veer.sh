#!/bin/bash
# ============================================================
#  复现「真狗走路跑偏 → 跟随器绕圈」的 bug
#
#  关键：给 sim_dog 加 --walk-veer 3.6（实测每米左偏 3.6°）
#        ★ 不加的话仿真狗走得【绝对笔直】——
#          横向偏移恒为 0 ⇒ "到近处路点的方位角"永远不会爆炸
#          ⇒ 这个 bug 在仿真里【永远复现不出来】
#          （这就是 P1/P2 回归全过、真机却翻车的原因）
#
#  用法: bash repro_veer.sh [跑偏deg/m, 默认3.6]
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
source $HOME/ros2_ws/install/setup.bash >/dev/null 2>&1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=7

VEER="${1:-3.6}"

echo "################################################################"
echo "#  跑偏复现：walk-veer = ${VEER} deg/m"
echo "################################################################"

pkill -f sim_dog 2>/dev/null; sleep 1
python3 -u sim_dog.py --seed 42 --x0 1.5 --y0 2.5 --yaw0 0 \
    --walk-veer "$VEER" > /tmp/simdog_veer.log 2>&1 &
sleep 4

python3 -u tests/test_p2_sim.py --world corridor --start 1.5 2.5 \
    --goal 11.0 7.5 --label "veer${VEER}"
RC=$?

pkill -f sim_dog 2>/dev/null
echo
if [ $RC -eq 0 ]; then
    echo ">>> 通过了 —— 没复现（可能跑偏太小，或 bug 已被修掉）"
else
    echo ">>> ★ 没通过 —— bug 复现了"
fi
exit $RC
