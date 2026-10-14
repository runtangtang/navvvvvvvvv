#!/bin/bash
cd /mnt/d/迷雾通/nav || exit 1
source $HOME/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate ros2
RC=0
for t in tests/test_frames.py tests/test_p2_plan.py tests/test_p3_faults.py \
         tests/test_avoid.py tests/test_obstacle_source.py \
         tests/test_final_approach.py; do
    echo
    echo "==================== $t ===================="
    python3 "$t" 2>&1 | tail -6
    [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
done
echo
echo "==================== P1 (闭环) ===================="
bash run_p1.sh > /tmp/p1.log 2>&1; P1=$?
grep -E "最终真值误差|比例控制卡死|P1 " /tmp/p1.log
[ $P1 -ne 0 ] && RC=1
echo
echo "==================== P2 (闭环避障) ===================="
bash run_p2_sim.sh > /tmp/p2.log 2>&1; P2=$?
grep -E "① 到达|② 避障|P2 " /tmp/p2.log
[ $P2 -ne 0 ] && RC=1
echo
[ $RC -eq 0 ] && echo "###### 全量回归全部通过 ######" || echo "###### 有未通过 ######"
exit $RC
