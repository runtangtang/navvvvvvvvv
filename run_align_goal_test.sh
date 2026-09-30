#!/bin/bash
# ============================================================
#  ★ 验证【对齐由 goal 的 s_* 驱动】—— 2026-09-27 新增
#
#  背景：以前 T 只能靠命令行 --start 建，goal 里的 s_* 读都不读；
#        现在节点【不带 --start】，等第一个 goal 的 s_* 来对齐。
#
#  三个场景：
#   ① 单段      s=0.5 → g=2.0        ⇒ ★ 应该走 1.5m（不是 2.0m）
#   ② 多段链式  s=0.5 → 2.0 ; 3.0    ⇒ 第二段 s 自动 = 2.0 ⇒ 【一致】，不重对齐
#   ③ 多段不链  s=0.5 → 2.0 ; 3.0    ⇒ 第二段 s 还是 0.5 ⇒ 【不一致】⇒ 重对齐 + 报警
#                                      （--no-chain，专门验证报警那条路）
#
#  用法: bash run_align_goal_test.sh
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

pkill -f sim_dog 2>/dev/null
pkill -f nav_node_sim 2>/dev/null
sleep 1

run_case() {
    local title="$1"; shift
    local log="$1"; shift
    echo
    echo "############################################################"
    echo "#  $title"
    echo "############################################################"
    pkill -f sim_dog 2>/dev/null; pkill -f nav_node_sim 2>/dev/null; sleep 1

    python3 -u sim_dog.py --seed 42 --x0 0.0 --y0 0.0 --yaw0 0 \
        > /tmp/simdog_align.log 2>&1 &
    sleep 4
    # ★ 故意【不】传 --start ⇒ 走"等 goal 的 s_* 对齐"这条路
    python3 -u demo/nav_node_sim.py --grid tools/lab_scene.npz --action \
        > "$log" 2>&1 &
    sleep 5
    python3 -u demo/group2_action.py "$@"
    sleep 3
    pkill -f nav_node_sim 2>/dev/null; pkill -f sim_dog 2>/dev/null; sleep 1
}

# ---------------- ① 单段：s=0.5 → g=2.0  ⇒ 走 1.5m ----------------
run_case "① 单段   s=0.5  →  g=2.0    ★ 应该走 1.5 m" /tmp/nav_align1.log \
    --start 0.5 0 0 --tasks "2.0,0,0" --yaw-tol 0.03
echo "--- 对齐 ---"
grep -E "未做开机对齐|首次对齐|组2 报的起点|CAPO 当时读数|⇒ T =" /tmp/nav_align1.log
echo "--- 路径长度（判据：≈1.50 m，不是 2.00 m）---"
grep -E "规划出|跟随结束" /tmp/nav_align1.log

# ---------------- ② 多段链式：第二段 s 自动 = 第一段 g ⇒ 一致 ----------------
run_case "② 多段链式  s=0.5 → 2.0 ; 3.0   ★ 第二段应该【一致】，不重对齐" \
    /tmp/nav_align2.log \
    --start 0.5 0 0 --tasks "2.0,0,0;3.0,0,0" --yaw-tol 0.03 --gap 1
echo "--- 对齐（应该看到【起点校验 ✅ 一致】，不该有【重新对齐】）---"
grep -E "首次对齐|起点校验|重新对齐|差太多" /tmp/nav_align2.log

# ---------------- ③ 多段不链：第二段 s 还是 0.5 ⇒ 不一致 ⇒ 重对齐+报警 -----------
run_case "③ 多段不链  ⇒ 第二段应该【重对齐 + 大声报警】" /tmp/nav_align3.log \
    --start 0.5 0 0 --tasks "2.0,0,0;3.0,0,0" --yaw-tol 0.03 --gap 1 --no-chain
echo "--- 对齐（应该看到【差太多】+【重新对齐】）---"
grep -E "首次对齐|起点校验|重新对齐|差太多|抹掉" /tmp/nav_align3.log

echo
echo "============================================================"
echo "  汇总"
echo "============================================================"
echo "① 单段    路径长度: $(grep -oE '长 [0-9.]+ m' /tmp/nav_align1.log | head -1)"
echo "② 链式    第二段:   $(grep -cE '起点校验 ✅ 一致' /tmp/nav_align2.log) 次一致 / $(grep -cE '重新对齐' /tmp/nav_align2.log) 次重对齐"
echo "③ 不链    第二段:   $(grep -cE '起点校验 ✅ 一致' /tmp/nav_align3.log) 次一致 / $(grep -cE '重新对齐' /tmp/nav_align3.log) 次重对齐"
