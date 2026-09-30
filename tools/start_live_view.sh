#!/bin/bash
# ============================================================
#  启动笔记本侧实时可视化（★ 在 WSL 里跑）
#
#  为什么要有这个文件（2026-09-26 踩的两次坑）：
#    ① `pkill -f live_view` 写在【命令行里】会杀掉自己 ——
#       那条命令自己的命令行就含 "live_view"，pkill -f 匹配命令行 ⇒ 自杀。
#       试过 `[l]ive_view` 的写法，但命令里还有 `tools/live_view.py`，
#       正则照样匹配 ⇒ 还是自杀。
#       ⇒ 放进【脚本文件】里，脚本自己的命令行是 `bash .../start_live_view.sh`，
#         不含 "live_view.py" ⇒ 安全。
#    ② `wsl.exe -- bash -lc '... nohup python3 xxx & sleep 6'` ——
#       这条命令返回时，WSL 会把会话里剩下的进程一起收掉，nohup 挡不住。
#       ⇒ 用 `setsid` 脱离会话 + `< /dev/null` 断开 stdin。
#
#  用法（Windows git-bash）:
#      wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/迷雾通/nav/tools/start_live_view.sh
# ============================================================
cd /mnt/d/迷雾通/nav || exit 1
source "$HOME/capo_live/env.sh" >/dev/null 2>&1

GRID="${1:-tools/empty_grid.npz}"
NPZ="${2:-/mnt/d/迷雾通/nav/tools/last_run.npz}"

# 先清旧的（★ 在脚本文件里，安全）
pkill -f "live_view.py" 2>/dev/null
sleep 1

setsid nohup python3 -u tools/live_view.py \
    --grid "$GRID" --npz "$NPZ" \
    > /tmp/liveview.log 2>&1 < /dev/null &

sleep 7
echo "--- live_view 日志 ---"
tail -5 /tmp/liveview.log 2>&1
echo "--- 进程 ---"
pgrep -af "live_view.py" | head -2 || echo "(没起来！)"
