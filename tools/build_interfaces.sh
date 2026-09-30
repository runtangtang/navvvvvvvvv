#!/bin/bash
# 编译并验证 nav_interfaces 消息包（在 WSL 里跑）
# 用法: wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/迷雾通/nav/tools/build_interfaces.sh
set -e
cd /mnt/d/迷雾通/nav || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1
conda activate ros2
source "$HOME/ros2_ws/install/setup.bash" >/dev/null 2>&1

echo "=== 编译 nav_interfaces ==="
colcon build --packages-select nav_interfaces \
    --cmake-args -DCMAKE_CXX_FLAGS="-include cstdint" 2>&1 | tail -12

echo
echo "=== 验证 Python 能 import ==="
source install/setup.bash >/dev/null 2>&1
python3 -c "
from nav_interfaces.action import NavigateTo
print('Goal    :', list(NavigateTo.Goal().get_fields_and_field_types().keys()))
print()
print('Result  :', list(NavigateTo.Result().get_fields_and_field_types().keys()))
print()
print('Feedback:', list(NavigateTo.Feedback().get_fields_and_field_types().keys()))
"
