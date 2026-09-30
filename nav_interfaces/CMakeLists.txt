cmake_minimum_required(VERSION 3.8)
project(nav_interfaces)

# =============================================================================
#  nav_interfaces —— 导航组 → 组2 的任务接口
#
#  ★ 生成的 Python 模块：
#       nav_interfaces.action.NavigateTo
#       nav_interfaces.action.NavigateTo_Goal / _Result / _Feedback / _SendGoal 等
#
#  ★ 编译（笔记本 WSL / 狗上 各来一次）：
#       colcon build --packages-select nav_interfaces
#     在 ~/nav_ws/ 里编（★ 别动别人的工作区）
# =============================================================================

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/NavigateTo.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
