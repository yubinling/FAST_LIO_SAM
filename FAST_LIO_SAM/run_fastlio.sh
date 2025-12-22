#!/usr/bin/env bash

# 获取当前脚本所在目录（工程根目录 FAST_LIO_SAM/FAST_LIO_SAM）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置 ROS 日志目录到当前工程下的 Log 目录
export ROS_LOG_DIR="${SCRIPT_DIR}/Log"

# 可选：创建 Log 目录（如果不存在）
mkdir -p "${ROS_LOG_DIR}"

echo "ROS_LOG_DIR set to: ${ROS_LOG_DIR}"

# 选择要启动的 launch 文件，这里默认用 velodyne16，可按需改
LAUNCH_PKG="fast_lio_sam"
LAUNCH_FILE="mapping_robosense.launch"

# 如果你想换别的 launch，改成对应的文件名即可，例如：
# LAUNCH_FILE="mapping_avia.launch"
# LAUNCH_FILE="mapping_horizon.launch"

# 启动 roslaunch
exec roslaunch "${LAUNCH_PKG}" "${LAUNCH_FILE}" "$@"
