#!/usr/bin/env bash
set -e

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTION_DIR="$PACKAGE_DIR/detection/livox_detection"

# roslaunch 启动的是这个可执行脚本；即使外部终端在 base 环境，
# 这里也会切到检测网络实际依赖的 se-ssd 环境。
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate se-ssd

# TensorRT 运行库安装在 conda 环境外；TRT 后端启动前需要把库路径补进来。
export TENSORRT_DIR="${TENSORRT_DIR:-/usr/local/TensorRT-8.5.3.1}"
export PATH="$TENSORRT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH:-}"

cd "$DETECTION_DIR"

# 用 exec 让 ROS 直接管理 Python 检测进程，Ctrl+C/roslaunch 退出时信号能正确传递。
exec python realtime_detection_limot.py "$@"
