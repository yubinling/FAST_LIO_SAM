#!/usr/bin/env bash
set -e

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTION_DIR="$PACKAGE_DIR/detection/livox_detection"

# 离线生成 txt 也使用检测网络依赖的 se-ssd 环境，避免手动切环境。
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate se-ssd

# TRT 后端需要系统 TensorRT 运行库；TF 后端设置这些变量也不影响结果。
export TENSORRT_DIR="${TENSORRT_DIR:-/usr/local/TensorRT-8.5.3.1}"
export PATH="$TENSORRT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH:-}"

cd "$DETECTION_DIR"

exec python offline_detection_to_txt.py "$@"
