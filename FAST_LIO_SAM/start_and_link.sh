#!/usr/bin/env bash
set -e

# 工程根目录（和 run_fastlio.sh 同一层）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${SCRIPT_DIR}/Log"

# 确保日志根目录存在
mkdir -p "${LOG_ROOT}"

# 先删除旧的 current 软连接（如果有）
if [ -L "${LOG_ROOT}/current" ] || [ -e "${LOG_ROOT}/current" ]; then
  rm -f "${LOG_ROOT}/current"
fi

# # 在后台启动原来的 run_fastlio.sh（里面已经设置好 ROS_LOG_DIR）
# "${SCRIPT_DIR}/run_fastlio.sh" "$@" &
# RUN_PID=$!

# echo "run_fastlio.sh started with PID ${RUN_PID}"
# echo "waiting for ROS to create a new log subdirectory under ${LOG_ROOT} ..."
# echo "running: ${SCRIPT_DIR}/run_fastlio.sh $@"

# 轮询，直到出现新的日志子目录
latest_dir=""
for i in $(seq 1 50); do   # 最多等 50 次，每次 0.2s，大约 10 秒
  latest_dir=$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 || true)
  if [ -n "${latest_dir}" ]; then
    break
  fi
  sleep 0.2
done

if [ -z "${latest_dir}" ]; then
  echo "No log subdirectory found under ${LOG_ROOT} (timeout)."
  echo "FAST_LIO_SAM may not have started correctly."
  exit 1
fi

# 建立 current 软链接指向这次运行的日志目录
ln -s "${latest_dir}" "${LOG_ROOT}/current"
echo "Linked ${LOG_ROOT}/current -> ${latest_dir}"

echo "You can now inspect logs under: ${LOG_ROOT}/current"
echo "Example: ls ${LOG_ROOT}/current"

# # 把后台的 FAST_LIO_SAM 进程前台等待（方便 Ctrl+C 一起退出）
# wait "${RUN_PID}"