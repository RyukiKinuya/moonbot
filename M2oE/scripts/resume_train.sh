#!/usr/bin/env bash
# 文件：resume_train.sh

# 首次启动时不带 resume
RESUME_FLAG=""

# 捕获 Ctrl+C，让脚本优雅退出
trap "echo; echo '脚本收到终止信号，退出'; exit 0" SIGINT SIGTERM

while true; do
  # 组装命令
  CMD="python -m M2oE.scripts.train${RESUME_FLAG:+ $RESUME_FLAG}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行：$CMD"
  
  # 执行训练
  $CMD
  RET=$?
  
  if [ $RET -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练正常完成 (exit $RET)，脚本退出。"
    break
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进程退出 (exit $RET)，3 秒后使用恢复模式重启……"
    # 从第二次开始都加上 resume
    RESUME_FLAG="--resume"
    sleep 3
  fi
done
