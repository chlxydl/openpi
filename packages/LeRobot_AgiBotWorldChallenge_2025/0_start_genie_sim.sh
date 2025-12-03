#!/bin/bash

# ============================================================================
# GenieSim 启动脚本
# 适用于 Ubuntu 22.04
# ============================================================================
# 请将 ASSETS_DIR 修改为你的 GenieSimAssets 文件夹路径
ASSETS_DIR=/home/pengying/agitbot/GenieSimAssets

# 进入 genie_sim 根目录（请根据你的路径修改）
GENIE_SIM_DIR=/home/pengying/agitbot/genie_sim

# ----------------------------------------------------------------------------
# 切换到 genie_sim 目录
cd "$GENIE_SIM_DIR" || { echo "目录不存在: $GENIE_SIM_DIR"; exit 1; }

# 启动 GUI 容器
echo "启动 GenieSim Docker 容器..."
sudo SIM_ASSETS="$ASSETS_DIR"  ./scripts/start_gui.sh

# 进入容器交互终端
echo "进入容器..."
./scripts/into.sh
