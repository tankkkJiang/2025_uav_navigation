#!/bin/bash

# 指定配置文件
CONFIG_FILE="navigation_config.yaml"

# 遍历需要的种子
for seed in 3 4
do
    echo "正在设置 seed = $seed"

    # 用 yq 修改 config.yaml 中的 ppo_init_params.seed

    yq eval ".sac_init_params.seed = $seed" -i $CONFIG_FILE
    yq eval '.sac_init_params.device = "cuda:3"' -i $CONFIG_FILE
    yq eval '.algo_name = "SAC"' -i $CONFIG_FILE

    # 运行训练脚本
    echo "开始训练，seed = $seed"
    python train_navigation_strategy.py &
    sleep 60
done
