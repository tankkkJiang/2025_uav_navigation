import yaml
import logging
import threading
import wandb
import os
from datetime import datetime

from stable_baselines3 import PPO, SAC, TD3, DDPG
from utils.depth_feature_extractor import FeatureExtractor
from env.navigation_env import NavigationEnv
from utils.wandb_callback import WandbCallback
import gym



def main():
    # === 1. 设置无界面渲染
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # === 2. 读取配置文件
    with open("navigation_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_params = config["env_params"]
    feature_extractor_params = config["feature_extractor_params"]
    net_arch_parms = config["net_arch"]
    # 1. 基础信息
    algo_name = config["algo_name"].lower()
    seed = config.get(f"{algo_name}_init_params", {}).get("seed", 0)

    # 2. 奖励组件首字母缩写
    active_rewards = config["env_params"]["reward"]["active_components"]
    reward_abbr = "".join([r.split('_')[0][0] for r in active_rewards])  # 每个奖励取第一个单词首字母

    # 3. 生成run name
    time_str = datetime.now().strftime('%m%d_%H%M%S')  # 月日_时分
    run_name = f"{algo_name}_{reward_abbr}_s{seed}_{time_str}"

    # === 3. 创建日志、模型、配置保存目录
    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join(log_dir, "models")
    config_dir = os.path.join(log_dir, "config")

    for directory in [log_dir, model_dir, config_dir]:
        os.makedirs(directory, exist_ok=True)

    # === 4. 设置日志只写入文件
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'  # 'w' 每次重新写，'a' 是追加写
    )

    # 示例日志输出
    logging.info("环境初始化完成")
    logging.info(f"日志目录: {log_dir}")
    logging.info(f"模型目录: {model_dir}")
    logging.info(f"配置目录: {config_dir}")

    with open(os.path.join(config_dir, "navigation_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    env = NavigationEnv(env_params)
    


    wandb.init(
        project="drone_navigation",
        name=run_name,
        config=config,                  # 直接同步wandb记录超参数
        dir=log_dir                      # wandb日志也放到log_dir下面
    )
    # 提取 policy_kwargs 并合并/确认必要字段
    policy_kwargs = dict(
        # features_extractor_class=FeatureExtractor,
        # features_extractor_kwargs=dict(cfg=feature_extractor_params),
        net_arch=net_arch_parms
    )
    # 1. 读取算法名称
    algo_name = config["algo_name"].upper()

    # 2. 根据算法名称选择
    if algo_name == "PPO":
        algo_init_params = config["ppo_init_params"]
        model = PPO(
            env=env,
            policy_kwargs=policy_kwargs,
            **algo_init_params
        )

    elif algo_name == "SAC":
        algo_init_params = config["sac_init_params"]
        model = SAC(
            env=env,
            policy_kwargs=policy_kwargs,
            **algo_init_params
        )

    elif algo_name == "TD3":
        algo_init_params = config["td3_init_params"]
        model = TD3(
            env=env,
            policy_kwargs=policy_kwargs,
            **algo_init_params
        )

    elif algo_name == "DDPG":
        algo_init_params = config["ddpg_init_params"]
        model = DDPG(
            env=env,
            policy_kwargs=policy_kwargs,
            **algo_init_params
        )

    else:
        raise ValueError(f"不支持的算法: {algo_name}")

    logging.info(f"已初始化 {algo_name} 模型！")

    logging.info(f"开始训练 {algo_name} 模型...")
    model_learn_params = config["model_learn_params"]
    model.learn(
        callback=WandbCallback(env_params, model_dir),
        **model_learn_params
    )

    # ========== 9. 保存最终模型 ==========
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    logging.info(f"最终模型已保存到: {final_model_path}")

    # ========== 10. 关闭环境 ==========
    env.close()



if __name__ == '__main__':
    main()