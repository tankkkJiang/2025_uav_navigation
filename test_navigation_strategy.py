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
import numpy as np

# 假设你已经实现了 NavigationEnv 类，并在当前模块中或已导入
# from your_env_file import NavigationEnv

class ModelTester:
    def __init__(self, env_params, num_test_episodes=10):
        self.env_params = env_params
        self.num_test_episodes = num_test_episodes

    def _test_model_from_path(self, model_path):
        model = PPO.load(model_path)
        test_env = NavigationEnv(self.env_params)

        total_rewards = []
        test_episode_collisions = 0
        test_episode_successes = 0

        for _ in range(self.num_test_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

            if info.get("arrival", False):
                test_episode_successes += 1
            if info.get("collision", False):
                test_episode_collisions += 1

        avg_test_reward = np.mean(total_rewards)
        success_rate = test_episode_successes / self.num_test_episodes
        collision_rate = test_episode_collisions / self.num_test_episodes
        # Console logging
        print(f"[Test] {self.num_test_episodes} episodes | Avg Reward: {avg_test_reward:.2f} | "
                     f"Success Rate: {success_rate:.2f} | Collision Rate: {collision_rate:.2f}")

def main(run_dir):
    # === Load configuration file
    config_path = os.path.join(run_dir, "config", "navigation_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_params = config["env_params"]
    algo_name = config["algo_name"].lower()
    seed = config.get(f"{algo_name}_init_params", {}).get("seed", 0)

    # === Generate run name from existing folder (optional)
    run_name = os.path.basename(run_dir)

    # === Load model
    model_path = os.path.join(run_dir, "models", "final_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # === Test
    tester = ModelTester(env_params, num_test_episodes=100)
    tester._test_model_from_path(model_path)
if __name__ == "__main__":
    run_dir = "/home/congshan/uav/uav_roundup/navigation_strategy_2/logs/ppo_i_s0_0520_1628"
    main(run_dir)
