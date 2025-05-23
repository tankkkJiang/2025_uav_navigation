import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from env.navigation_env import NavigationEnv
import logging
import os

class WandbCallback(BaseCallback):
    def __init__(self, env_params, save_path):
        """
        初始化 WandbCallback，用于在每一回合记录奖励。

        参数:
        - gradient_save_freq: 保存梯度或其他统计信息的频率
        - verbose: 输出日志的详细程度
        """
        super(WandbCallback, self).__init__()
        self.update_count = 0
        self.env_params = env_params
        self.save_freq = env_params['rollout']['save_freq']
        self.save_path = save_path
        self.episode_successes = 0
        self.test_episode_successes = 0
        self.episode_count = 0
        self.episode_collisions = 0
        self.current_episode_rewards = {}
        self.num_test_episodes = env_params['rollout']['num_test_episodes']

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        metrics = {}
        
        # === PPO Loss ===
        logs = self.model.logger.name_to_value
        for key in ["loss", "std", "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction", "explained_variance"]:
            full_key = f"train/{key}"
            if full_key in logs:
                metrics[f"loss/{key}"] = logs[full_key]

        for info in infos:
            # === 累积当前 step 的各组件奖励 ===
            for key, value in info.items():
                if key.startswith("episode/"):
                    comp = key.split("/")[1]  # e.g. "target_progress"
                    self.current_episode_rewards[comp] = self.current_episode_rewards.get(comp, 0.0) + value

            # === 回合结束，统计并清空 ===
            if info.get("done", False):
                self.episode_count += 1
                if info.get("arrival", False):
                    self.episode_successes += 1
                if info.get("collision", False):
                    self.episode_collisions += 1

                # 每个组件总和
                for comp, total_reward in self.current_episode_rewards.items():
                    metrics[f"train/{comp}"] = total_reward

                metrics["train/total_reward"] = info.get("episode/total_reward", 0.0)
                metrics["train/length"] = info.get("step_count", 0)
                metrics["train/success_rate"] = self.episode_successes / self.episode_count
                metrics["train/collision_rate"] = self.episode_collisions / self.episode_count

                self.current_episode_rewards = {}  # 清空为下一个 episode 准备

        if metrics:
            wandb.log(metrics, step=self.num_timesteps)

        return True


    def _on_rollout_end(self):
        # 每经过一定步数进行一次测试
        self.update_count += 1
        # === 开始测试 ===
        logging.info(f"[Callback] 第 {self.update_count} 次更新后模型测试...")
        self._test_model()
        if self.update_count % self.save_freq == 0:
            model_file = os.path.join(self.save_path, f"model_update_{self.update_count}.zip")
            self.model.save(model_file)
            logging.info(f"[Callback] 第 {self.update_count} 次更新后保存模型至 {model_file}")

    
    def _test_model(self):
        """
        进行模型测试并将结果记录到 WandB。
        """
        # 通过模型在测试环境中进行评估
        model = self.model  # 获取当前训练的模型
        total_rewards = []
        test_episode_collisions = 0  # 用于测试回合撞击的计数器
        test_episode_successes = 0  # 用于测试回合成功的计数器
        test_env = NavigationEnv(self.env_params)

        for _ in range(self.num_test_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)  # 获取动作
                obs, reward, done, info = test_env.step(action)  # 执行动作
                episode_reward += reward

            total_rewards.append(episode_reward)
            if info.get('arrival', False):  # 如果成功到达目标
                test_episode_successes += 1  # 增加成功次数
            if info.get('collision', False):
                test_episode_collisions += 1  # 增加撞击次数

        # 计算平均奖励、成功率、碰撞率并记录到 WandB
        avg_test_reward = np.mean(total_rewards)
        success_rate = test_episode_successes / self.num_test_episodes
        collision_rate = test_episode_collisions / self.num_test_episodes

        # 记录测试指标到 WandB
        wandb.log({
            "test/average_reward": avg_test_reward,
            "test/success_rate": success_rate,
            "test/collision_rate": collision_rate,
        })

        # 输出日志
        logging.info(f"[Callback] Tested {self.num_test_episodes} episodes. Average Test Reward: {avg_test_reward}")
        logging.info(f"[Callback] Success Rate: {success_rate}, Collision Rate: {collision_rate}")