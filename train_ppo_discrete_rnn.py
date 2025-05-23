import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from model.ppo_discrete_rnn.normalization import Normalization, RewardScaling
from model.ppo_discrete_rnn.replaybuffer import ReplayBuffer
from model.ppo_discrete_rnn.ppo_discrete_rnn import PPO_discrete_RNN
from env.navigation_env import NavigationEnv
import wandb
import yaml
from datetime import datetime
import os
import logging


class Runner:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed

        # === 2. 读取配置文件
        with open("config/navigation_env_config.yaml", "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f)
        # 1. 基础信息
        algo_name = "ppo_discrete_rnn"

        # 2. 生成run name
        time_str = datetime.now().strftime('%m%d_%H%M%S')  # 月日_时分
        run_name = f"{algo_name}_s{seed}_{time_str}"

        # === 3. 创建日志、模型、配置保存目录
        self.log_dir = os.path.join("logs", run_name)
        self.model_dir = os.path.join(self.log_dir, "models")
        self.config_dir = os.path.join(self.log_dir, "config")

        for directory in [self.log_dir, self.model_dir, self.config_dir]:
            os.makedirs(directory, exist_ok=True)

        # === 4. 设置日志只写入文件
        log_file = os.path.join(self.log_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'  # 'w' 每次重新写，'a' 是追加写
        )
        
        # 示例日志输出
        logging.info("环境初始化完成")
        logging.info(f"日志目录: {self.log_dir}")
        logging.info(f"模型目录: {self.model_dir}")
        logging.info(f"配置目录: {self.config_dir}")

        with open(os.path.join(self.config_dir, "env_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(env_config, f)
        
        # 保存为 YAML 文件
        with open(os.path.join(self.config_dir, "ppo_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f)
        
        wandb.init(
            project="drone_navigation",
            name=run_name,
            config= {**vars(args), **env_config},                  # 直接同步wandb记录超参数
            dir=self.log_dir                      # wandb日志也放到log_dir下面
        )

        self.env = NavigationEnv(env_config)

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode

        self.replay_buffer = ReplayBuffer(args)
        self.agent = PPO_discrete_RNN(args)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            logging.info("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            logging.info("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = 0  # Record the number of evaluations
        episode_total_count = 0
        episode_success_count = 0
        episode_collision_count = 0
        episode_other_count = 0
        while self.total_steps < self.args.max_train_steps:

            infos, episode_steps = self.run_episode()  # Run an episode
            
            episode_total_count += 1
            if infos.get("arrival", False):
                episode_success_count += 1
            elif infos.get("collision", False):
                episode_collision_count += 1
            else:
                episode_other_count += 1
            
            success_rate = episode_success_count / episode_total_count if episode_total_count > 0 else 0.0
            collision_rate = episode_collision_count / episode_total_count if episode_total_count > 0 else 0.0

            wandb.log({
                "train/success_rate": success_rate,
                "train/collision_rate": collision_rate
            }, step=self.total_steps)

            for key, value in infos.items():
                if key.startswith("episode/"):
                    comp = key.split("/")[1]  # e.g. "target_progress"
                    wandb.log({f"train/{comp}": value}, step=self.total_steps)
                
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                metrics = self.agent.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

                wandb.log(metrics, step=self.total_steps)
                # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
                eval_metrics = self.evaluate_policy()
                wandb.log(eval_metrics, step=self.total_steps)
                # 保存模型权重
                if evaluate_num % args.save_freq == 0:
                    model_path = os.path.join(self.model_dir, f"model_update_{evaluate_num}.zip")
                    torch.save({
                        'ac': self.agent.ac.state_dict(),
                    }, model_path)
                    logging.info(f"[INFO] Saved model to {model_path}")
            
        model_path = os.path.join(self.model_dir, f"final_model.zip")
        torch.save({
            'ac': self.agent.ac.state_dict(),
        }, model_path)
        logging.info(f"[INFO] Saved model to {model_path}")
        
        evaluate_num += 1
        eval_metrics = self.evaluate_policy()
        wandb.log(eval_metrics, step=self.total_steps)
        self.env.close()

    def run_episode(self, ):
        episode_reward = 0
        s = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.agent.reset_rnn_hidden()
        for episode_step in range(self.args.episode_limit):
            if self.args.use_state_norm:
                s = self.state_norm(s)
            a, a_logprob = self.agent.choose_action(s, evaluate=False)
            v = self.agent.get_value(s)
            s_, r, done, infos = self.env.step(a)
            episode_reward += r

            if done and episode_step + 1 != self.args.episode_limit:
                dw = True
            else:
                dw = False
            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)
            # Store the transition
            self.replay_buffer.store_transition(episode_step, s, v, a, a_logprob, r, dw)
            s = s_
            if done:
                break

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            s = self.state_norm(s)
        v = self.agent.get_value(s)
        self.replay_buffer.store_last_value(episode_step + 1, v)

        return infos, episode_step + 1

    def evaluate_policy(self, times=20):
        total_reward = 0
        success_count = 0
        collision_count = 0
        timeout_count = 0
        step_counts = []
        for _ in range(times):
            episode_reward, done = 0, False
            s = self.env.reset()
            self.agent.reset_rnn_hidden()
            steps = 0
            while not done:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                a, a_logprob = self.agent.choose_action(s, evaluate=True)
                s_, r, done, infos = self.env.step(a)
                episode_reward += r
                s = s_
                steps += 1
            # 从 info 中读取评估指标
            if 'arrival' in infos and infos['arrival']:
                success_count += 1
            if 'collision' in infos and infos['collision']:
                collision_count += 1
            if 'timeout' in infos and infos['timeout']:
                timeout_count += 1
            step_counts.append(steps)
            total_reward += episode_reward
        
        success_rate = success_count / times
        collision_rate = collision_count / times
        average_reward = total_reward / times

        return {
            "test/average_reward": average_reward,
            "test/success_rate": success_rate,
            "test/collision_rate": collision_rate,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--algo_name", type=str, default="ppo_discrete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g., "cuda:0" or "cpu"')
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=15, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")

    args = parser.parse_args()

    env_index = 0
    for seed in [0, 10, 100]:
        runner = Runner(args, seed=seed)
        runner.run()
