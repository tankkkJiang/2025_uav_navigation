"""
train_ppo_navrl.py
执行完整的 PPO 训练循环。
训练 NavRLEnv （连续 3D 速度控制）—— PPO‑Clip + （可选）GRU
"""

import os, yaml, gym, argparse, logging, torch, numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.ppo_continuous_rnn.normalization import Normalization, RewardScaling
from model.ppo_continuous_rnn.replaybuffer import ReplayBuffer  # 和离散共用
from model.ppo_continuous_rnn.ppo_continuous_rnn import PPO_continuous_RNN
from env.navrl_env import NavRLEnv


# --------------------------- Runner ----------------------------
class Runner:
    def __init__(self, args, seed:int):
        self.args, self.seed = args, seed

        # === 1) 读取环境配置 ===
        config_path = "config/navrl_env_config.yaml"
        logging.info(f"Reading environment config from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f)
        logging.info(f"Environment config loaded:\n{yaml.dump(env_cfg)}")

        # === 2) 生成 run 名称 & 路径 ===
        algo   = "ppo_continuous_rnn"
        t_str  = datetime.now().strftime("%m%d_%H%M%S")
        self.run_name = f"{algo}_s{seed}_{t_str}"
        logging.info(f"Run name: {self.run_name}")

        # 所有输出都放到 result/{run_name} 下
        base_dir = "result"
        self.log_dir = os.path.join(base_dir, self.run_name)
        self.model_dir  = os.path.join(self.log_dir, "models")
        self.config_dir = os.path.join(self.log_dir, "config")
        logging.info(f"Log directory: {self.log_dir}")
        logging.info(f"Model directory: {self.model_dir}")
        logging.info(f"Config directory: {self.config_dir}")

        for d in [self.log_dir, self.model_dir, self.config_dir]:
            os.makedirs(d, exist_ok=True)

        # === 3) 纯文件日志 ===
        log_file = os.path.join(self.log_dir, "train.log")
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info(f"Logging to file: {log_file}")

        # === 4) 保存配置到 log 目录 ===
        env_config_out = os.path.join(self.config_dir, "env_config.yaml")
        ppo_config_out = os.path.join(self.config_dir, "ppo_config.yaml")
        with open(env_config_out, "w", encoding="utf-8") as f:
            yaml.dump(env_cfg, f)
        logging.info(f"Saved environment config to: {env_config_out}")
        with open(ppo_config_out, "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f)
        logging.info(f"Saved PPO config to: {ppo_config_out}")
        logging.info(f"PPO hyperparameters:\n{yaml.dump(vars(args))}")

        # === 5) wandb ===
        logging.info("Initializing Weights & Biases")
        wandb.init(
            project="drone_navigation",
            name=self.run_name,
            dir=self.log_dir,
            config={**vars(args), **env_cfg}
        )

        # === 6) 创建环境 & 随机种子 ===
        logging.info("Creating NavRLEnv with provided configuration")
        self.env = NavRLEnv(env_cfg)
        logging.info(f"NavRLEnv observation_space: {self.env.observation_space}")
        logging.info(f"NavRLEnv action_space: {self.env.action_space}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        logging.info("Environment reset with seed; initial observation obtained")


        # === 7) 根据环境补充超参 ===
        args.state_dim   = self.env.observation_space.shape[0]
        args.action_dim  = self.env.action_space.shape[0]        # 连续 3 维
        args.episode_limit = self.env.max_steps
        logging.info(f"Derived state_dim: {args.state_dim}, action_dim: {args.action_dim}, episode_limit: {args.episode_limit}")

        # === 8) Buffer / Agent ===
        self.buffer = ReplayBuffer(args)
        self.agent  = PPO_continuous_RNN(args)
        logging.info("ReplayBuffer and agent initialized")

        if args.use_state_norm:
            logging.info("Using state normalization")
            self.state_norm = Normalization(shape=args.state_dim)
        if args.use_reward_scaling:
            logging.info("Using reward scaling")
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        self.total_steps = 0
        self.eval_id     = 0

    # ---------------------- 主训练循环 -------------------------
    def run(self):
        logging.info("Starting main training loop")
        episode_total = success_n = collision_n = other_n = 0
        while self.total_steps < self.args.max_train_steps:

            info, ep_steps = self.run_episode()
            episode_total += 1
            # —— 统计到达 / 碰撞 ——
            if info["arrival"]:   success_n   += 1
            elif info["collision"]: collision_n += 1
            else:                 other_n     += 1

            success_rate = success_n / episode_total
            collision_rate = collision_n / episode_total
            logging.info(f"Episode {episode_total}: success_rate={success_rate:.3f}, collision_rate={collision_rate:.3f}")
            wandb.log({
                "train/success_rate": success_rate,
                "train/collision_rate": collision_rate
            }, step=self.total_steps)

            # buffer 满一 batch 就更新
            if self.buffer.episode_num == self.args.batch_size:
                metrics = self.agent.train(self.buffer, self.total_steps)
                self.buffer.reset_buffer()
                wandb.log(metrics, step=self.total_steps)

                # —— 定期评估 & 保存 ——
                if self.total_steps // self.args.evaluate_freq > self.eval_id:
                    self.eval_id += 1
                    eval_metrics = self.evaluate_policy()
                    wandb.log(eval_metrics, step=self.total_steps)

                    if self.eval_id % self.args.save_freq == 0:
                        path = os.path.join(self.model_dir, f"model_{self.eval_id}.pt")
                        torch.save({'ac': self.agent.ac.state_dict()}, path)
                        logging.info(f"✔ Saved model to {path}")

        # 训练完保存最终模型
        final_path = os.path.join(self.model_dir, "final_model.pt")
        torch.save({'ac': self.agent.ac.state_dict()}, final_path)
        logging.info(f"✔ Saved final model to {final_path}")

    # ---------------------- 单回合 -----------------------------
    def run_episode(self):
        s, _ = self.env.reset()
        logging.info("Episode reset; initial observation obtained")
        if self.args.use_reward_scaling: self.reward_scaling.reset()
        if self.args.use_state_norm: s = self.state_norm(s)
        self.agent.reset_rnn()

        ep_reward = 0
        for t in range(self.args.episode_limit):
            a, a_logp = self.agent.choose_action(s, evaluate=False)
            v         = self.agent.get_value(s)

            s2, r, done, info = self.env.step(a)
            ep_reward += r
            if self.args.use_reward_scaling: r = self.reward_scaling(r)

            dw = done and (t+1 != self.args.episode_limit)
            self.buffer.store_transition(t, s, v, a, a_logp, r, dw)

            s = s2
            if self.args.use_state_norm: s = self.state_norm(s)
            if done:
                logging.info(f"Episode done at step {t + 1} with reward {ep_reward:.3f}, info={info}")
                break

        v_last = self.agent.get_value(s)
        self.buffer.store_last_value(t+1, v_last)

        self.total_steps += t+1
        return info, t+1

    # ---------------------- 评估 -------------------------------
    def evaluate_policy(self, n=10):
        success = collision = timeout = 0
        R = []
        for _ in range(n):
            s, _ = self.env.reset()
            if self.args.use_state_norm: s = self.state_norm(s, update=False)
            self.agent.reset_rnn()
            ep_r, done = 0, False
            while not done:
                a, _ = self.agent.choose_action(s, evaluate=True)
                s, r, done, info = self.env.step(a)
                if self.args.use_state_norm: s = self.state_norm(s, update=False)
                ep_r += r
            success += int(info["arrival"])
            collision += int(info["collision"])
            timeout += int(info["timeout"])
            R.append(ep_r)

        return {
            "test/average_reward": np.mean(R),
            "test/success_rate":   success/n,
            "test/collision_rate": collision/n,
            "test/timeout_rate":   timeout/n
        }


# -------------------- argparse 超参 ---------------------------
def get_args():
    p = argparse.ArgumentParser("PPO‑连续动作‑RNN for NavRL")
    # 通用
    p.add_argument("--max_train_steps", type=int, default=6_000_00)
    p.add_argument("--device", default="cuda:0")      # CUDA:0 / cpu
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evaluate_freq", type=int, default=5_000)
    p.add_argument("--save_freq", type=int, default=10)
    # PPO / 网络
    p.add_argument("--batch_size", type=int,  default=15)
    p.add_argument("--mini_batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int,  default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lamda", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--K_epochs", type=int,  default=10)
    # Tricks
    p.add_argument("--use_adv_norm",       type=bool, default=True)
    p.add_argument("--use_state_norm",     type=bool, default=False)
    p.add_argument("--use_reward_scaling", type=bool, default=True)
    p.add_argument("--entropy_coef", type=float, default=0.00)
    p.add_argument("--use_lr_decay",  type=bool, default=True)
    p.add_argument("--use_grad_clip", type=bool, default=True)
    p.add_argument("--use_orthogonal_init", type=bool, default=True)
    p.add_argument("--set_adam_eps", type=bool, default=True)
    p.add_argument("--use_tanh", type=bool, default=True)   # 激活
    p.add_argument("--use_gru",  type=bool, default=True)   # GRU / LSTM
    return p.parse_args()


# --------------------------- main -----------------------------
if __name__ == "__main__":
    args = get_args()
    for seed in [0, 10, 100]:
        Runner(args, seed).run()