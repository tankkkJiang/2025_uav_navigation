import torch
import numpy as np
import gym
import argparse
from model.ppo_discrete.normalization import Normalization, RewardScaling
from model.ppo_discrete.replaybuffer import ReplayBuffer
from model.ppo_discrete.ppo_discrete import PPO_discrete
from env.navigation_env import NavigationEnv
import wandb
import yaml
from datetime import datetime
import os
import logging


def evaluate_policy(args, env, agent, state_norm):
    times = 20
    total_reward = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    step_counts = []
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, infos = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
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
    # timeout_rate = timeout_count / times
    # avg_steps = sum(step_counts) / times
    average_reward = total_reward / times


    return {
        "test/average_reward": average_reward,
        "test/success_rate": success_rate,
        "test/collision_rate": collision_rate,
    }


def main(args):
    # === 2. 读取配置文件
    with open("config/navigation_env_config.yaml", "r", encoding="utf-8") as f:
        env_config = yaml.safe_load(f)
    # 1. 基础信息
    algo_name = "ppo_discrete"
    seed = args.seed

    # 2. 生成run name
    time_str = datetime.now().strftime('%m%d_%H%M%S')  # 月日_时分
    run_name = f"{algo_name}_s{seed}_{time_str}"

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

    with open(os.path.join(config_dir, "env_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(env_config, f)
    
    # 保存为 YAML 文件
    with open(os.path.join(config_dir, "ppo_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)
    
    wandb.init(
        project="drone_navigation",
        name=run_name,
        config= {**vars(args), **env_config},                  # 直接同步wandb记录超参数
        dir=log_dir                      # wandb日志也放到log_dir下面
    )

    env = NavigationEnv(env_config)
    env_evaluate = NavigationEnv(env_config)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    episode_total_count = 0
    episode_success_count = 0
    episode_collision_count = 0
    episode_other_count = 0

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, infos = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            if done:
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
                }, step=total_steps)

                for key, value in infos.items():
                    if key.startswith("episode/"):
                        comp = key.split("/")[1]  # e.g. "target_progress"
                        wandb.log({f"train/{comp}": value}, step=total_steps)
                
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                metrics = agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0          
                
                wandb.log(metrics, step=total_steps)
                # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
                eval_metrics = evaluate_policy(args, env, agent, state_norm)
                wandb.log(eval_metrics, step=total_steps)
                # 保存模型权重
                if evaluate_num % args.save_freq == 0:
                    model_path = os.path.join(model_dir, f"model_update_{total_steps}.zip")
                    torch.save({
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'args': args  # 可选：保存训练参数
                    }, model_path)
                    print(f"[INFO] Saved model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--algo_name", type=str, default="ppo_discrete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g., "cuda:0" or "cpu"')
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args)
