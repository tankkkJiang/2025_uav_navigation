"""
test_navrl_gui.py
在 PyBullet GUI 中可视化 NavRL 环境运行 / 已训练策略回放

python test_navrl_gui.py
python test_navrl_gui.py --model logs/ppo_continuous_rnn_s0_0601_120000/models/final_model.pt
"""

import os, time, argparse, yaml, torch, numpy as np
import gym
from typing import Dict

from env.navrl_env import NavRLEnv
from model.ppo_continuous_rnn.ppo_continuous_rnn import PPO_continuous_RNN
from model.ppo_continuous_rnn.normalization import Normalization       # 仅播放时可选

# ------------------------------ Utils ------------------------------
def load_env(gui: bool = True) -> NavRLEnv:
    """读取 YAML 并创建带 GUI 的 NavRLEnv"""
    with open("config/navrl_env_config.yaml", "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)
    cfg["use_gui"] = gui                             # 强制开启 GUI
    return NavRLEnv(cfg)

def build_agent(env: NavRLEnv, model_path: str) -> PPO_continuous_RNN:
    """根据保存目录里的 ppo_config.yaml 恢复超参并加载权重"""
    # ----- 1) 还原超参数 -----
    run_dir  = os.path.dirname(os.path.dirname(model_path))  # logs/.../models -> logs/...
    ppo_cfg_file = os.path.join(run_dir, "config/ppo_config.yaml")
    if not os.path.exists(ppo_cfg_file):
        raise FileNotFoundError(f"未找到对应的 ppo_config.yaml: {ppo_cfg_file}")

    with open(ppo_cfg_file, "r", encoding="utf-8") as f:
        args_dict = yaml.safe_load(f)

    # runtime 补充环境相关参数
    args_dict["state_dim"]   = env.observation_space.shape[0]
    args_dict["action_dim"]  = env.action_space.shape[0]
    args_dict["episode_limit"] = env.max_steps
    # 默认推理用 CPU；可手动传 device
    args_dict.setdefault("device", "cpu")
    # 转成简单对象（模拟 argparse.Namespace）
    class _A: pass
    args = _A(); args.__dict__.update(args_dict)

    # ----- 2) 创建 agent & 加载权重 -----
    agent = PPO_continuous_RNN(args)
    checkpoint = torch.load(model_path, map_location=args.device)
    agent.ac.load_state_dict(checkpoint["ac"])
    agent.ac.eval()
    agent.reset_rnn()
    return agent

# --------------------------- Main Loop -----------------------------
def random_play(env: NavRLEnv, hz: int = 30):
    """无模型随机动作，可视化"""
    print(">>> 随机动作测试，按 Ctrl+C 终止 ...")
    dt = 1.0 / hz
    ep = 0
    try:
        while True:
            obs, _ = env.reset()
            terminated = False
            ep += 1
            while not terminated:
                a = env.action_space.sample()
                obs, r, terminated, info = env.step(a)
                time.sleep(dt)
            print(f"[Episode {ep}] 结束 | arrival={info['arrival']}  collision={info['collision']}")
    except KeyboardInterrupt:
        print("\n用户中断，已退出。")
    finally:
        env.close()

def policy_play(env: NavRLEnv, agent: PPO_continuous_RNN, hz: int = 30):
    """加载模型后回放，可视化"""
    print(">>> 策略回放，按 Ctrl+C 终止 ...")
    # 如果训练时用了状态归一化，可自行选择是否加载均值方差；此处简单关闭
    dt = 1.0 / hz
    ep = 0
    try:
        while True:
            obs, _ = env.reset()
            agent.reset_rnn()
            terminated = False
            ep += 1
            while not terminated:
                a, _ = agent.choose_action(obs, evaluate=True)
                obs, r, terminated, info = env.step(a)
                time.sleep(dt)
            print(f"[Episode {ep}] 结束 | arrival={info['arrival']}  collision={info['collision']}")
    except KeyboardInterrupt:
        print("\n用户中断，已退出。")
    finally:
        env.close()

# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("NavRL GUI visualizer")
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径 (.pt). 为空则随机动作。")
    parser.add_argument("--hz", type=int, default=30,
                        help="可视化 step 频率(Hz)")
    args = parser.parse_args()

    # 1. 创建带 GUI 的环境
    env = load_env(gui=True)

    # 2. 随机 or 模型
    if args.model is None:
        random_play(env, hz=args.hz)
    else:
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")
        agent = build_agent(env, args.model)
        policy_play(env, agent, hz=args.hz)