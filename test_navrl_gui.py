"""
test_navrl_gui.py
在 PyBullet GUI 中可视化 NavRL 环境运行 / 已训练策略回放

python test_navrl_gui.py
python test_navrl_gui.py --model logs/ppo_continuous_rnn_s0_0601_120000/models/final_model.pt
"""

import os, time, argparse, yaml, torch, numpy as np
import gym
import pybullet as p
from typing import Dict

from env.navrl_env import NavRLEnv
from model.ppo_continuous_rnn.ppo_continuous_rnn import PPO_continuous_RNN
from model.ppo_continuous_rnn.normalization import Normalization       # 仅播放时可选

# ------------------------------ Utils ------------------------------
def load_env(gui: bool = True) -> NavRLEnv:
    """读取 YAML 并创建带 GUI 的 NavRLEnv"""
    config_path = "config/navrl_env_config.yaml"
    print(f"[load_env] 即将读取配置文件：{config_path}")
    print(f"[load_env] 传入的 gui 参数 = {gui}")

    with open("config/navrl_env_config.yaml", "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)
    cfg["use_gui"] = gui                             # 强制开启 GUI

    print("[load_env] 环境创建完毕，use_gui =", cfg["use_gui"])

    env = NavRLEnv(cfg)
    scene_cfg = cfg["scene"]
    obs_cfg = scene_cfg["obstacle"]
    scene_type = scene_cfg["type"]
    num_static = obs_cfg.get("num_obstacles", 0)
    dynamic_cfg = obs_cfg.get("dynamic", {})
    num_dynamic = dynamic_cfg.get("num_obstacles", 0)
    print(f"[load_env] 读取到 scene.type = {scene_type}")
    print(f"[load_env] 静态障碍数量 = {num_static}，动态障碍数量 = {num_dynamic}")
    # 如果是 voxelized 且启用了动态，则实际使用的类应该是 DynamicVoxelizedScene
    actual_scene = env.world.scene.__class__.__name__
    print(f"[load_env] 实际创建的 Scene 类 = {actual_scene}")

    return env

def draw_circle(center: np.ndarray, radius: float = 0.3, color=(0,1,0), segments: int = 36):
    """在 PyBullet GUI 中画一个圆圈"""
    theta = np.linspace(0, 2*np.pi, segments)
    pts = [(center[0] + radius*np.cos(t), center[1] + radius*np.sin(t), center[2]) for t in theta]
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i+1) % len(pts)]
        p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=2, lifeTime=0)  # lifeTime=0 永久

def mark_start_goal(env: NavRLEnv):
    """在GUI中画出 Ps(P_start) 和 Pg(P_goal)"""
    Ps = env.start_pos
    Pg = env.goal_pos
    # 画圆
    draw_circle(Ps, radius=0.3, color=(0,1,0))
    draw_circle(Pg, radius=0.3, color=(1,0,0))
    # 标字，文字大小仅示意
    p.addUserDebugText("Ps", Ps + np.array([0,0,0.5]), textColorRGB=[0,1,0], textSize=1.5, lifeTime=0)
    p.addUserDebugText("Pg", Pg + np.array([0,0,0.5]), textColorRGB=[1,0,0], textSize=1.5, lifeTime=0)
    print(f"[mark_start_goal] 已在 GUI 上可视化 Ps={Ps} 和 Pg={Pg}")

def build_agent(env: NavRLEnv, model_path: str) -> PPO_continuous_RNN:
    """根据保存目录里的 ppo_config.yaml 恢复超参并加载权重"""
    # ----- 1) 还原超参数 -----
    run_dir  = os.path.dirname(os.path.dirname(model_path))  # logs/.../models -> logs/...
    ppo_cfg_file = os.path.join(run_dir, "config/ppo_config.yaml")
    print(f"[build_agent] 将读取 PPO 配置文件：{ppo_cfg_file}")
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
    print(f"[build_agent] 加载模型权重：{model_path}")
    checkpoint = torch.load(model_path, map_location=args.device)
    agent.ac.load_state_dict(checkpoint["ac"])
    agent.ac.eval()
    agent.reset_rnn()
    return agent

# --------------------------- Main Loop -----------------------------
def random_play(env: NavRLEnv, hz: int = 30, print_freq: int = 1):
    """无模型随机动作，可视化"""
    print(">>> 随机动作测试，按 Ctrl+C 终止 ...")
    dt = 1.0 / hz
    ep = 0
    step_freq = print_freq if print_freq > 0 else float("inf")
    try:
        while True:
            obs, _ = env.reset()
            print(f"[random_play] Episode {ep + 1} 重置完成: start_pos={env.start_pos}, goal_pos={env.goal_pos}")
            if env.world.use_gui:
                mark_start_goal(env)

            terminated = False
            ep += 1
            step_cnt = 0
            while not terminated:
                a = env.action_space.sample()
                obs, r, terminated, info = env.step(a)
                # 如果 step_cnt % print_freq == 0，则打印当前步的 obs 和 action
                if step_cnt % step_freq == 0:
                    print(f"[random_play][Ep {ep}][Step {step_cnt}] action={a}, obs={obs}")
                step_cnt += 1
                time.sleep(dt)
            print(f"[Episode {ep}] 结束 | arrival={info['arrival']}  collision={info['collision']}")
    except KeyboardInterrupt:
        print("\n用户中断，已退出。")
    finally:
        env.close()

def policy_play(env: NavRLEnv, agent: PPO_continuous_RNN, hz: int = 30, print_freq: int = 1):
    """加载模型后回放，可视化"""
    print(">>> 策略回放，按 Ctrl+C 终止 ...")
    # 如果训练时用了状态归一化，可自行选择是否加载均值方差；此处简单关闭
    dt = 1.0 / hz
    ep = 0
    step_freq = print_freq if print_freq > 0 else float("inf")
    try:
        while True:
            obs, _ = env.reset()
            print(f"[policy_play] Episode {ep + 1} 重置完成: start_pos={env.start_pos}, goal_pos={env.goal_pos}")
            if env.world.use_gui:
                mark_start_goal(env)

            agent.reset_rnn()
            terminated = False
            ep += 1
            step_cnt = 0
            while not terminated:
                a, _ = agent.choose_action(obs, evaluate=True)
                obs, r, terminated, info = env.step(a)
                # 如果 step_cnt % print_freq == 0，则打印当前步的 obs 和 action
                if step_cnt % step_freq == 0:
                    print(f"[policy_play][Ep {ep}][Step {step_cnt}] action={a}, obs={obs}")
                step_cnt += 1
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
    parser.add_argument("--print_freq", type=int, default=1,
                        help = "每隔多少步打印一次动作与观测 (1 表示每步都打印，0 表示不打印)")
    args = parser.parse_args()

    # 1. 创建带 GUI 的环境
    env = load_env(gui=True)

    # 2. 随机 or 模型
    if args.model is None:
        random_play(env, hz=args.hz, print_freq=args.print_freq)
    else:
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")
        agent = build_agent(env, args.model)
        policy_play(env, agent, hz=args.hz, print_freq=args.print_freq)