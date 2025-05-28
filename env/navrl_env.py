# coding: utf‑8
"""
env/navrl_env.py
NavRL 环境（静态 + 动态障碍，连续 3D 速度控制）
与 sim.world.World 解耦，所有物理细节交给 World。
"""
from typing import Dict, Tuple, List
import gym
from gym import spaces
import numpy as np
import logging
import ast

from sim.world import World
from env.wrappers.navrl_reward_wrapper import (
    VelReward, StaticSafetyReward, DynamicSafetyReward,
    SmoothReward, HeightPenalty, TerminalReward
)

# -------------------------------------------------------------

class NavRLEnv(gym.Env):
    """
    观测：
        S_int ∈ R7  +  S_dyn (Nd×M)  +  S_stat (Nh×Nv) → 展平成 1 维
    动作：
        \hat V_ctrl^G ∈ [0,1]^3 （目标坐标系下归一化控制向量），后面实现 hat_vg → v_g → v_w
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self._build_sim()
        self._build_spaces()
        self._build_rewards()

        self.max_steps = cfg["episode"]["max_episode_timesteps"]
        self.action_repeat = cfg["episode"]["action_repeat"]
        self.step_cnt = 0
        self.prev_vel = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _build_sim(self):
        scene_cfg   = self.cfg["scene"]
        drone_cfg   = self.cfg["drone"]
        voxel_size  = scene_cfg.get("voxel", {}).get("size", None)
        self.world  = World(
            use_gui      = self.cfg["use_gui"],
            scene_type   = scene_cfg["type"],            # "navrl"
            scene_region = scene_cfg["region"],
            obstacle_params = scene_cfg["obstacle"],
            drone_params = drone_cfg,
            voxel_size   = voxel_size
        )

    def _build_spaces(self):
        # ----- 动作 -----
        self.v_lim = self.cfg["action"]["v_limit"]
        self.action_space = spaces.Box(low=np.float32([0, 0, 0]),
                                       high=np.float32([1, 1, 1]),
                                       dtype=np.float32)

        # ----- 观测 -----
        obs_cfg = self.cfg["observation"]
        int_dim    = 7
        dyn_dim    = obs_cfg["num_dyn"] * obs_cfg["dyn_feat_dim"]
        stat_dim   = obs_cfg["num_h"] * obs_cfg["num_v"]
        self.obs_dim = int_dim + dyn_dim + stat_dim
        self.observation_space = spaces.Box(
            low  = -np.inf, high = np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )

    def _build_rewards(self):
        r_cfg = self.cfg["reward"]
        self.reward_components = [
            VelReward               ("vel"    , r_cfg["weights"]["vel"]),
            StaticSafetyReward      ("ss"     , r_cfg["weights"]["ss"]),
            DynamicSafetyReward     ("ds"     , r_cfg["weights"]["ds"],
                                      num_dyn=self.cfg["observation"]["num_dyn"]),
            SmoothReward            ("smooth" , r_cfg["weights"]["smooth"]),
            HeightPenalty           ("height" , r_cfg["weights"]["height"]),
            TerminalReward          ("terminal",
                                     success_r = r_cfg["success_reward"],
                                     collision_p = r_cfg["collision_penalty"])
        ]
        self.comp_total = {c.name: 0.0 for c in self.reward_components}

    # ------------------------------------------------------------------
    # Gym 必备接口
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.step_cnt = 0
        self.prev_vel = np.zeros(3, dtype=np.float32)

        # 随机目标
        self.start_pos  = np.array(self.world.drone.state.position)
        self.goal_pos   = self._sample_goal()

        # 在目标/世界坐标系之间构造旋转矩阵
        gx    = self.goal_pos - self.start_pos
        gx[2] = 0.0
        gx   /= (np.linalg.norm(gx) + 1e-6)            # e1
        gz    = np.array([0, 0, 1], dtype=np.float32)  # e3
        gy    = np.cross(gz, gx)                       # e2
        self.R_g2w = np.stack([gx, gy, gz], axis=1)    # 3×3
        self.R_w2g = self.R_g2w.T

        # 记录目标在 G 坐标中的 x 坐标（其余为 0）
        self.goal_dist = np.linalg.norm(self.goal_pos - self.start_pos)

        obs = self._get_obs()  # 坐标系就绪后再取观测
        self.comp_total = {c.name: 0.0 for c in self.reward_components}

        return obs, {}

    def step(self, action: np.ndarray):
        # 归一化 → 真实速度（目标坐标系）→ 世界坐标系
        hat_vg = np.clip(action.astype(np.float32), 0.0, 1.0)
        v_g    = self.v_lim * (2.0 * hat_vg - 1.0)          # [-v_lim, v_lim]
        v_w    = self.R_g2w @ v_g                            # 3D 世界向量
        # 推进物理
        collided, _ = self.world.step(v_w, num_steps=self.action_repeat)

        self.step_cnt += 1
        arrived  = self._check_arrived()
        timeout  = self.step_cnt >= self.max_steps
        done = collided or arrived or timeout

        obs = self._get_obs()
        reward, comp_rs = self._calc_reward(obs, arrived, collided, v_g)

        info = {
            "arrival": arrived,
            "collision": collided,
            "timeout": timeout
        }
        # 统计每集奖励
        for k, v in comp_rs.items():
            self.comp_total[k] += v
            info[f"episode/{k}"] = self.comp_total[k]

        self.prev_vel = v_g  # 在目标坐标系内算 smooth
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    # 目标生成（50m×50m）
    def _sample_goal(self):
        region = self.cfg["goal_region"]         # {x_min,x_max,y_min,y_max,z}
        x = np.random.uniform(region["x_min"], region["x_max"])
        y = np.random.uniform(region["y_min"], region["y_max"])
        z = region.get("z", 1.5)
        self.world.drone.target_position = [x, y, z]
        return np.array([x, y, z], dtype=np.float32)

    # 到达判定
    def _check_arrived(self, thr=1.0):
        p   = np.array(self.world.drone.state.position)
        dis = np.linalg.norm(p - self.goal_pos)
        return dis <= thr

    # ------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------
    def _get_obs(self):
        obs_cfg = self.cfg["observation"]

        # ----- 世界 → 目标坐标系 -----
        p_r_w = np.array(self.world.drone.state.position, dtype=np.float32)
        v_r_w = np.array(self.world.drone.state.linear_vel, dtype=np.float32)
        # 位置、速度映射到 G
        p_r_g = self.R_w2g @ (p_r_w - self.start_pos)  # 3,
        v_r_g = self.R_w2g @ v_r_w  # 3,

        # --------- S_int (7)  -----------------------------------
        # 目标在 G 中是 [goal_dist, 0, 0]
        dir_pg_g = np.array([self.goal_dist, 0.0, 0.0], dtype=np.float32) - p_r_g
        dist_pg = np.linalg.norm(dir_pg_g) + 1e-6
        dir_norm = dir_pg_g / dist_pg
        s_int = np.concatenate([dir_norm, [dist_pg], v_r_g], axis=0)  # (7,)

        # --------- S_dyn ---------
        dyn_list = self.world.drone.get_nearest_dynamic_obs(k=obs_cfg["num_dyn"])  # List[dict]
        num_dyn = obs_cfg["num_dyn"]
        feat_dim = obs_cfg["dyn_feat_dim"]

        buf = np.zeros((num_dyn, feat_dim), dtype=np.float32)  # 全零初始化
        for i, d in enumerate(dyn_list[:num_dyn]):  # 最多填 num_dyn 条
            rel_w = d["pos"] - p_r_w
            rel_g = self.R_w2g @ rel_w
            dist = np.linalg.norm(rel_g) + 1e-6
            dir_g = rel_g / dist
            vel_g = self.R_w2g @ d["vel"]
            buf[i] = np.concatenate([dir_g, [dist], vel_g, d["size"]], axis=0)

        s_dyn = buf.flatten()

        # --------- S_stat ---------
        rays = self.world.drone.cast_static_rays(
            num_h=obs_cfg["num_h"], num_v=obs_cfg["num_v"],
            max_dist=obs_cfg["max_ray"])
        s_stat = np.array(rays, dtype=np.float32).flatten()

        obs = np.concatenate([s_int, s_dyn, s_stat], axis=0)
        return obs[: self.obs_dim]      # 保底裁剪

    # ------------------------------------------------------------
    # 奖励
    # ------------------------------------------------------------
    def _calc_reward(self, obs, arrived, collided, cur_vel):
        total, comp = 0.0, {}
        for rc in self.reward_components:
            r = rc.compute(self, obs=obs,
                           arrived=arrived, collided=collided,
                           cur_vel=cur_vel, prev_vel=self.prev_vel)
            comp[rc.name] = r
            total += r
        return total, comp