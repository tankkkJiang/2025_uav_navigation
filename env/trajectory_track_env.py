"""
env/low_level_env.py
"""

import gym
import numpy as np
import random
import logging
from sim.world import World



# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class TrajectoryTrackingEnv(gym.Env):
    def __init__(self, env_params):
        super().__init__()
        self.env_params = env_params
        self._init_simulation()
        self.reset()

    def _init_simulation(self):
        self.sim = World(env_params=self.env_params, use_gui=self.env_params['use_gui'])

    def reset(self):
        logging.info("仿真环境重置 ...")
        self.sim.reset()

        # 初始化起点与目标点
        self._generate_initial_and_target_positions()

        # 设置无人机初始位置
        self._set_drone_initial_position()

    
    def _generate_initial_and_target_positions(self):
        bounds = self.env_params['scene']['region']
        self.init_position = np.array([
            random.uniform(bounds['x_min'], bounds['x_max']),
            random.uniform(bounds['y_min'], bounds['y_max']),
            random.uniform(bounds['z_min'], bounds['z_max']),
        ])

        # 在 ±25m 范围内生成目标偏移
        delta = 25.0
        offset = np.random.uniform(-delta, delta, size=3)
        target_candidate = self.init_position + offset

        # 裁剪到地图边界
        self.target_position = np.array([
            np.clip(target_candidate[0], bounds['x_min'], bounds['x_max']),
            np.clip(target_candidate[1], bounds['y_min'], bounds['y_max']),
            np.clip(target_candidate[2], bounds['z_min'], bounds['z_max']),
        ])

    def _set_drone_initial_position(self):
        self.sim.drone.set_position(self.init_position)
        self.sim.drone.init_pos = self.init_position


    
    def step(self, path: np.ndarray):
        """
        控制无人机沿轨迹路径飞行直到最后一个点。

        参数：
            path (np.ndarray): N x 3 的轨迹点序列
        返回：
            info (dict): 包括飞行是否完成、最终位置、是否碰撞等信息
        """
        assert path.ndim == 2 and path.shape[1] == 3, "路径应为 N x 3 的数组"

        drone = self.sim.drone
        info = {
            "reached": False,
            "collision": False,
            "final_position": None,
        }

        logging.info(f"开始沿路径飞行，共 {len(path)} 个目标点。")

        for idx, target in enumerate(path):

            while True:
                current_pos = np.array(drone.state.position)
                dist = np.linalg.norm(target - current_pos)

                # 如果已接近目标点，跳到下一个
                if dist < 0.5:
                    break

                # PID控制生成速度命令（匀速）
                direction = (target - current_pos) / dist
                velocity = direction * min(15.0, dist / (1 / 4.0))  # 最多5m/s

                # 控制并推进仿真
                drone.apply_velocity_control(velocity)
                self.sim.step()

                # 如果无人机已死亡（如撞击）
                if not drone.state.alive:
                    logging.warning(f"无人机在飞往第 {idx + 1} 个点时发生碰撞，位置：{drone.state.position}")
                    info["collision"] = True
                    info["final_position"] = drone.state.position
                    return info

        # 全部路径点完成，无碰撞
        info["reached"] = True
        info["final_position"] = drone.state.position
        logging.info("无人机成功完成所有路径点，无碰撞。")
        logging.info(f"最终位置：{info['final_position']}")
        return info


    def generate_target_position(self):
        """
        在地图范围内生成一个目标位置
        返回：
            tuple: 生成的目标位置 (x, y, z)
        """
        bounds = self.env_params['scene']['region']
        # 随机生成一个目标位置
        target_position = np.array([
            random.uniform(bounds['x_min'], bounds['x_max']),
            random.uniform(bounds['y_min'], bounds['y_max']),
            random.uniform(bounds['z_min'], bounds['z_max'])
        ])

        return target_position
