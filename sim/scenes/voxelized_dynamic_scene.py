# coding: utf-8
"""
sim/scenes/voxelized_dynamic_scene.py
体素化随机场景（含动态障碍）
-------------------------------------------------
继承自 VoxelizedRandomScene：
    • 保留父类静态障碍体素化逻辑
    • 额外生成 n 个圆柱动态障碍
    • 在 step(dt) 中更新动态障碍位置，并做边界反弹
"""
import numpy as np
import pybullet as p
from typing import List, Dict, Tuple
from sim.scenes.voxelized_random_scene import VoxelizedRandomScene

# ======== 工具函数 =================================================
def _spawn_dyn_cylinders(num: int,
                         x_min, x_max, y_min, y_max,
                         r_min, r_max, h_min, h_max,
                         v_max) -> List[Dict]:
    """生成 num 个圆柱动态障碍并返回列表"""
    dyn_list = []
    for _ in range(num):
        r = np.random.uniform(r_min, r_max)
        h = np.random.uniform(h_min, h_max)
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                  rgbaColor=[1, 0, 0, 0.6])
        body = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[x, y, h/2])

        ang   = np.random.uniform(-np.pi, np.pi)
        speed = np.random.uniform(0.5*v_max, v_max)
        vel   = np.array([np.cos(ang)*speed, np.sin(ang)*speed, 0], dtype=np.float32)

        dyn_list.append({'body': body, 'vel': vel, 'radius': r, 'height': h})
    return dyn_list


def _update_dyn(dyn_list: List[Dict],
                dt: float,
                x_min, x_max, y_min, y_max):
    """推进所有动态障碍，并对超出边界的分量做反弹"""
    for d in dyn_list:
        pos, orn = p.getBasePositionAndOrientation(d['body'])
        pos = np.array(pos) + d['vel'] * dt
        # X 边界
        if pos[0] < x_min or pos[0] > x_max:
            d['vel'][0] *= -1
            pos[0] = np.clip(pos[0], x_min, x_max)
        # Y 边界
        if pos[1] < y_min or pos[1] > y_max:
            d['vel'][1] *= -1
            pos[1] = np.clip(pos[1], y_min, y_max)
        # 更新位置
        p.resetBasePositionAndOrientation(d['body'], pos.tolist(), orn)

# ======== 主类 =====================================================
class DynamicVoxelizedScene(VoxelizedRandomScene):
    def __init__(self,
                 scene_size_x, scene_size_y, scene_size_z,
                 voxel_params: dict,
                 static_params: dict,
                 dynamic_params: dict):
        """
        参数拆分：
            • voxel_params  : {'size': 1.0}
            • static_params : {'num_obstacles', 'min_radius', ...}
            • dynamic_params: {'num_obstacles', 'min_radius', ... , 'max_speed'}
        """
        super().__init__(scene_size_x, scene_size_y, scene_size_z,
                         **static_params,
                         voxel_size=voxel_params['size'])
        # 动态障碍配置
        self.dyn_cfg = dynamic_params
        self.dynamic_obs: List[Dict] = []

        # 方便边界判定
        self.x_min, self.x_max = -scene_size_x/2, scene_size_x/2
        self.y_min, self.y_max = -scene_size_y/2, scene_size_y/2

    # ----------------------------------------------------------------
    def build(self):
        """生成静态 + 动态障碍"""
        super().build_scene()           # ①生成静态 + 体素化
        # ②生成动态
        self.dynamic_obs = _spawn_dyn_cylinders(
            num     = self.dyn_cfg['num_obstacles'],
            x_min   = self.x_min, x_max=self.x_max,
            y_min   = self.y_min, y_max=self.y_max,
            r_min   = self.dyn_cfg['min_radius'],
            r_max   = self.dyn_cfg['max_radius'],
            h_min   = self.dyn_cfg['min_height'],
            h_max   = self.dyn_cfg['max_height'],
            v_max   = self.dyn_cfg['max_speed']
        )

    # ----------------------------------------------------------------
    def step(self, dt: float):
        """供 World.step 调用，推进动态障碍"""
        _update_dyn(self.dynamic_obs, dt,
                    self.x_min, self.x_max,
                    self.y_min, self.y_max)