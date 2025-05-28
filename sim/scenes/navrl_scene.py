# coding: utf-8
"""
sim/scenes/navrl_scene.py — 同时支持静态体素化障碍 + 动态障碍
------------------------------------------------
提供 step(dt) 方法，交由 World 在仿真循环里调用。
"""

import pybullet as p
import numpy as np
from typing import List, Dict, Tuple


# ====================================================================
# 辅助函数：生成 / 更新动态障碍
# ====================================================================

def _spawn_dynamic_obstacles(num: int,
                             region_xy: Tuple[float, float, float, float],
                             r_min: float, r_max: float,
                             h_min: float, h_max: float,
                             v_max: float) -> List[Dict]:
    """
    在给定 XY 区域内随机生成若干圆柱动态障碍
    返回列表，每项字典：{'body': id, 'vel': np.ndarray(3), 'radius': r, 'height': h}
    """
    x_min, x_max, y_min, y_max = region_xy
    obstacles = []
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

        ang = np.random.uniform(-np.pi, np.pi)
        speed = np.random.uniform(0.5*v_max, v_max)
        vel = np.array([np.cos(ang)*speed, np.sin(ang)*speed, 0.0], dtype=np.float32)

        obstacles.append({'body': body, 'vel': vel, 'radius': r, 'height': h})
    return obstacles


def _step_dynamic_obstacles(obstacles: List[Dict],
                            dt: float,
                            region_xy: Tuple[float, float, float, float]):
    """
    更新动态障碍位置，超出边界则反弹
    """
    x_min, x_max, y_min, y_max = region_xy
    for obs in obstacles:
        pos, orn = p.getBasePositionAndOrientation(obs['body'])
        pos = np.array(pos) + obs['vel'] * dt

        # 边界检测并反弹
        if pos[0] < x_min or pos[0] > x_max:
            obs['vel'][0] *= -1
            pos[0] = np.clip(pos[0], x_min, x_max)
        if pos[1] < y_min or pos[1] > y_max:
            obs['vel'][1] *= -1
            pos[1] = np.clip(pos[1], y_min, y_max)

        p.resetBasePositionAndOrientation(obs['body'], pos.tolist(), orn)


# ====================================================================
# 主场景类
# ====================================================================

class NavRLScene:
    def __init__(self,
                 scene_size_x: float,
                 scene_size_y: float,
                 scene_size_z: float,
                 static_params: dict,
                 dynamic_params: dict,
                 voxel_size: float = 1.0,
                 use_voxel: bool = True):
        """
        scene_size_*: 场地尺寸（米）
        static_params / dynamic_params: 对应 YAML 中 obstacle.static 及 obstacle.dynamic
        """
        # 区域边界
        half_x = scene_size_x / 2.0
        half_y = scene_size_y / 2.0
        self.x_min, self.x_max = -half_x, half_x
        self.y_min, self.y_max = -half_y, half_y
        self.z_min, self.z_max = 0.0, scene_size_z
        self.size_x, self.size_y, self.size_z = scene_size_x, scene_size_y, scene_size_z

        # 参数
        self.static = static_params
        self.dynamic = dynamic_params
        self.voxel_size = voxel_size
        self.use_voxel = use_voxel

        # 存储
        self.static_ids: List[int] = []
        self.dynamic_obs: List[Dict] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def build(self):
        """生成静态 + 动态障碍（地面由 World 统一处理）"""
        if self.use_voxel:
            self._build_voxel_static()
        else:
            self._spawn_static_primitives()

        self.dynamic_obs = _spawn_dynamic_obstacles(
            num=self.dynamic['num_obstacles'],
            region_xy=(self.x_min, self.x_max, self.y_min, self.y_max),
            r_min=self.dynamic['min_radius'],
            r_max=self.dynamic['max_radius'],
            h_min=self.dynamic['min_height'],
            h_max=self.dynamic['max_height'],
            v_max=self.dynamic['max_speed']
        )

    # ------------------------------------------------------------------
    # 静态障碍
    # ------------------------------------------------------------------
    def _build_voxel_static(self):
        """简单体素随机：按概率在体素中心放柱体"""
        nx = int(self.size_x / self.voxel_size)
        ny = int(self.size_y / self.voxel_size)
        prob = min(1.0, self.static['num_obstacles'] / (nx*ny))  # 控制数量

        for i in range(nx):
            for j in range(ny):
                if np.random.rand() < prob:
                    r = np.random.uniform(self.static['min_radius'], self.static['max_radius'])
                    h = np.random.uniform(self.static['min_height'], self.static['max_height'])
                    x = self.x_min + (i + 0.5) * self.voxel_size
                    y = self.y_min + (j + 0.5) * self.voxel_size
                    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                              rgbaColor=[0.6, 0.6, 0.6, 1])
                    body = p.createMultiBody(0, col, vis, basePosition=[x, y, h/2])
                    self.static_ids.append(body)

    def _spawn_static_primitives(self):
        """随机生成固定数量的圆柱 / 盒子"""
        for _ in range(self.static['num_obstacles']):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            is_cyl = np.random.rand() < 0.5
            color = [0.3, 0.8, 0.3, 1]

            if is_cyl:
                r = np.random.uniform(self.static['min_radius'], self.static['max_radius'])
                h = np.random.uniform(self.static['min_height'], self.static['max_height'])
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
                body = p.createMultiBody(0, col, vis, basePosition=[x, y, h/2])
            else:
                lx = np.random.uniform(self.static['min_radius'], self.static['max_radius'])
                ly = np.random.uniform(self.static['min_radius'], self.static['max_radius'])
                h  = np.random.uniform(self.static['min_height'], self.static['max_height'])
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, h/2])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, h/2], rgbaColor=color)
                body = p.createMultiBody(0, col, vis, basePosition=[x, y, h/2])

            self.static_ids.append(body)

    # ------------------------------------------------------------------
    # 每步更新
    # ------------------------------------------------------------------
    def step(self, dt: float):
        """供 World 调用，推进动态障碍"""
        _step_dynamic_obstacles(self.dynamic_obs, dt,
                                region_xy=(self.x_min, self.x_max, self.y_min, self.y_max))