import os
import json
import random

# ===================== 参数配置 =====================
NUM_MAPS = 100             # 要生成的地图数量
NUM_OBSTACLES = 100         # 每张地图中的障碍物数量
ENV_SIZE_X = 500.0
ENV_SIZE_Y = 500.0
ENV_SIZE_Z = 200.0

MIN_HEIGHT = 60
MAX_HEIGHT = 200
MIN_RADIUS = 1.0             # 最小尺寸
MAX_RADIUS = 5.0            # 最大尺寸
VOXEL_SIZE = 1.0

OUTPUT_JSON_DIR = "./maps"
OUTPUT_VOXEL_DIR = "./voxel_maps"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_VOXEL_DIR, exist_ok=True)

# ===================== 障碍物生成函数 =====================
def generate_random_obstacles(NUM_OBSTACLES):
    obstacle_list = []
    for _ in range(NUM_OBSTACLES):
        x = random.uniform(-ENV_SIZE_X / 2, ENV_SIZE_X / 2)
        y = random.uniform(-ENV_SIZE_Y / 2, ENV_SIZE_Y / 2)
        radius = random.uniform(MIN_RADIUS, MAX_RADIUS)
        height = random.uniform(MIN_HEIGHT, MAX_HEIGHT)
        color = [random.random(), random.random(), random.random(), 1.0]
        is_cylinder = random.choice([True, False])
        obstacle_list.append({
            'x': x,
            'y': y,
            'radius': radius,
            'height': height,
            'color': color,
            'is_cylinder': is_cylinder
        })
    return obstacle_list

import numpy as np

def voxelize_obstacles(obstacle_list, voxel_size=1.0):
    # 地图体素范围（单位 voxel）
    nx = int(ENV_SIZE_X / voxel_size)
    ny = int(ENV_SIZE_Y / voxel_size)
    nz = int(ENV_SIZE_Z / voxel_size)

    voxel_map = np.zeros((nx, ny, nz), dtype=np.uint8)

    for obs in obstacle_list:
        x = obs['x']
        y = obs['y']
        z = obs.get('z', obs['height'] / 2)  # 如果没有 z，则默认 z 为 height 中心
        color = obs['color']
        is_cylinder = obs['is_cylinder']
        radius = obs['radius']
        height = obs['height']

        # 体素坐标范围
        min_x = int((x - radius) / voxel_size)
        max_x = int((x + radius) / voxel_size)
        min_y = int((y - (radius if is_cylinder else obs.get('depth', radius))) / voxel_size)
        max_y = int((y + (radius if is_cylinder else obs.get('depth', radius))) / voxel_size)
        min_z = int((z - height / 2) / voxel_size)
        max_z = int((z + height / 2) / voxel_size)

        # 裁剪边界
        min_x = max(0, min_x)
        max_x = min(nx - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(ny - 1, max_y)
        min_z = max(0, min_z)
        max_z = min(nz - 1, max_z)

        # 填充体素
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                for k in range(min_z, max_z + 1):
                    # 可选：精细判断是否在圆柱/盒子内（提升精度）
                    voxel_map[i, j, k] = 1
    return voxel_map

# ===================== 批量生成地图和体素 =====================
for i in range(NUM_MAPS):
    obstacle_list = generate_random_obstacles(NUM_OBSTACLES)
    voxel_map = voxelize_obstacles(obstacle_list, voxel_size=VOXEL_SIZE)

    json_path = os.path.join(OUTPUT_JSON_DIR, f"map_{i:03d}.json")
    voxel_path = os.path.join(OUTPUT_VOXEL_DIR, f"voxel_map_{i:03d}.npy")

    with open(json_path, 'w') as f_json:
        json.dump(obstacle_list, f_json, indent=2)
    np.save(voxel_path, voxel_map)

    print(f"✅ Saved: {json_path}, {voxel_path}")
