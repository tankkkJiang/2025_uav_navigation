import numpy as np
from env import TrajectoryTrackingEnv
from utils.astar_planner import AStarPlanner3D  # 即上面定义的类保存的文件
import yaml
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(
    level=logging.INFO,  # 或 logging.DEBUG 显示更详细的调试信息
)
def visualize_voxel_map_with_obstacles(voxel_map, obstacle_list, voxel_size=1.0, scene_size_x=100, scene_size_y=100):
    """
    可视化 voxel_map 中被标记为障碍物的体素块，以及 obstacle_list 中的障碍物中心点。

    参数：
        voxel_map: np.ndarray, 体素占据图，1 表示占据，0 表示空闲
        obstacle_list: list of dict，每个字典包含 x/y/z 坐标及 is_cylinder 字段
        voxel_size: float，每个体素的边长（米）
        scene_size_x: float，场景 X 方向大小（米）
        scene_size_y: float，场景 Y 方向大小（米）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 体素地图中被占据的体素索引
    occupied_voxels = np.argwhere(voxel_map == 1)
    if occupied_voxels.size > 0:
        x_voxels = occupied_voxels[:, 0] * voxel_size
        y_voxels = occupied_voxels[:, 1] * voxel_size
        z_voxels = occupied_voxels[:, 2] * voxel_size
        ax.scatter(x_voxels, y_voxels, z_voxels, c='red', marker='s', alpha=0.4, label='Occupied Voxels')

    # 障碍物中心点（世界坐标系）
    for obs in obstacle_list:
        x = obs['x'] + scene_size_x / 2
        y = obs['y'] + scene_size_y / 2
        z = obs['z']
        ax.scatter(x, y, z, c='blue', marker='o', s=40, label='Obstacle Center')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Obstacle List and Voxel Map")
    ax.legend(loc='upper left')
    plt.tight_layout()

    # 保存图片
    plt.savefig("voxel_map_visualization.pdf", dpi=300)


def plan_path_in_simulation(voxel_map, voxel_size, start_world, goal_world):
    # 初始化A*路径规划器
    planner = AStarPlanner3D(voxel_map, voxel_size)

    # 规划路径（输入为真实世界坐标）
    path = planner.plan(start_world, goal_world)

    if path:
        print(f"✅ Path found with {len(path)} steps")
    else:
        print("❌ No path found")

    return path

with open("navigation_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

env_params = config["env_params"]
# 初始化环境
env = TrajectoryTrackingEnv(env_params)

num_arrival = 0
num_collision = 0
num_total = 50

# 示例调用
visualize_voxel_map_with_obstacles(env.sim.scene.voxel_map, env.sim.scene.obstacle_list,
                                   voxel_size=env.sim.scene.voxel_size,
                                   scene_size_x=env.sim.scene.scene_size_x,
                                   scene_size_y=env.sim.scene.scene_size_y)


for i in range(num_total):
    print(f"--- 第 {i+1} 回合 ---")
    start_pos = env.sim.drone.init_pos
    end_pos = env.target_position
    logging.info(f"控制无人机从{start_pos}飞往{end_pos}")
    voxel_map = env.sim.scene.voxel_map
    voxel_size = env.sim.scene.voxel_size

    path = plan_path_in_simulation(voxel_map, voxel_size, start_pos, end_pos)
    if path is None:
        logging.info("⚠️ 未找到路径，跳过该回合")
        env.reset()
        continue

    path = np.array(path)
    info = env.step(path)

    if info.get("reached", False):
        num_arrival += 1
    if info.get("collision", False):
        num_collision += 1

    env.reset()

logging.info(f"🏁 总计 {num_total} 回合")
logging.info(f"✅ 到达目标次数: {num_arrival} ({num_arrival / num_total:.2%})")
logging.info(f"💥 碰撞次数: {num_collision} ({num_collision / num_total:.2%})")



