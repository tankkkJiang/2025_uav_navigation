import numpy as np
import heapq
import logging

class AStarPlanner3D:
    def __init__(self, voxel_map, voxel_size=2.0):
        self.map = voxel_map
        self.shape = voxel_map.shape
        self.voxel_size = voxel_size
        self.directions = self.get_26_connected_directions()
        self.origin = np.array([
            -self.shape[0] // 2 * voxel_size,
            -self.shape[1] // 2 * voxel_size,
            0  # 通常 z 从 0 开始
        ])
    
    def world_to_voxel(self, xyz):
        """世界坐标 -> voxel 索引"""
        ijk = ((np.array(xyz) - self.origin) / self.voxel_size).astype(int)
        return tuple(ijk)

    def voxel_to_world(self, ijk):
        """voxel 索引 -> 世界坐标"""
        xyz = np.array(ijk) * self.voxel_size + self.origin + self.voxel_size / 2
        return xyz

    def get_26_connected_directions(self):
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dx, dy, dz))
        return directions

    def in_bounds(self, x, y, z):
        return (0 <= x < self.shape[0] and
                0 <= y < self.shape[1] and
                0 <= z < self.shape[2])

    def is_free(self, x, y, z):
        x = int(x)
        y = int(y)
        z = int(z)
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and 0 <= z < self.map.shape[2]:
            return self.map[x, y, z] == 0
        return False

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def plan(self, start_world, goal_world):
        # 世界坐标 -> 体素索引
        start = self.world_to_voxel(start_world)
        goal = self.world_to_voxel(goal_world)
        
        logging.info(f"🚀 启动A*路径规划 start_voxel={start}, goal_voxel={goal}")

        # 合法性检查
        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            logging.error("❌ 起点或终点超出地图范围")
            return None
        if not self.is_free(*start):
            logging.error("❌ 起点被障碍物占据")
            return None
        if not self.is_free(*goal):
            logging.error("❌ 终点被障碍物占据")
            return None

        # A*初始化
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        visited_nodes = 0

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            visited_nodes += 1

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                logging.info(f"✅ 路径规划完成，访问 {visited_nodes} 个节点，路径长度：{len(path)}")
                # 转为世界坐标路径
                return [self.voxel_to_world(p) for p in path]

            for dx, dy, dz in self.directions:
                neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                if not self.in_bounds(*neighbor):
                    continue
                if not self.is_free(*neighbor):
                    continue
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        logging.warning("⚠️ 未找到可行路径")
        return None  # No path found


    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
