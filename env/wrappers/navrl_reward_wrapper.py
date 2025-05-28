# coding: utf‑8
"""
NavRL 论文版奖励组件
"""
import numpy as np

class _Base:
    def __init__(self, name: str, weight: float = 1.0):
        self.name   = name
        self.weight = weight
    # 默认接口：子类只需实现 _raw()
    def compute(self, env, **kw):
        return self.weight * self._raw(env, **kw)
    def _raw(self, env, **kw):
        raise NotImplementedError

# ------------------------------------------------------------------
class VelReward(_Base):
    """r_vel = v ⋅ dir_to_goal"""
    def _raw(self, env, **kw):
        p_r = np.array(env.world.drone.state.position)
        v_r = np.array(env.world.drone.state.linear_vel)
        dir_pg = env.goal_pos - p_r
        dir_pg /= (np.linalg.norm(dir_pg) + 1e-6)
        return np.dot(v_r, dir_pg)

class StaticSafetyReward(_Base):
    """r_ss = mean( log(ray_dist) )"""
    def _raw(self, env, **kw):
        rays = env.world.drone.cast_static_rays(
            num_h=env.cfg["observation"]["num_h"],
            num_v=env.cfg["observation"]["num_v"],
            max_dist=env.cfg["observation"]["max_ray"])
        return float(np.mean(np.log(np.array(rays) + 1e-3)))

class DynamicSafetyReward(_Base):
    """r_ds = mean( log(dist_to_dyn) )"""
    def __init__(self, name, weight, num_dyn):
        super().__init__(name, weight)
        self.num_dyn = num_dyn
    def _raw(self, env, **kw):
        p_r = np.array(env.world.drone.state.position)
        dyns = env.world.drone.get_nearest_dynamic_obs(k=self.num_dyn)
        if not dyns:
            return 1.0   # 没有动态障碍给一个正值
        dists = [np.linalg.norm(d["pos"] - p_r) for d in dyns]
        return float(np.mean(np.log(np.array(dists) + 1e-3)))

class SmoothReward(_Base):
    """惩罚速度突变"""
    def _raw(self, env, **kw):
        cur = kw["cur_vel"]
        prev = kw["prev_vel"]
        return -np.linalg.norm(cur - prev)

class HeightPenalty(_Base):
    """超高惩罚：负平方距离"""
    def _raw(self, env, **kw):
        p_r = np.array(env.world.drone.state.position)
        z_s = env.start_pos[2]
        z_g = env.goal_pos[2]
        delta = min(abs(p_r[2] - z_s), abs(p_r[2] - z_g))
        return -delta ** 2

class TerminalReward(_Base):
    def __init__(self, name, success_r, collision_p):
        super().__init__(name, 1.0)
        self.success_r   = success_r
        self.collision_p = collision_p
    def compute(self, env, **kw):
        arrived  = kw["arrived"]
        collided = kw["collided"]
        if arrived:
            return self.success_r
        if collided:
            return self.collision_p
        return 0.0