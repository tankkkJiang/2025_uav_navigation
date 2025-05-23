import pybullet as p
import pybullet_data
import time

def main():
    # ===== 配置参数 =====
    urdf_path = "/home/congshan/uav/uav_roundup/navigation_strategy/assets/cf2x.urdf"  # 你的URDF文件路径
    init_pos = [0, 0, 100]            # 初始位置，z要足够高
    global_scaling = 50.0
    color = [0, 0, 1, 1]            # 蓝色

    # 启动物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.setRealTimeSimulation(0)

    # 加载地面
    p.loadURDF("plane.urdf")


    # === 创建无人机 ===
    ori = p.getQuaternionFromEuler([0, 0, 0])
    drone_id = p.loadURDF(
        urdf_path,
        basePosition=init_pos,
        baseOrientation=ori,
        globalScaling=global_scaling,
        useFixedBase=False
    )

    # 设置颜色
    try:
        p.changeVisualShape(drone_id, -1, rgbaColor=color)
    except Exception as e:
        print("⚠️ 无法修改颜色（可能模型没有 visual shape）:", e)

    # 设置动力学属性
    p.changeDynamics(
        drone_id,
        -1,
        restitution=0.0,
        lateralFriction=1.0,
        linearDamping=0.3,
        angularDamping=0.3
    )

    # 设置线性速度 (x, y, z)
    linear_velocity = [0, 0, -10]  # 在 x 方向以 5 m/s 的速度前进

    # 设置角速度 (roll, pitch, yaw)
    angular_velocity = [0, 0, 1]  # 在 z 轴上以 1 rad/s 旋转

    while True:
        # 施加一个向下的外力（一次性或持续）
        p.resetBaseVelocity(drone_id, linearVelocity=linear_velocity)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

if __name__ == "__main__":
    main()
