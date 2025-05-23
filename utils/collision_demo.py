import pybullet as p
import pybullet_data
import time

def main():
    # ===== 配置参数 =====
    urdf_path = "/home/congshan/uav/uav_roundup/navigation_strategy/assets/cf2x.urdf"  # 你的URDF文件路径
    init_pos = [0, 0, 100]            # 初始位置，z要足够高
    global_scaling = 25.0
    color = [0, 0, 1, 1]            # 蓝色

    # 启动物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.setRealTimeSimulation(0)

    # 加载地面
    p.loadURDF("plane.urdf")

    # === 创建方块障碍物 ===
    obstacle_half_extents = [1, 1, 1]  # 尺寸 2x2x2 m
    obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
    obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=[1, 0, 0, 1])
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obs_col,
        baseVisualShapeIndex=obs_vis,
        basePosition=[0, 0, 1]  # 中心在 z=1，高度=2
    )

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

    # 碰撞形状确认
    shapes = p.getCollisionShapeData(drone_id, -1)
    if not shapes:
        print("❌ 警告：无人机模型中缺少 <collision> 元素，可能不会检测到碰撞")
    else:
        print("✅ 已加载 collision shape:", shapes[0])

    print("✅ 初始化完成，开始模拟无人机自由下落...")

    collided = False
    while True:
        linear_velocity = [0, 0, -10]  # 在 x 方向以 5 m/s 的速度前进
        p.resetBaseVelocity(drone_id, linearVelocity=linear_velocity)
        # 施加一个向下的外力（一次性或持续）
        p.applyExternalForce(
            objectUniqueId=drone_id,
            linkIndex=-1,
            forceObj=[0, 0, -10],
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME
        )
        p.stepSimulation()
        contacts = p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id)
        if contacts and not collided:
            print(f"共 {len(contacts)} 个接触点")
            collided = True
        time.sleep(1.0 / 240.0)

if __name__ == "__main__":
    main()
