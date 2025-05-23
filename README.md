# UAV Navigation

## 参考资料
在下述仓库基础上进行整理工作。
https://github.com/congshan22104/navigation_strategy.git


## 代码架构
```bash
drone_navigation/                            # 项目根目录
├── config/
│   └── navigation_env_config.yaml           # 仿真环境参数（场景、无人机、动作、奖励等）
├── env/
│   ├── wrappers/
│   │   └── reward_wrapper.py                # 各种奖励组件的实现
│   ├── navigation_env.py                    # 基于Gym封装的导航环境
│   └── trajectory_track_env.py              # 基于Gym封装的轨迹跟踪环境
├── sim/
│   ├── agents/
│   │   └── drone_agent.py                   # 无人机Agent：状态管理、物理控制、深度图获取
│   └── scenes/
│       ├── random_scene.py                  # 随机障碍物场景
│       ├── real_scene.py                    # 真实建筑模型场景
│       └── voxelized_random_scene.py        # 随机场景的体素化实现
│   └── world.py                             # PyBullet初始化、场景构建、DroneAgent管理
├── ppo_discrete_main.py                     # 启动脚本：训练 & 评估离散PPO
└── train_ppo_discrete_rnn.py                # 启动脚本：训练 & 评估带RNN的离散PPO
```

### Env 环境层
`Env` 的代码把底层仿真层的 PyBullet 仿真（`sim`，包括场景和无人机）包装成一个标准的强化学习环境，关心如何接受一个动作、调用仿真推进、收集观测、计算奖励并返回给 RL 算法。

#### Env 模块化设计
1. 动作空间：根据配置选择连续／离散、多种调节方式； 
2. 观测处理：从深度图池化到一维向量； 
3. 奖励计算：组件化地调用各个 `RewardComponent`； 
4. 信息返回：把 “碰撞／到达／超时” 这些标志，以及累计奖励等打包到 info 中。

### Sim 仿真层
接收来自上层环境层的速度指令，循环推进多帧物理仿真，与pybullet交互，构建无人机和场景。



## 其他
### 深度图
PyBullet 使用的是 OpenGL 风格的 z-buffer 深度图，这是一种 非线性映射。
映射公式使得 近处对象的深度变化很敏感，而 远处对象的深度变化被极度压缩；
因此，大多数场景中，远处的背景（例如几米外的障碍或地面）会被映射到深度值 非常接近 1.0；只有靠近摄像头的物体，其深度值才会明显小于 0.9；
可视的范围目前是0-100米，这与实际不符合，但是不重要，先能规划出来路径再说。

### 特征提取器：
concat：将深度图从（240，320）转为（3，4）大小。
resent：利用resnet18与训练网络将深度图转为维度为25的tensor。
