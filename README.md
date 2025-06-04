# UAV Navigation

## 1. 参考资料
在下述仓库基础上进行整理工作。
https://github.com/congshan22104/navigation_strategy.git


## 2. 代码架构与功能
```bash
drone_navigation/                            # 项目根目录
├── config/
│   ├── navigation_env_config.yaml           # 仿真环境参数（场景、无人机、动作、奖励等）
│   └── navrl_env_config.yaml
├── env/
│   ├── wrappers/
│   │   ├── navrl_reward_wrapper.py
│   │   └── reward_wrapper.py                # 各种奖励组件的实现
│   ├── navigation_env.py                    # 基于Gym封装的导航环境
│   ├── navrl_env.py                         # NavRL导航环境
│   └── trajectory_track_env.py              # 基于Gym封装的轨迹跟踪环境
├── sim/
│   ├── agents/
│   │   └── drone_agent.py                   # 无人机Agent：状态管理、物理控制、深度图获取
│   └── scenes/
│       ├── random_scene.py                  # 随机障碍物场景
│       ├── voxelized_dynamic_scene.py       # 静态+动态障碍物场景的体素化实现
│       ├── real_scene.py                    # 真实建筑模型场景
│       └── voxelized_random_scene.py        # 随机场景的体素化实现
│   └── world.py                             # PyBullet初始化、场景构建、DroneAgent管理
├── utils/
│   ├── depth_feature_extractors.py          # 特征提取器
│   └── depth_utils.py                        # 深度图处理工具
├── result/                                    # 结果存储目录
├── models/                                   # 模型存储目录
│   ├── ppo_discrete_rnn/                  # PPO离散动作RNN模型
│   │   ├── normalization.py
│   │   ├── ppo_discrete_rnn.py
│   └── └── replaybuffer.py    
├── train_ppo_navrl.py
├── test_ppo_navrl.py
├── test_navrl_gui.py 
├── test_ppo_discrete_rnn.py    
├── uav_env.yml      
└── train_ppo_discrete_rnn.py                
```

### 2.1 Env 环境层
`Env` 的代码把底层仿真层的 PyBullet 仿真（`sim`，包括场景和无人机）包装成一个标准的强化学习环境，关心如何接受一个动作、调用仿真推进、收集观测、计算奖励并返回给 RL 算法。

#### 2.1.1 Env 模块化设计
1. 动作空间：根据配置选择连续／离散、多种调节方式； 
2. 观测处理：从深度图池化到一维向量； 
3. 奖励计算：组件化地调用各个 `RewardComponent`； 
4. 信息返回：把 “碰撞／到达／超时” 这些标志，以及累计奖励等打包到 info 中。

#### 2.1.2 各份代码讲解
##### env/trajectory_track_env.py — 轨迹跟踪环境

接受一条事先规划好的三维轨迹（path: np.ndarray，形状为 N×3），控制无人机按照这条轨迹逐点飞行。

每一步`step`计算当前位置到目标的距离，生成一个恒速（或基于距离衰减）的速度向量，调用 drone.apply_velocity_control 推进仿真。

##### env/navigation_env.py — 强化学习导航环境

接受单步动作（连续或离散，根据配置），计算一步的物理推进、多帧重复执行（action_repeat），并返回标准的 (obs, reward, done, info)。

### 2.2 Sim 仿真层
接收来自上层环境层的速度指令，循环推进多帧物理仿真，与pybullet交互，构建无人机和场景。为上层强化学习或轨迹跟踪环境提供清晰、简单的物理仿真接口。

#### 2.2.1 各份代码讲解 
##### sim/world.py — PyBullet世界
负责连接PyBullet、构建场景（调用 RandomScene / RealScene / VoxelizedRandomScene等）、生成 DroneAgent。

sim/world.py 是仿真总控台，统一管理场景与无人机。

##### sim/agents/drone_agent.py — 无人机Agent

##### sim/scenes/... — 场景
###### sim/scenes/random_scene.py — 随机障碍物场景
在平面上随机生成若干圆柱或长方体障碍物，并将它们添加到 PyBullet 中。维护一个 obstacle_list 以便后续查询（例如可视化或避障检测）。
###### sim/scenes/real_scene.py — 真实建筑模型场景
将用户提供的三维网格（.obj、.stl、.dae 等）加载为不可动态碰撞（静态环境），以模拟真实世界的建筑或地形。
###### sim/scenes/voxelized_random_scene.py — 体素化随机场景
在随机场景的基础上，额外维护一个三维体素网格（voxel_map），标记所有障碍物占据的体素单元。

### 2.3 奖励层

#### 2.3.1 各份代码讲解
首先关注`config/navigation_env_config.yaml`中的 `reward` 配置项，包含以下内容：
1. **extra_rewards**：用来配置那些一次性、与回合终止条件（到达／碰撞）直接相关的奖励。
2. **active_components**：列表中每一项的键名必须对应到 reward_wrapper.py 里的一个 RewardComponent 子类名；值就是它的权重 weight。


#### 2.3.2 可扩展：加新奖惩
1. 在 `reward_wrapper.py` 里新增一个继承自 `RewardComponent` 的类；
2. 在 YAML 的 `active_components` 中放开注释、写上它的类名和权重。把其他权重进行修改。

## 3. 运行
安装环境，再根据配置安装pytorch，参考：
```bash
conda env create -f uav_env.yml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 4. 其他
### 4.1 深度图
PyBullet 使用的是 OpenGL 风格的 z-buffer 深度图，这是一种 非线性映射。
映射公式使得 近处对象的深度变化很敏感，而 远处对象的深度变化被极度压缩；
因此，大多数场景中，远处的背景（例如几米外的障碍或地面）会被映射到深度值 非常接近 1.0；只有靠近摄像头的物体，其深度值才会明显小于 0.9；
可视的范围目前是0-100米，这与实际不符合，但是不重要，先能规划出来路径再说。

### 4.2 特征提取器
concat：将深度图从（240，320）转为（3，4）大小。
resent：利用resnet18与训练网络将深度图转为维度为25的tensor。

### 4.3 体素版
体素版的核心目的是用离散、规则的三维网格来替代“点、线、面”形式的稀疏几何，既能大幅度提高对静态障碍的查询速度，也方便与深度学习模型对接。

### 4.4 动态障碍物
默认情况下，PyBullet 里任何两个带碰撞体的刚体都会发生碰撞检测并产生碰撞响应（如果都是质量大于 0，则还会产生真实的物理反弹等）。

静态障碍与动态障碍默认是可以互相碰撞的，在物理仿真里会报 contact，但因为都为零质量，视觉上看不出实质挤压。在每帧的物理仿真中，动态障碍体（也是零质量）会被我们通过 _update_dyn 强制更新位置。

每隔若干个 PyBullet stepSimulation() 之后，就会检查一遍 self.scene.step(dt)，从而把所有动态障碍更新最新位置。再之后根据 self.drone.check_collision() 来判断无人机是否与任意障碍物（静态或动态）产生碰撞。

设置全部为动态障碍物的环境进行测试，发现会碰撞。
![](media/2025-06-04-00-15-08.png)