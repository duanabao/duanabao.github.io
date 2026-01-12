# PINN + Safe MARL: 物理信息驱动的安全多智能体强化学习

## 研究方向概述

**Physics-Informed Neural Networks (PINN) + Safe Multi-Agent Reinforcement Learning (MARL)** 是将物理先验知识与安全约束强化学习相结合的前沿研究方向，旨在实现多智能体系统的安全、高效协同控制。

**核心思想**: 利用PINN编码多智能体间的**几何约束关系**（距离、方位、刚性），结合**共识协议**实现分布式安全控制。

---

## 〇、核心概念：几何约束与共识

### 0.1 多智能体几何关系

```
┌─────────────────────────────────────────────────────────────┐
│           Multi-Agent Geometric Constraints                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Agent i ●───────────────● Agent j                         │
│            \     d_ij    /                                   │
│             \           /                                    │
│         β_ik \         / β_jk                               │
│               \       /                                      │
│                ●─────●                                       │
│              Agent k                                         │
│                                                              │
│   Constraints:                                               │
│   • Distance: ||p_i - p_j|| = d_ij                          │
│   • Bearing:  (p_j - p_i)/||p_j - p_i|| = b_ij              │
│   • Rigidity: rank(R) = dn - d(d+1)/2                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 0.2 刚性理论 (Rigidity Theory)

| 概念 | 定义 | 应用 |
|------|------|------|
| **Distance Rigidity** | 距离约束唯一确定编队形状 | 编队保持 |
| **Bearing Rigidity** | 方位约束唯一确定编队 | 视觉导航 |
| **Infinitesimal Rigidity** | 刚性矩阵满秩 | 稳定性分析 |

**刚性矩阵 (Rigidity Matrix)**:
$$R(p) = \begin{bmatrix} \frac{\partial g_1}{\partial p_1} & \cdots & \frac{\partial g_1}{\partial p_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial g_m}{\partial p_1} & \cdots & \frac{\partial g_m}{\partial p_n} \end{bmatrix}$$

其中 $g_k$ 是几何约束函数。

### 0.3 共识协议 (Consensus Protocol)

**基本共识**:
$$\dot{x}_i = \sum_{j \in \mathcal{N}_i} a_{ij}(x_j - x_i)$$

**Port-Hamiltonian 共识**:
$$\dot{x} = (J - R) \nabla H(x)$$

其中:
- $J$ 是反对称矩阵（能量守恒）
- $R$ 是正半定矩阵（能量耗散）
- $H(x)$ 是 Hamiltonian 函数（系统能量）

### 0.4 PINN 在几何约束中的作用

```
┌─────────────────────────────────────────────────────────────┐
│              PINN for Geometric Constraints                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Neural Network Input: (p_i, p_j, v_i, v_j, t)             │
│                    ↓                                         │
│   ┌──────────────────────────────────┐                      │
│   │         PINN Architecture        │                      │
│   │  ┌─────────────────────────────┐ │                      │
│   │  │   Geometric Loss (PDE)      │ │                      │
│   │  │   • Distance constraint     │ │                      │
│   │  │   • Bearing constraint      │ │                      │
│   │  │   • Rigidity preservation   │ │                      │
│   │  └─────────────────────────────┘ │                      │
│   │  ┌─────────────────────────────┐ │                      │
│   │  │   Consensus Loss            │ │                      │
│   │  │   • Energy conservation     │ │                      │
│   │  │   • Stability (Lyapunov)    │ │                      │
│   │  └─────────────────────────────┘ │                      │
│   │  ┌─────────────────────────────┐ │                      │
│   │  │   Data Loss                 │ │                      │
│   │  │   • Trajectory matching     │ │                      │
│   │  └─────────────────────────────┘ │                      │
│   └──────────────────────────────────┘                      │
│                    ↓                                         │
│   Output: Control action u_i (satisfies constraints)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、研究背景与动机

### 1.1 现有方法的局限性

| 方法 | 局限性 |
|------|--------|
| 传统MARL | 缺乏安全保障，可能产生危险动作 |
| Model Predictive Control (MPC) | 计算复杂度高，难以扩展 |
| Safety Filtering | 保守性强，牺牲性能 |
| 纯数据驱动方法 | 忽略物理规律，样本效率低 |

### 1.2 PINN + Safe MARL 的优势

```
┌─────────────────────────────────────────────────────────────┐
│                    PINN + Safe MARL                         │
├─────────────────────────────────────────────────────────────┤
│  ✓ 物理约束确保动作符合动力学规律                            │
│  ✓ 安全机制保障实时约束满足                                  │
│  ✓ 多智能体协同实现复杂任务                                  │
│  ✓ 可扩展性支持大规模系统                                    │
│  ✓ 样本效率高，收敛速度快                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心论文精读

### 2.1 phMARL (2024-2025) ⭐⭐ 最核心论文

**论文信息:**
- 标题: Physics-Informed Multi-Agent Reinforcement Learning for Distributed Multi-Robot Problems
- 链接: [arXiv:2401.00212](https://arxiv.org/abs/2401.00212)
- 发表: **IEEE Transactions on Robotics (TRO) 2025**
- 代码: [GitHub](https://github.com/EduardoSebastianRodriguez/phMARL)
- 主页: [Project Page](https://eduardosebastianrodriguez.github.io/phMARL/)

**核心思想:**

用 **Port-Hamiltonian 结构** 编码多智能体的几何关系和能量守恒：

$$\dot{x} = (J(x) - R(x)) \nabla H(x) + g(x)u$$

**三大关键技术:**

| 技术 | 作用 | 与几何/共识的关系 |
|------|------|-------------------|
| **Port-Hamiltonian Policy** | 策略网络结构 | 编码能量守恒和智能体交互 |
| **Self-Attention** | 处理邻居信息 | 稀疏表示，处理动态拓扑 |
| **Graph Structure** | 建模网络拓扑 | 几何约束通过图传递 |

**Port-Hamiltonian 与共识的联系:**

```python
# phMARL 中的 Port-Hamiltonian 策略
class PortHamiltonianPolicy(nn.Module):
    def __init__(self):
        self.J = SkewSymmetricMatrix()  # 能量守恒 (共识)
        self.R = PositiveSemidefinite()  # 能量耗散 (稳定性)
        self.H = HamiltonianNetwork()    # 系统能量 (几何约束)
        self.attention = SelfAttention() # 邻居选择

    def forward(self, state, neighbors):
        # 1. 编码几何关系
        edge_features = self.encode_geometry(state, neighbors)

        # 2. 注意力机制选择重要邻居
        neighbor_info = self.attention(edge_features)

        # 3. Port-Hamiltonian 动力学
        dH = self.H.gradient(state)
        action = (self.J - self.R) @ dH + neighbor_info

        return action
```

**实验验证:**
- 任务: Flocking, Formation Control, Cooperative Navigation
- 平台: Georgia Tech Robotarium (真实机器人)
- 结果: **Zero-shot sim-to-real transfer**, 可扩展到 50+ 机器人

---

### 2.2 GCBF+ (CoRL 2023) ⭐⭐ 安全+图结构

**论文信息:**
- 标题: Learning Safe Control for Multi-Robot Systems: Methods, Verification, and Open Challenges
- 链接: [arXiv:2311.13714](https://arxiv.org/abs/2311.13714)
- 团队: **MIT CSAIL**
- 代码: [GitHub - MIT-REALM/gcbfplus](https://github.com/MIT-REALM/gcbfplus)

**核心思想:**

用**图神经网络 (GNN)** 学习可扩展的 **Control Barrier Function (CBF)**:

```
┌─────────────────────────────────────────────────────────────┐
│  GCBF+ Architecture                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Multi-robot state graph G = (V, E)                  │
│           V = {robot states}                                 │
│           E = {pairwise geometric relations}                 │
│                                                              │
│  ┌───────────────┐                                          │
│  │  GNN Encoder  │ ─── Message Passing on Graph             │
│  └───────────────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐                                          │
│  │  CBF Network  │ ─── h(x) > 0 defines safe set           │
│  └───────────────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐                                          │
│  │ Safe Action   │ ─── ḣ(x,u) + α·h(x) ≥ 0                 │
│  └───────────────┘                                          │
│                                                              │
│  Key: GNN enables zero-shot scaling to more robots          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**与几何约束的关系:**
- 边特征 (Edge Features) 编码机器人间的几何关系
- GNN消息传递保持几何约束的传播
- CBF保证安全距离约束 h(x) = ||p_i - p_j|| - d_safe

**实验结果:**
- 任务: Multi-robot collision avoidance, formation control
- 扩展: 训练4机器人，部署64+机器人
- 平台: Crazyflie quadrotors (真实验证)

---

### 2.3 MAD-PINN (2025) ⭐ 必读

**论文信息:**
- 标题: MAD-PINN: A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control
- 链接: [arXiv:2509.23960](https://arxiv.org/abs/2509.23960)
- 状态: OpenReview under review

**核心思想:**

解决 **Multi-Agent State-Constrained Optimal Control Problem (MASC-OCP)**:

$$\min_{u_i} \sum_{i=1}^{N} J_i(x_i, u_i) \quad \text{s.t.} \quad \dot{x}_i = f(x_i, u_i), \quad g(x_i) \leq 0$$

**方法框架:**

```
┌─────────────────────────────────────────────────────────────┐
│  MAD-PINN Architecture                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Epigraph Reformulation                                   │
│     └─ 将状态约束转化为可优化的 epigraph 形式                │
│                                                              │
│  2. Physics-Informed Value Function                          │
│     └─ 用 PINN 学习满足 HJB 方程的值函数                     │
│                                                              │
│  3. HJ Reachability-based Neighbor Selection                 │
│     └─ 基于可达性分析选择安全关键邻居                        │
│                                                              │
│  4. Receding-Horizon Policy Execution                        │
│     └─ 滚动时域执行，适应动态环境                            │
│                                                              │
│  5. Decentralized Deployment                                 │
│     └─ 每个智能体只需局部观测                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键技术点:**

1. **Epigraph-based SC-OCP**
   - 将不等式约束 $g(x) \leq 0$ 转化为目标函数的一部分
   - 同时优化性能和安全性

2. **PINN for Value Function**
   - 值函数 $V(x)$ 满足 Hamilton-Jacobi-Bellman (HJB) 方程
   - PINN 损失函数包含 HJB 残差

3. **Scalability Strategy**
   - 在小规模系统 (2-3 agents) 上训练
   - 部署时去中心化扩展到大规模系统

**实验结果:**
- 任务: Multi-agent navigation with collision avoidance
- 比较: 优于 MARL baseline, MPC, safety filtering
- 扩展性: 成功扩展到 50+ agents

**学习笔记:**
```python
# MAD-PINN 核心算法伪代码
class MADPINN:
    def __init__(self):
        self.value_network = PINN()  # 物理信息神经网络
        self.hjb_loss = HJBResidual()  # HJB方程残差

    def train(self, small_scale_data):
        # 1. 在小规模系统上训练
        for epoch in range(epochs):
            # PINN损失 = 数据损失 + HJB残差 + 边界条件
            loss = self.data_loss + self.hjb_loss + self.boundary_loss
            self.optimize(loss)

    def deploy(self, agent_i, local_obs):
        # 2. 去中心化部署
        neighbors = self.hj_neighbor_selection(local_obs)  # HJ可达性选择
        action = self.receding_horizon_policy(local_obs, neighbors)
        return action
```

---

### 2.2 Physics-Informed MARL for Voltage Control (2024)

**论文信息:**
- 标题: Physics-Informed Multi-Agent DRL for Distributed Voltage Control
- 应用: 配电网电压控制
- 链接: [ResearchGate](https://www.researchgate.net/publication/377045218)

**核心贡献:**

| 组件 | 技术 | 作用 |
|------|------|------|
| 物理约束 | 电压预测辅助任务 | 确保电气物理一致性 |
| 联邦学习 | Federated MARL | 保护数据隐私 |
| 物理奖励 | Physics-informed reward | 促进可再生能源利用 |

**学习笔记:**
- 物理信息可以作为**辅助任务** (auxiliary task) 引入
- 联邦学习解决多智能体数据隐私问题
- 物理奖励比纯数据驱动奖励更稳定

---

### 2.3 MACPO: Safe MARL Benchmark (AI 2023)

**论文信息:**
- 标题: Safe Multi-Agent Reinforcement Learning for Multi-Robot Control
- 期刊: Artificial Intelligence
- 链接: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0004370223000516)

**核心贡献:**

1. **问题建模**: Constrained Markov Game
   $$\max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C] \leq d$$

2. **算法**:
   - MACPO (Multi-Agent Constrained Policy Optimization)
   - MAPPO-Lagrangian

3. **基准环境**:
   - Safe MAMuJoCo
   - Safe MARobosuite
   - Safe MAIG (Isaac Gym)

**学习笔记:**
```python
# MACPO 核心思想
class MACPO:
    def update(self):
        # 1. 计算奖励优势和约束优势
        reward_adv = compute_advantage(rewards)
        cost_adv = compute_advantage(costs)

        # 2. 约束策略优化
        # 在满足约束的前提下最大化奖励
        for _ in range(K):
            ratio = new_prob / old_prob
            reward_loss = -min(ratio * reward_adv, clip(ratio) * reward_adv)
            cost_loss = ratio * cost_adv

            # Lagrangian 方法处理约束
            loss = reward_loss + lambda * (cost_loss - d)
            self.optimize(loss)

            # 更新 Lagrange 乘子
            lambda = max(0, lambda + lr * (cost - d))
```

---

### 2.4 PI-DDPG: Physics-Informed Robot Control (2025)

**论文信息:**
- 标题: Physics-Informed Reward Shaped RL Control of Robot Manipulator
- 链接: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2090447925003363)

**核心思想:**

将PINN集成到Actor-Critic架构:

```
┌─────────────────────────────────────────────────────────────┐
│  PI-DDPG Architecture                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                │
│  │  Actor  │────▶│  PINN   │────▶│ Action  │                │
│  └─────────┘     │ Dynamics│     └─────────┘                │
│       ▲          └─────────┘          │                      │
│       │               │               │                      │
│       │               ▼               ▼                      │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                │
│  │ Critic  │◀────│ Physics │◀────│  Env    │                │
│  └─────────┘     │  Loss   │     └─────────┘                │
│                  └─────────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**优势:**
- 利用动力学先验加速收敛
- 物理损失约束动作空间
- 减少不安全探索

---

### 2.5 Hierarchical Safe RL for Multi-Robot (Nature 2025)

**论文信息:**
- 标题: Multi-robot Hierarchical Safe RL with UUB Constraints
- 期刊: Scientific Reports (Nature)
- 链接: [Nature](https://www.nature.com/articles/s41598-025-89285-6)

**核心贡献:**

UBSRL (Uniformly Ultimately Bounded Safe RL):
- 分层架构: 高层规划 + 低层控制
- UUB约束: 保证状态最终有界
- 适用于多机器人自主决策

---

## 三、技术栈总结

### 3.1 PINN 技术要点

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

    def physics_loss(self, x, t):
        u = self.forward(x, t)

        # 自动微分计算导数
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]

        # 物理方程残差 (例如: 机器人动力学)
        # m * ddq + C(q, dq) + G(q) = tau
        residual = dynamics_equation(u, u_t, u_x)

        return torch.mean(residual ** 2)
```

### 3.2 Safe MARL 技术要点

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **Lagrangian** | 将约束转化为惩罚项 | 简单但可能震荡 |
| **Control Barrier Function** | 定义安全集，保证不变性 | 理论保证强 |
| **Shielding** | 运行时安全过滤 | 保守但安全 |
| **Constrained Policy Optimization** | 信赖域内约束优化 | 稳定但计算量大 |

### 3.3 多智能体扩展技术

| 技术 | 描述 |
|------|------|
| **CTDE** | 集中训练，分散执行 |
| **Graph Neural Network** | 处理智能体间拓扑结构 |
| **Mean Field** | 用平均场近似大规模交互 |
| **Attention Mechanism** | 动态选择重要邻居 |

---

## 四、研究生研究方向建议

### 4.1 研究题目

**基于几何刚性约束的物理信息多智能体强化学习**

**Rigidity-Constrained Physics-Informed Multi-Agent Reinforcement Learning for Formation Control**

### 4.2 研究问题

在多机器人编队控制中，如何利用PINN编码**几何刚性约束**实现安全、可扩展的分布式控制：

1. **几何一致性**: PINN编码距离/方位刚性约束，保证编队形状
2. **共识收敛**: Port-Hamiltonian结构保证能量耗散与共识收敛
3. **安全保障**: 基于刚性矩阵的碰撞避免
4. **可扩展性**: 图神经网络处理动态邻居拓扑

### 4.3 研究框架

```
┌─────────────────────────────────────────────────────────────┐
│       Rigidity-PINN Framework for Multi-Agent Control        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Geometric Constraint Encoding (PINN)               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Distance Rigidity: ||p_i - p_j||² = d²_ij        │    │
│  │  • Bearing Rigidity: b_ij = (p_j-p_i)/||p_j-p_i||   │    │
│  │  • Rigidity Matrix: rank(R) ≥ dn - d(d+1)/2         │    │
│  │  • PINN Loss: L = L_data + λ·L_rigidity             │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  Layer 2: Port-Hamiltonian Consensus                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ẋ = (J - R) ∇H(x)                                  │    │
│  │  • J: 反对称矩阵 (能量守恒)                          │    │
│  │  • R: 正半定矩阵 (能量耗散 → 共识收敛)               │    │
│  │  • H: Hamiltonian (编队势能 + 动能)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  Layer 3: Safe Multi-Agent RL                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Graph Neural Network for neighbor aggregation    │    │
│  │  • Self-Attention for sparse representation         │    │
│  │  • CBF/HJ safety constraints                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  Application: Formation Control, Flocking, SLAM              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 具体研究点

#### 研究点1: PINN编码刚性约束的编队控制

**核心思想**: 用PINN学习满足刚性约束的编队控制策略

**创新点**:
- PINN损失函数包含刚性矩阵约束
- 保证编队形状在控制过程中保持

```python
class RigidityPINN(nn.Module):
    def __init__(self, n_agents, dim=2):
        self.n_agents = n_agents
        self.dim = dim

    def rigidity_loss(self, positions, target_distances):
        """
        刚性约束损失: ||p_i - p_j||² - d²_ij = 0
        """
        loss = 0
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                dist_sq = torch.sum((positions[i] - positions[j])**2)
                loss += (dist_sq - target_distances[i,j]**2)**2
        return loss

    def bearing_loss(self, positions, target_bearings):
        """
        方位刚性约束: b_ij = (p_j - p_i) / ||p_j - p_i||
        """
        loss = 0
        for i, j in self.edges:
            direction = positions[j] - positions[i]
            bearing = direction / torch.norm(direction)
            loss += torch.sum((bearing - target_bearings[i,j])**2)
        return loss

    def total_loss(self, positions, velocities, target):
        L_data = self.mse(positions, target)
        L_rigidity = self.rigidity_loss(positions, self.d_ij)
        L_bearing = self.bearing_loss(positions, self.b_ij)
        return L_data + λ1*L_rigidity + λ2*L_bearing
```

#### 研究点2: Port-Hamiltonian共识策略网络

**核心思想**: 将phMARL的Port-Hamiltonian结构与刚性约束结合

**创新点**:
- Hamiltonian函数编码编队势能
- Port-Hamiltonian结构保证共识收敛

```python
class RigidityHamiltonian(nn.Module):
    """
    H(x) = 1/2 Σ_i ||v_i||² + Σ_{(i,j)∈E} V_ij(||p_i-p_j||)

    其中 V_ij 是编队势能函数:
    V_ij(d) = k/2 * (d - d*_ij)²
    """
    def __init__(self, target_formation):
        self.d_star = target_formation  # 目标距离矩阵
        self.k = 1.0  # 弹性系数

    def forward(self, positions, velocities):
        # 动能
        T = 0.5 * torch.sum(velocities**2)

        # 编队势能 (基于刚性约束)
        V = 0
        for i, j in self.edges:
            d_ij = torch.norm(positions[i] - positions[j])
            V += 0.5 * self.k * (d_ij - self.d_star[i,j])**2

        return T + V

class PortHamiltonianPolicy(nn.Module):
    def __init__(self):
        self.J = SkewSymmetricNN()      # 能量守恒
        self.R = PositiveSemidefiniteNN() # 耗散 → 共识
        self.H = RigidityHamiltonian()    # 编队能量

    def forward(self, state, neighbors):
        grad_H = torch.autograd.grad(self.H(state), state)
        action = (self.J(state) - self.R(state)) @ grad_H
        return action
```

#### 研究点3: 基于刚性的安全避碰

**核心思想**: 利用刚性矩阵的零空间检测碰撞风险

**创新点**:
- 刚性矩阵秩亏损表示编队退化/碰撞
- 设计基于刚性的Control Barrier Function

```python
class RigidityBasedCBF:
    """
    刚性约束下的安全避碰

    CBF: h(x) = min_{(i,j)∈E} (||p_i - p_j|| - d_safe)

    安全条件: ḣ(x) + α(h(x)) ≥ 0
    """
    def __init__(self, d_safe=0.5):
        self.d_safe = d_safe

    def barrier_function(self, positions):
        min_dist = float('inf')
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = torch.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d)
        return min_dist - self.d_safe

    def safe_action(self, action, positions):
        h = self.barrier_function(positions)
        h_dot = self.barrier_derivative(positions, action)

        # 如果违反安全约束，修正动作
        if h_dot + self.alpha * h < 0:
            action = self.project_to_safe(action, positions)
        return action
```

### 4.5 实验设计

| 阶段 | 内容 | 工具 |
|------|------|------|
| 仿真验证 | Multi-agent navigation | Isaac Gym / MuJoCo |
| 安全基准 | Safe MAMuJoCo benchmark | MACPO代码库 |
| 真实部署 | 小型机器人集群 | ROS2 + 实验室平台 |

### 4.6 预期贡献

1. **理论贡献**:
   - 建立PINN+Safe MARL的统一框架
   - 证明物理约束对安全性的增益

2. **方法贡献**:
   - 提出PINN-CBF (物理信息控制障碍函数)
   - 设计可扩展的去中心化训练算法

3. **应用贡献**:
   - 多机器人安全协同定位系统
   - 开源代码和基准

---

## 五、学习路线

### Phase 1: 基础 (1-2个月)

| 主题 | 资源 |
|------|------|
| PINN基础 | [PINN原论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [DeepXDE库](https://github.com/lululxvi/deepxde) |
| 强化学习 | Sutton & Barto, Spinning Up |
| 多智能体RL | [MARL Book](https://www.marl-book.com/) |

### Phase 2: 进阶 (2-3个月)

| 主题 | 资源 |
|------|------|
| Safe RL | [Safe RL Survey](https://www.sciencedirect.com/science/article/pii/S1367578824000178) |
| Control Barrier Functions | [CBF Tutorial](https://arxiv.org/abs/1903.11199) |
| HJ Reachability | [Berkeley Tutorial](https://hjreachability.github.io/) |

### Phase 3: 前沿 (持续)

| 主题 | 资源 |
|------|------|
| MAD-PINN | [论文精读](https://arxiv.org/abs/2509.23960) |
| MACPO | [代码实践](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco) |
| 最新论文 | arXiv cs.RO, cs.LG, cs.MA |

---

## 六、参考代码库

| 库 | 描述 | 链接 |
|---|------|------|
| **phMARL** | Port-Hamiltonian MARL | [GitHub](https://github.com/EduardoSebastianRodriguez/phMARL) |
| **GCBF+** | 图神经CBF安全控制 | [GitHub](https://github.com/MIT-REALM/gcbfplus) |
| DeepXDE | PINN库 | [GitHub](https://github.com/lululxvi/deepxde) |
| Safe-MARL | 安全MARL基准 | [GitHub](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco) |
| safety-gymnasium | 安全RL环境 | [GitHub](https://github.com/PKU-Alignment/safety-gymnasium) |
| MAPPO | 多智能体PPO | [GitHub](https://github.com/marlbenchmark/on-policy) |
| hj_reachability | HJ可达性 | [GitHub](https://github.com/StanfordASL/hj_reachability) |
| Georgia Tech Robotarium | 真实机器人平台 | [Robotarium](https://www.robotarium.gatech.edu/) |

---

## 七、参考文献

### 核心论文 (几何约束+共识)

1. **phMARL** (TRO 2025). Physics-Informed Multi-Agent Reinforcement Learning for Distributed Multi-Robot Problems. arXiv:2401.00212. ⭐⭐ **Port-Hamiltonian + 几何约束**

2. **GCBF+** (CoRL 2023). Learning Safe Control for Multi-Robot Systems: Methods, Verification, and Open Challenges. arXiv:2311.13714. ⭐⭐ **GNN + CBF安全**

3. **MAD-PINN** (2025). A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control. arXiv:2509.23960.

4. **MACPO** (2023). Safe multi-agent reinforcement learning for multi-robot control. Artificial Intelligence.

5. **PI-DDPG** (2025). Physics-informed reward shaped reinforcement learning control of a robot manipulator. ScienceDirect.

6. **PINN-MARL-Voltage** (2024). Physics-Informed Multi-Agent DRL for Distributed Voltage Control. ResearchGate.

7. **UBSRL** (2025). Multi-robot hierarchical safe reinforcement learning. Scientific Reports.

### 综述论文

6. **Safe Learning Survey** (2024). Learning safe control for multi-robot systems. Annual Reviews in Control.

7. **Cooperative MARL Review** (2025). Cooperative multi-agent reinforcement learning for robotic systems. SAGE.

8. **PINN Survey** (2025). Physics-Informed Neural Networks and Neural Operators for Parametric PDEs. arXiv.

---

*Last Updated: January 2025*
*Author: SIRR Lab @ USTB*
