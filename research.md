# Research Directions / 研究方向

## Swarm Intelligence for Robotics / 面向机器人的群智能研究

Our research focuses on swarm intelligence principles and their applications in multi-robot systems, aiming to enable intelligent collective behavior in complex real-world environments.

---

## Core Research Areas / 核心研究领域

### 1. Multi-Agent Cooperative Localization / 多智能体协同定位

Developing distributed localization algorithms for robot teams using reinforcement learning, federated learning, and belief propagation techniques.

**Key Technologies:**
- Particle Filter-based Cooperative Tracking
- Reinforcement Learning Compensated Filter
- Federated Learning for Privacy-Preserving Localization
- Belief Propagation for Target Tracking

**Representative Publications:**
- Cooperative localization for multi-agents based on reinforcement learning compensated filter (IEEE JSAC, 2024)
- Federated learning-enabled cooperative localization in multi-agent system (IJWIN, 2024)
- Spatial-temporal constrained particle filter for cooperative target tracking (JNCA, 2021)

---

### 2. Sim2Real Transfer / 仿真到现实迁移

Bridging the gap between simulation and real-world deployment for swarm robotics systems.

**Research Challenges:**
- Reality Gap: Policies successful in simulation fail on real robots
- Deployment Gap: Transfer across different physical platforms
- Scalability Gap: Policies trained on small swarms fail at larger scales

**Frontier Methods:**
| Method | Description |
|--------|-------------|
| Real2Sim2Real | Self-supervised representation learning + novelty search |
| GNN-based Transfer | Graph neural networks for zero-shot scale transfer |
| Domain Randomization | Training with parameter variations for robustness |
| Federated Distributed Training | Decentralized learning with 16-22% performance drop (vs 45-68% baseline) |

---

### 3. Explainable Swarm Intelligence (xSwarm) / 可解释群智能

Making swarm behavior transparent and interpretable for human operators and regulators.

**Why Explainability Matters:**
- Emergent behaviors are unpredictable
- Regulators require transparency for operational approval
- Human-swarm collaboration needs mutual understanding

**Approaches:**
| Approach | Technology |
|----------|------------|
| LLM-Powered Swarms | Replace hard-coded rules with LLM prompts for natural language explanations |
| Temporal Logic Queries | Encode behavior queries to trace decision causality |
| Audit Trail | Log reasoning process before high-stakes actions |
| xSwarm Design Framework | Expert-driven design space for explainable swarms |

---

### 4. Safe & Secure Swarm Robotics / 安全可信群智能

Ensuring robustness against adversarial attacks and failures in swarm systems.

**Threat Models:**
| Attack Type | Description |
|-------------|-------------|
| Data Poisoning | Inject malicious samples into collective learning |
| Adversarial Attack | Perturb sensor inputs to deceive perception |
| Pheromone Attack | Forge stigmergy signals to disrupt coordination |
| Byzantine Attack | Compromised robots send false information |
| Sybil Attack | Create fake identities to manipulate consensus |

**Defense Strategies:**
- Blockchain Consensus: Smart contracts detect and exclude Byzantine robots
- Quarantine Strategies: Isolate and mitigate attacked individuals
- Robust Multi-Robot Tracking: Against sensing and communication attacks
- Bounded-Time Recovery: Swarm recovers from failures in < 7 seconds

---

### 5. Human Activity Recognition / 人体活动识别

Leveraging deep learning and wireless sensing for activity recognition.

**Key Technologies:**
- Deep Neural Networks (InnoHAR - 401 citations)
- WiFi CSI-based Sensing (WiDriver - 93 citations)
- IMU/TOA Sensor Fusion

---

### 6. Near-Ground Localization & UWB / 近地定位与超宽带通信

Developing accurate positioning systems for ground robots using UWB and sensor fusion.

**Research Topics:**
- UWB Radio Channel Modeling for Swarm Robots
- TOA Ranging Error Modeling
- IMU/TOA Fusion for 3D Localization
- Height-Dependent Positioning

---

## Research Philosophy / 研究理念

- **Bio-Inspired**: Drawing from nature's swarm systems (ants, bees, birds, fish)
- **Scalable**: From small teams to large-scale swarms
- **Robust**: Decentralized systems resilient to individual failures
- **Adaptive**: Self-organizing behavior in dynamic environments
- **Trustworthy**: Explainable, safe, and verifiable collective intelligence

---

## Emerging Trends / 前沿趋势

### LLM + Swarm Intelligence
```
NetLogo <-> Python Extension <-> GPT-4o API

Applications:
- Ant Colony Foraging with natural language control
- Bird Flocking with explainable decisions
```

### Federated Learning for Swarms
- Distributed training without centralized data
- Privacy-preserving collaboration
- Communication-efficient parameter sharing

### Industry Applications (2024-2025)
| Company/Project | Progress |
|-----------------|----------|
| Thales COHESION | Autonomous drone swarm tactical coordination |
| OpenAI Swarm | Open-source multi-AI agent collaboration framework |
| Market Size | $1.11B → $1.46B in 2025 (CAGR 31.6%) |

---

## Contact / 联系方式

For collaboration inquiries, please contact:
- Email: duansh@ustb.edu.cn
- Email: ustbmicl_sirr@gmail.com

---

*Last Updated: January 2025*
