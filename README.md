# AgentRouter —— 跨设备模型路由（agent 多步，第一性原理版 M1 v5）

四设备（手表/手机/平板/PC）+ 云，agent 多步路由：每步做一次"**执行决策**"（用哪台设备 × 哪种执行类型 × 要不要工具），直接 RL 优化「任务成功 − 成本」。

## 核心架构（v23 · 自推骨架，不借论文）

```
单层 MDP：
  State  = (进度, 成本, 上一步输出, 设备状态, λ)
  Action = { Stop } ∪ { 执行决策 = (设备, 执行类型, 工具) }   ← 统一，不拆两个头
  Reward = 成功 − 成本；γ=1；T_max=3

策略网络：一个网络，扁平化 softmax 出 10 个动作的联合分布（不拆 VAS/CE）
训练：离线答案表(查表) → IL(最便宜够用标签) → cost-aware RL(REINFORCE+shaping)
```

## 文件

| 文件 | 职责 |
|---|---|
| `schema.py` | 数据模型 + 统一执行决策动作空间（10 动作） |
| `config.py` | LLM API 三档设备配置 |
| `device_env.py` | 三档设备 + 真实 API 执行 |
| `benchmark.py` | 自建 benchmark（执行类型 × 难度 二维） |
| `state_encoder.py` | StepState → 特征向量 |
| `policy.py` | 策略网络（一个网络，10 动作） |
| `answer_table.py` | 离线答案表（建表/查表/弱强分化） |
| `train_il.py` | IL 冷启动（最便宜够用标签） |
| `train_rl.py` | cost-aware RL（REINFORCE + shaping） |
| `smoke_test.py` | 骨架冒烟测试 |
| `train.py` | 全流程（答案表→IL→RL） |

## 运行

```bash
# 冒烟（mock）
python smoke_test.py
python train.py

# 真实 API
$env:LLM_API_KEY="sk-..."
$env:LLM_API_BASE="http://113.46.219.251/v1"
python train.py   # 把 mock=True 改 mock=False
```
