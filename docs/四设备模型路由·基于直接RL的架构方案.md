# 四设备模型路由 · 架构设计与整体方案（agent 多步路由 v22）

> **定位**：四设备（手表/手机/平板/PC）+ 云，做 **agent 多步路由**——一个 query 拆多步，每步选一台执行设备 + 选一种路由模式。
> **核心**：**单层 MDP**，直接 RL 优化「任务最终成功 − 累计调用成本」；quality-estimator-free。
> **v22 修订（今天梳理的正本清源）**：
> 1. **骨架主次纠正**：Hera（step-level 多步）升为**主骨架**，RLCascadeRouter 的 VAS/CE 降为**策略网络实现**——"级联重试"不再是主轴，而是 multi 模式内部由策略学的 handoff（Handoff Curves 实证：later handoff 未必优于 initial selection）。
> 2. **M1 就学三路由模式**（RouteWeaver）：动作空间 = `{停} ∪ {single, multi, agentic} × 设备`，模式不再写死。
> 3. **删 P(fail)**：回到 quality-estimator-free；升级信号推迟 M2 且必须过"五种自欺"清单（2026 一整簇论文证明升级信号易自欺：TRIAGE 0.571 / CreateScore 0.47 / Who Pays 0.510）。
> 4. **训练 = 离线答案表 + 查表**（RLCascadeRouter 实现细节）；IL 用 Handoff supervision target，RL 用 RIVET response-informed 值估计。
> 5. **β 只作评估天花板**（1−β），不学、不做训练 ROI gate、不做模式空间剪枝——β 是反事实宏观量，运行时不可得。

---

## 0. 守弱执行总纲

> 守弱 = 假设「收益不确定」，先做便宜、可证伪、能回退的。四条执行原则：

1. **训练靠"查表"，不靠"实时调用"**：对有 GT 的 benchmark，先一次性预生成「设备×任务→对错/成本」表，训练时查表——解决"真执行慢 + API 超时"。
2. **IL 只当"不崩"初始化，RL 真执行采样是核心**：IL 标签区分度弱，只用来让策略"先选最便宜的够用设备、别崩"；真正的弱/中/强分化靠 RL 的 success−cost 学出来。
3. **验收看"成本节省"，不看"成功率涨分"**：路由天花板 = 池内最强设备；收益 = 同样成功率省多少成本。
4. **每一步带证伪条件**：证伪就回退，不往上跳级。

---

## 0.1 三维坐标定位（守弱）

| 方案 | X 更新对象/深度 | Y 反馈硬度 | Z 路由器位置 |
|---|---|---|---|
| **AgentRouter（本方案）** | 0.45 训路由器 | 0.60 偏硬 | 0.10 外部 |
| RSI-Router 2609.34712 | 0.05 改文本策略 | 0.62 偏硬 | 0.25 外部 |
| NeoHorse 2609.08183 | 0.95 训 agent LLM | 0.50 中 | 0.90 内部 |
| CERA-MoA | 0.55 两者同训 | 0.55 中 | 0.85 内部 |

**演进主轴**：`外部·改文本（RSI-Router）→ 外部·训路由器（本方案）→ 内部·训 LLM（NeoHorse/CERA）`。守弱底线：本方案改的是「路由器」（中段），势能区在「训 LLM」（顶端），三道坎（训谁/反馈可不可靠/有没有预算），**不跳级**。

---

## 1. 一句话设计

```
用户 query → agent 拆多步 → 每步由策略在「停」∪「single/multi/agentic × 在线设备」里选一个
→ 直到任务完成；"级联重试"= multi 模式内部由策略学的 handoff（换更强设备重做这一步）
```

- 一次"路由决策" = 一步选「模式 × 设备」；
- `can_execute=False` 的设备只作入口/输出，不进动作空间（部署配置，非架构假设）。

---

## 2. 单层 MDP（核心，Hera 主骨架）

```
State:  s_t = ( 轨迹摘要, 任务进度, 上一步输出, 已用成本 C_t, 设备状态 D_t, 步数 t )
Action: a_t ∈ { Stop } ∪ { single, multi, agentic } × { 在线可执行设备 }   # 模式 × 设备，动态集合
        agentic 模式再选一个工具（M1 预定义工具集）
Reward: R = 任务最终成功 − Σ 各设备调用成本
折扣/深度: γ = 1；T_max = 可执行设备数（M1 = 3 档设备 → T_max=3）
```

**成本函数（每步）**：`κ(设备i) = c_tier(cap_embed_i) + λ_lat·latency_ms/1000 + λ_energy·energy_j + λ_kv·kv_bytes`（本机 kv=0；`c_tier` 随能力档递增保证"云最贵"）。

**λ 旋钮（吸收 RouteFM）**：λ=(λ_lat,λ_energy,λ_kv) **条件化进策略输入**（StepState 追加 λ 字段），换 λ 免重训。

**骨架主次（正本清源）**：**Hera（step-level 多步）为主、RLCascadeRouter（VAS/CE）为实现**。"级联重试"是 multi 模式内部的 handoff（策略学的动作），不是写死的默认动作。RLCascadeRouter / Hera / GraphPlanner 全是单层 MDP——单层这个结论不变，但**骨架归属从 RLCascadeRouter 纠回 Hera**。

---

## 3. 策略网络（VAS + CE + 模式门，quality-estimator-free）

```
CE： self-attention 处理 { Stop } ∪ (模式 × 设备) 实例 → 给每个 (模式,设备) 打分（能力互补）
     每个设备实例 action token：e_i = [h_s; cap_embed_i; 状态_i; f(q, device_i)]
VAS： 模式门 产 {停, single, multi, agentic} logits + Critic 估价值    # 吸收 RouteWeaver
最终： softmax( mask( [logit_stop] ∪ [logit_mode + score_i] ∪ [logit_agentic + score_tool_j + score_i] ) )
```

**无 P(fail)**：回到 RLCascadeRouter 的 quality-estimator-free——不建质量估计器、不设校准阈值。升级信号推迟 M2 且必须过"五种自欺"清单（difficulty-only 对照等）。

> **论文分工**：**Hera → 主骨架**（step-level 多步训练范式）；RLCascadeRouter → 策略网络结构（VAS/CE）；ProgRouter → 状态设计（进度）；RouteWeaver → 模式门（三模式）；RIVET → 训练侧 response-informed 值估计；Handoff Curves → supervision target。

---

## 4. 训练（离线答案表 + Hera 两阶段 + 吸收）

### 4.0 离线答案表（训练前一次性建，吸收 RLCascadeRouter）

```
对每个 task × 每个设备：调真实 API → 拿答案 + 成本 → 判对错（对照 GT）
→ answer_table[task][device] = (correct: bool, cost: float)
→ 之后 IL/RL 全查表，几十万局不碰真实 API（解决真执行慢 + API 超时）
```

### 4.1 IL 冷启动（只当"不崩"初始化 + Handoff supervision target）

```
① 标签 = 最便宜够用设备（从答案表 GT 算）
② 吸收 Handoff（2610.15993）：标签用"剩余工作所需能力档 + 前方困难步"（monotone fit），
   不用"噪声逐步对错"——Handoff 实测换 supervision target 提 7.8pp
③ 交叉熵训 VAS/CE；目的只是"先选最便宜的够用设备、别崩"
```

### 4.2 cost-aware RL（核心 · 查表采样 + RIVET）

```
① 以 IL 权重初始化
② 查表采样轨迹：每步策略采样 (模式, 设备) → 查表拿 correct/cost + 该步 response → 更新状态
③ 奖励 = 最终成功 − λ·累计成本（+ 增量 shaping r_t = U(s_{t+1})−U(s_t)，γ=1 → 轨迹回报=终态效用）
④ 吸收 RIVET（2610.23239）：训练侧加 training-only 的 response-informed value estimator
   （看得到已产生的 response），把"响应引入的回报方差"从噪声变观测协变量 → 减小估计误差；
   部署仍 pre-call（看不到 response）
⑤ REINFORCE/PPO + exploration，学"single 一步够 / multi 多步 / agentic 调工具 + 何时升级"
```

**β 只作评估天花板（1−β）**：β 是反事实宏观量（需"全模型×全query对错表"，运行时不可得），**不进 RL 输入、不做训练 ROI gate、不做模式空间剪枝**；只在 §7.6 评估时作为"多模型组合收益上限"。

---

## 5. 数据模型（schema.py）

| 结构 | 字段 |
|---|---|
| `DeviceState` | device_id, cap_embed, battery_pct, temp_c, mem_free_gb, network_bw_mbps, can_execute, online, last_heartbeat |
| `StepState` | 轨迹摘要, 任务进度, 上一步输出, C_t, D_t, t, **λ(偏好旋钮)** |
| `StepAction` | action(Stop / 模式×设备×工具) |
| `StepObservation` | latency_ms, energy_j, kv_bytes, kv_ms, quality, success |
| `TrainingRecord` | 轨迹（state/action/reward + 终态效用 + 该步 response） |
| `StrategyPackage` | version, vas_weights, ce_weights |

---

## 6. 模块清单

| 层 | 模块 | 输入 → 输出 |
|---|---|---|
| L1 | 设备发现·心跳 | — → List\<DeviceState\> |
| L1 | 设备状态采集 | 黑盒 → 归一化连续值 |
| L2 | 状态编码 | 轨迹+进度+上一步输出+设备状态+λ → StepState |
| L2 | VAS+CE+模式门 | StepState → StepAction（模式×设备） |
| L3 | 执行器 | 动作 → 本步输出+观测（本机=迁移0 / 跨设备=KV迁移+传输） |
| L4 | 离线答案表 + IL + cost-aware RL | 轨迹 → VAS/CE 权重 |

---

## 7. 演进路线

### 7.1 里程碑

| 里程碑 | 交付 | 验收 |
|---|---|---|
| **M1** | agent 多步路由：单层 MDP + VAS/CE + 模式门（三模式）+ 离线答案表 + IL/RL（3 档设备） | **同样成功率下成本节省**；多种子+误差棒；消融比 Pareto；β 作天花板 |
| **M2** | 在线更新：云侧 bandit 持续精调（吸收 MixLLM 候选集动态变化）+ 升级信号（过五自欺清单） | regret↓、无退化 |
| **M3** | 四设备全池 + KV 迁移优化 | 延迟↓、KV 成本↓ |
| **M4（远期）** | 路由内化 + 自我改进 LLM（CERA/NeoHorse） | 四门槛全满足才动 |

### 7.2 三层改进路线（守弱 · 带证伪条件）

**层1（M1→M2）**：子任务特征（替换手工 progress，AB 证伪）、技能（绑能力档，量化 token 代价）、路由器级闭环（regret 证伪）、**训练数据治理**（ReST/SPIN 前移）。

**层2（M2→M3 · 咽喉）**：无 GT 验证器（仅 R2V，硬门槛=验证器可靠）+ **升级信号（过"五种自欺"清单）**。

**层3（M4+）**：训 LLM（四门槛：验证器/预算/防 reward hacking/跨域证据）。

### 7.3 吸收做法

- RSI-Router 的「子任务/技能」= 第三档（改笔记），天花板被底座锁死，只当有限 prompt 增强；
- RouteFM（2609.37362）：匿名候选+行为观测 → cap_embed 升多维（M2）；λ 条件化（M1）；预训练底座（M2）；
- SkillEval（2610.28767）：认知诊断学 per-skill 能力画像（cap_embed 多维的现成做法）。

### 7.4 重训触发点（绑能力档不绑模型名）

换模型（同档）/ 设备池加退 → 不重训（cap_embed 更新 + 动态 mask）；任务分布漂移 → M2 在线精调；**新能力档（分布外）→ 重训/在线精调**。

### 7.5 评估口径 + 跨域验证

- **评估口径**：成功率@成本 → PGR/APD/成本节省率 → Pareto；基线含 纯弱/纯强/oracle/随机/固定阈值 + RSI-Router。
  - **天花板**：路由天花板 = 池内最强设备；**β 只作"多模型组合收益上限 1−β"**（评估用，不学、不做 ROI gate）。
  - **统计严谨**：多种子 + 标准差/置信区间。
  - **消融**：比 Pareto 前沿，不拆两列只盯成功率。
  - **评测基准**：**TwinRouterBench 开源仓**（per-step 静态标签 + live SWE-bench + leaderboard）为主；RouterWorld（多轮 + 部分可观测）作补充。
- **跨域**：AppWorld 训 → WebShop/ALFWorld 测；特征领域无关**最多少样本迁移（RouteFM 路线）**，零样本别预设。

---

## 8. 设计红线（守弱 5 条）

1. 不预设「自我改进会稳定增益」（未证明）；
2. 技能/训 LLM 不是免费午餐（量化 token 代价）；
3. 在线更新会退化（版本化回滚 + 外部锚校验）；
4. 每轮更新必做「新 vs 旧权重」对照评测（rsi_eval 式）；
5. **零样本跨域泛化是开放问题**（最多 RouteFM 少样本，别预设领域无关特征自动跨域）。

---

## 9. 论文分工表

| 论文 | 贡献 | 用在哪 |
|---|---|---|
| **Hera**（2605.24598） | step-level 多步 + IL/RL | **主骨架** |
| **RLCascadeRouter**（2608.15817） | VAS/CE + 增量奖励 + RL 直优 + 离线答案表 | **策略网络实现** |
| ProgRouter（2608.25992） | multi-view 进度 | 状态 |
| **RouteWeaver**（2610.19705） | 模式选择（single/multi/agentic）+ COMET-GRPO | 模式门（M1 就学） |
| **RIVET**（2610.23239） | response-informed 值估计 | RL 训练侧 |
| **Handoff Curves**（2610.15993） | supervision target 改造 + handoff 实证 | IL 标签 |
| β 上限（2606.27288） | 共失败率 | 评估天花板 |
| Strong Reasoners（2611.04098） | β 实测（强推理进池 premium 极小） | 天花板警示 |
| RouteFM（2609.37362） | 少样本跨域 + λ 条件化 + 预训练底座 | 跨域/M2 |
| MixLLM（2502.18482） | bandit 持续 + 候选集动态 | M2 |
| RSI-Router / NeoHorse | 子任务+技能 / 训 LLM | 层1 吸收 / M4 |
| TwinRouterBench（2605.18859）/ RouterWorld（2610.26511） | agentic 路由评测 | 评估基准 |

> **M1 最小实现只认**：Hera（骨架）+ RLCascadeRouter（VAS/CE + 离线答案表）+ RouteWeaver（模式门）+ RIVET/Handoff（训练配件）。

---

## 10. 落地路线图

| 阶段 | 对应 | 要做的事 |
|---|---|---|
| 现在 | M1 | 三模式 benchmark（模式×难度二维）→ 弱强分化验证 → 建答案表 → IL → RL → TwinRouterBench 评估 |
| 下一步 | M1→M2 | 子任务特征、技能、路由器级闭环、数据治理 |
| 中期咽喉 | M2→M3 | 无 GT 验证器 + 升级信号（五自欺清单） |
| 远期 | M4+ | 训 LLM（四门槛） |

---

> **一句话总结**：先把「路由器这一层」做到能自我改进、反馈可验证，M4 的 LLM 级自我改进才成立。最该学 RSI 家族的是「反馈闭环 + 能力档抽象」，最容易学坏的是把「技能」「训 LLM」当免费午餐。