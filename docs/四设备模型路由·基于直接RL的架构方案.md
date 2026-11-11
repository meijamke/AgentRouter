# 四设备模型路由 · 架构设计与整体方案（agent 多步路由版 v18）

> **定位**：四设备（手表/手机/平板/PC）+ 云，做 **agent 多步路由**——一个 query 拆多步，每步选一台执行设备/模型。
> **核心**：**单层 MDP**，直接 RL 优化「任务最终成功 − 累计调用成本」（成本随能力档递增，云/强设备自然最贵）。
> **v18 补充**：明确「路由 vs 自我改进」两个正交轴（§7.1）；M4 演进为「路由内化为 agent 能力 + 自我改进 LLM」（RSI/CERA-MoA 终局）。

---

## 1. 一句话设计

```
用户 query → agent 拆多步 → 每一步由路由策略选"执行设备"（从在线设备池选一台）→ 直到任务完成
   "级联/重试"不设独立层：当前步答得差（P(fail) 高）→ 下一步就是"换更强设备重试这一步"
```

- 一次"路由决策" = 一步选一台设备；
- `can_execute=False` 的设备（如当前部署的手表）只作入口/输出，**不进动作空间**——这是**部署配置，非架构假设**；未来某设备可执行时设 `can_execute=True` 即自动进池。

---

## 2. 单层 MDP（核心）

```
State:  s_t = ( 轨迹摘要, 任务进度, 上一步输出 + P(fail), 已用成本 C_t, 设备状态 D_t, 步数 t )
Action: a_t ∈ { Stop } ∪ { d_i : d_i ∈ 在线可执行设备池 }   # 动态集合，不枚举类型
Reward: R = 任务最终成功 − Σ 各设备调用成本              # 成本 = f(能力强度, 延迟, 能耗, KV)，云自然最贵
折扣/深度: γ = 1；不设 T_max（以 Stop 或任务自然完成为止）
```

**为什么单层**：RLCascadeRouter / Hera / GraphPlanner 全是单层 MDP。之前的"外层轨迹 + 内层级联"两层是过度设计——"级联"在 agent 多步里就是"下一步重试"，是同一 MDP 里的一个 action，不需要独立层。

---

## 3. 策略网络（VAS + CE，复用 RLCascadeRouter 结构）

```
CE： self-attention 处理 { Stop } ∪ 在线设备实例 → 给每个设备打分（能力互补）
    每个设备实例一个 action token：e_i = [h_s; cap_embed_i; 状态_i; f(q, device_i)]
    f(q, device_i) = query-设备匹配分（网络学，多维能力隐式捕获，非单标量）
VAS： Stop Gate 产 STOP/CONTINUE（这一步够不够好、要不要继续/重试）
最终： softmax( mask( [logit_stop] ∪ [logit_cont + score_i] ) )
```

**P(fail)**（学出来 + Brier 校准的失败概率）作为**状态**喂 VAS——告诉策略"上一步成色如何"。

> **论文分工**（不冗余，三个正交维度）：
> - **Hera** → 训练范式（IL 冷启动 + cost-aware RL）
> - **RLCascadeRouter** → 策略网络结构（VAS/CE + self-attention + 增量奖励）
> - **ProgRouter** → 状态设计（multi-view 任务进度）

---

## 4. 训练（Hera 两阶段）

```
① IL 冷启动：回放最高能力档设备的轨迹，端侧 agent 的每个状态打标签 = "端侧/目标动作是否一致" → 监督分类
② cost-aware RL：按"更高回报 + 更少高成本调用"更新；分组相同状态；联合优化 任务成功 + 调用效率
```

奖励归一化（对齐 RLCascadeRouter 公式 8–11）：
```
质量̂    = (P−P_min)/(P_max−P_min)；成本节省̂ = 1 − C/C_max；U = α·质量̂ + (1−α)·成本节省̂
```

---

## 5. 数据模型（schema.py）

| 结构 | 字段 |
|---|---|
| `DeviceState` | device_id, cap_embed(能力嵌入向量), battery_pct, temp_c, mem_free_gb, network_bw_mbps, can_execute, online, last_heartbeat |
| `StepState` | 轨迹摘要, 任务进度, 上一步输出+P(fail), C_t, D_t, t |
| `StepAction` | action(Stop / 设备实例id), p_fail |
| `StepObservation` | latency_ms, energy_j, kv_bytes, kv_ms, quality, success |
| `TrainingRecord` | 轨迹（state/action/reward 序列 + 终态效用） |
| `StrategyPackage` | version, vas_weights, ce_weights, p_fail_weights |

---

## 6. 模块清单（输入 → 输出）

| 层 | 模块 | 输入 → 输出 |
|---|---|---|
| L1 | 设备发现·心跳 | — → List\<DeviceState\> |
| L1 | 设备状态采集 | 黑盒 → 归一化连续值 |
| L2 | 状态编码 | 轨迹+进度+上一步输出+设备状态 → StepState（含 P(fail) 字段，编码器一并算出） |
| L2 | VAS+CE 策略 | StepState → StepAction（VAS Stop Gate 直接消费 P(fail)） |
| L3 | 执行器 | 动作 → 本步输出+观测（本机=迁移0 / 跨设备=KV迁移+传输） |
| L4 | IL 冷启动 + cost-aware RL | 轨迹 → VAS/CE 权重 |

> M1 只用 L1/L2/L3/L4 四层（L4=训练）；**记录器+脱敏、灰度下发 → M2（在线更新才需要）**。

---

## 7. 演进路线（3 个"可运行"里程碑）

| 里程碑 | 交付 | 验收 | 核心依据 |
|---|---|---|---|
| **M1** | agent 多步路由：单层 MDP + VAS/CE + Hera 两阶段训练（本机+邻居设备起步） | 多步任务成功率↑、高成本调用↓ | Hera + RLCascadeRouter |
| **M2** | 在线更新：云侧 bandit 持续精调 | regret↓、无退化 | WISERouter |
| **M3** | 四设备全池 + KV 迁移优化 | 延迟↓、KV 成本↓ | Unified AI Gateway |
| **M4（远期）** | 路由内化为 agent 能力 + 自我改进 LLM（轴A内部 + 轴B训 LLM） | 端到端自进化 | RSI / CERA-MoA / SPIN |

---

### 7.1 路由 vs 自我改进：两个正交轴（演进框架）

| 方案 | 轴A 路由器在哪 | 轴B 更新谁 |
|---|---|---|
| 本方案（AgentRouter） | 外部（独立 VAS/CE） | 路由器 |
| RSI / NeoHorse | 内部（agent 流水线一环） | agent 的 LLM |
| SPIN / ReST-MCTS / Tao | —（不一定是路由） | agent 的 LLM |
| CERA-MoA | 内部（co-evolve） | 两者都更新 |

- **轴A（路由器在哪）**：外部独立组件，还是 agent 的内在元认知能力（约束/验证/纠正 = "这任务我做不做得了、要不要升级求助"）；
- **轴B（更新谁）**：闭环里训路由器，还是训被路由的模型（agent LLM）。

**关键澄清**：SPIN/ReST-MCTS/Tao 本质是「自我改进 LLM」（反馈来自自对弈判别 / 过程奖励 / MCTS 树搜索），与"路由"无必然关系；**只有 NeoHorse 把"自我改进"和"路由采数"绑定**。

**演进判断**：M1–M3 走「轴A外部 + 轴B训路由器」（轻量、可独立训练、可复用，端侧务实）；M4+ 演进到「轴A内部 + 轴B训 LLM」（RSI/CERA-MoA 终局）。本方案 `Trajectory`/`device_env` 已在采集"执行轨迹 + 结果 + 能力"，数据管道 **RSI-ready**。

**M4 具体做法（吸收 RSI-Router，参考 2609.34712）**：
- ①「子任务」→ VAS/CE 的**状态特征**（替换手工 progress，使其能按子任务精细路由，如 Locate 用 PC / Verify 用手机）；
- ②「技能」→ 设备的**执行增强**（执行时追加"子任务×设备"专属技能 prompt，临时提升设备有效能力，等价 cap_embed 临时提升）。

**吸收原则（避免重新引入模型耦合）**：技能绑「**能力档**」不绑「模型名」；把"模型专属失败画像"下沉到 **cap_embed**（换模型只更新 cap_embed，技能不失效）；技能靠 **M2 在线微调（触发式）** 做精细自适应，而非换模型就全量重生成。这样吸收后仍保持"能力档抽象、模型可替换"的解耦优势。

**重训触发点对比**：

| 触发事件 | RSI-Router（技能绑死模型） | AgentRouter（绑死能力档 cap_embed） |
|---|---|---|
| 端侧小模型换掉/升级 | ⚠️ 重训（"子任务×模型"技能失效，分配重估） | ✅ cap_embed 更新 + M2 在线精调 |
| 云侧大模型换掉 | ⚠️ 重训（技能演化参照轨迹 r_L 变了） | ✅ 同上 |
| proposal agent 换 | ✅ 不重训（只影响下一轮进化） | 无对应 |
| 任务分布漂移 | ⚠️ 重跑整个 G 轮进化 | ✅ M2 在线 bandit 持续精调 |
| 设备池变化 | 未涵盖 | ✅ 动作空间动态 mask |

**更新频率（触发式，非固定定时）**：端侧阈值分钟级 / 云侧权重 24h 或按批 / 完整重进化周级 / 增量技能日级；触发条件 = 累计新轨迹 ≥ N、或检测到分布漂移、或设备池变化。

---

## 8. 设计红线（被证伪，明确不做）

1. ❌ 不做 one-shot 粗粒度、不做手工阈值（被 RLCascadeRouter/Hera 证伪）；
2. ❌ 不该设"内层级联"独立层（级联溶解为重试）；
3. **P(fail) 失准** → 外部锚周期校验；**退化收敛** → 回滚；**噪声裁判** → 删在线判官。

---

## 9. 论文分工表（哪些用得上、用在哪）

| 论文 | 贡献 | 用在哪 |
|---|---|---|
| Hera（2605.24598） | step-level 单层 MDP + 两阶段训练 | **训练范式（主线）** |
| RLCascadeRouter（2608.15817） | VAS/CE + self-attention + 增量奖励 + RL 直优 | **策略网络结构** |
| ProgRouter（2608.25992） | multi-view 任务进度打分 | **状态里的"进度"** |
| GraphPlanner（2604.23626） | workflow=MDP、每步选角色、图记忆 | 多智能体/选角色时 |
| R2V（2605.16604） | 过程验证器 + CVaR | 中间奖励信号、风险规避 |
| SynAct（2610.12651） | 价值加权反馈 | 反馈采样 |
| TwinRouterBench（2605.18859） | agentic 路由评测 | 评估基准 |
| Unified AI Gateway（2609.06940） | 路由+KV 联合 | M3 |
| CERA-MoA（2609.18779） | 路由-代理共演化 | 远期（M4+） |

> 结论：**主骨架 Hera 一篇够搭；充分实现需 RLCascadeRouter（策略）+ ProgRouter（状态）+ 按需取用其余**。不是"只用 3 篇"，也不是"全都要"。
> **M1 最小实现只认 2 篇**：Hera（训练范式）+ RLCascadeRouter（VAS/CE 结构）；其余（ProgRouter/GraphPlanner/R2V...）→ M2+ 或遇到具体问题时再查。
>
> **补充：自我改进 LLM 家族（与"路由"正交，M4+ 参考）**：SPIN（自对弈判别→迭代 SFT）、ReST-MCTS（过程奖励+树搜索→自训练）、Tao（Meta，RL+验证自改进）、NeoHorse/RSI（路由采数→curriculum SFT+on-policy 蒸馏，arXiv 2609.08183）。共同点 = "反馈→更新 agent LLM 权重"，区别只在反馈来源（自对弈/奖励/树搜索/路由）。**路由内化为 agent 能力 + 自我改进 = 终局形态**（见 §7.1）。