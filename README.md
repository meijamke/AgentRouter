# 跨设备模型路由 · 代码仓

四设备（手表/手机/平板/PC）+ 云的 **agent 多步路由** 实现。单层 MDP，Hera 两阶段训练（IL 冷启动 + cost-aware RL），VAS/CE 策略网络。

## 设计文档

- 架构方案：[../跨设备模型路由方案/docs/四设备模型路由·基于直接RL的架构方案.md](../跨设备模型路由方案/docs/四设备模型路由·基于直接RL的架构方案.md)（v17）
- M1 详细设计：[../跨设备模型路由方案/docs/M1-详细设计.md](../跨设备模型路由方案/docs/M1-详细设计.md)

## 目录

```
src/
├── schema.py        # 数据模型（DeviceState/StepState/StepAction/StepObservation/Trajectory/StrategyPackage）
├── state_encoder.py # (下一步) 状态编码器
├── policy.py        # (下一步) VAS/CE 策略网络
├── execute.py       # (下一步) 执行器
└── train.py         # (下一步) IL 冷启动 + cost-aware RL
```

## 开发顺序（M1）

```
0 schema.py（已建）→ 1 推理 loop（encoder/CE/VAS 前向）→ 2 IL 冷启动 → 3 cost-aware RL → 4 验收
```
