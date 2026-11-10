"""M1 · agent 多步路由 · 数据模型（schema）

对应《M1 详细设计》§1。核心抽象：
- 设备 = 实例（device_id），能力由 cap_embed 向量表示（无类型枚举）；
- 动作空间 = {Stop} ∪ 在线可执行设备实例；
- P(fail) 是 StepState 的字段，直接喂 VAS 的 Stop Gate。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class DeviceState:
    """一台设备的实时状态。"""

    device_id: str
    cap_embed: List[float]          # 能力嵌入向量（如 [文本/代码/数学/多模态] 或可学习 embedding）
    battery_pct: float
    temp_c: float
    mem_free_gb: float
    network_bw_mbps: float
    can_execute: bool               # 能否执行推理（纯 I/O 设备如手表 → False）
    online: bool
    last_heartbeat: float

    def is_selectable(self) -> bool:
        """是否进入动作空间：在线 + 可执行。"""
        return self.online and self.can_execute


@dataclass
class StepState:
    """单步状态 s_t（单层 MDP）。"""

    trajectory_summary: List[Any]   # 已完成步骤摘要（输出 + 结果）
    progress: List[float]           # 任务进度（ProgRouter 式 multi-view：粗 regime + 细完成度/趋势）
    last_output: str                # 上一步输出
    p_fail: float                   # 上一步失败概率（Brier 校准，喂 VAS）
    cost_accum: float               # 累计成本（延迟 + 能耗 + KV 迁移）
    device_states: List[DeviceState]
    step_idx: int


@dataclass
class StepAction:
    """单步动作 a_t = "stop" 或 目标设备 device_id（单一字段，无冗余）。"""

    action: str                     # "stop" 或 device_id
    p_fail: float

    @property
    def is_stop(self) -> bool:
        return self.action == "stop"


@dataclass
class StepObservation:
    """执行动作后观测到的结果。"""

    latency_ms: float
    energy_j: float
    kv_bytes: int                   # 跨设备 KV 迁移量（0 = 本机）
    kv_ms: float
    quality: float                  # 本步质量（客观 GT > 过程验证器分）
    success: bool
    output: str = ""                # 本步输出


@dataclass
class Trajectory:
    """一条多步轨迹（训练记录单元）。"""

    states: List[StepState] = field(default_factory=list)
    actions: List[StepAction] = field(default_factory=list)
    final_success: bool = False
    total_cost: float = 0.0


@dataclass
class StrategyPackage:
    """策略权重包（灰度下发用，M2 才需要下发闭环）。"""

    version: str
    vas_weights: Optional[Any] = None
    ce_weights: Optional[Any] = None
    p_fail_weights: Optional[Any] = None
