"""数据模型 + 统一执行决策动作空间（第一性原理版）。

核心：Action = { Stop } ∪ { 执行决策 = (设备, 执行类型, 工具?) }。
设备、执行类型、工具是"这一步怎么执行"这一个决策的三个属性，不拆两个头。
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEVICES = ["weak", "medium", "strong"]
EXEC_TYPES = ["finalize", "continue", "tool"]   # 收尾(出答案) / 续步(中间结果) / 调工具
TOOLS = ["calculator"]


def build_action_space():
    """动作空间：[(label, device, exec_type, tool)]，index 0 = stop。"""
    actions = [("stop", None, None, None)]
    for d in DEVICES:
        actions.append((f"{d}:finalize", d, "finalize", None))
    for d in DEVICES:
        actions.append((f"{d}:continue", d, "continue", None))
    for d in DEVICES:
        for t in TOOLS:
            actions.append((f"{d}:tool:{t}", d, "tool", t))
    return actions


ACTION_SPACE = build_action_space()      # ACTION_SPACE[i] = (label, device, exec_type, tool)
NUM_ACTIONS = len(ACTION_SPACE)          # 10 = 1 stop + 3 finalize + 3 continue + 3 tool


@dataclass
class DeviceState:
    device_id: str
    cap_embed: list                       # 4 维能力向量
    can_execute: bool = True
    online: bool = True


@dataclass
class StepState:
    progress: list = field(default_factory=lambda: [0.0, 0.0])   # [步进度, 完成度估计]
    last_output: str = ""
    cost_accum: float = 0.0
    device_states: list = field(default_factory=list)            # List[DeviceState]
    step_idx: int = 0
    lam: float = 0.5                                            # 成本-质量偏好旋钮（条件化输入）


@dataclass
class StepObservation:
    output: str
    cost: float
    success: bool
    latency_ms: float = 0.0


@dataclass
class TrainingRecord:
    states: list = field(default_factory=list)     # List[StepState]
    actions: list = field(default_factory=list)     # List[int]（动作索引）
    responses: list = field(default_factory=list)   # List[str]（每步实际 response，RIVET 用）
    final_success: bool = False
    total_cost: float = 0.0