"""模拟设备环境：3 档能力/成本设备 + 合成多步任务 benchmark。

M1 训练不依赖真机——"设备"抽象为不同能力/成本的**真实模型**（经 OpenAI 兼容 API 调用）。
三档映射：
  weak   = Qwen3.8-Flash     （弱，快/便宜）
  medium = GLM-5.3-Flash      （中）
  strong = DeepSeek-V4-Pro    （强，贵）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .config import API_BASE, API_KEY
from .schema import DeviceState, StepObservation

# ---- 设备档位 → 模型 / 成本 / 能力嵌入 ----
CAP_DIM = 4
DEVICE_MODELS = {
    "weak": "Qwen3.8-Flash",
    "medium": "GLM-5.3-Flash",
    "strong": "DeepSeek-V4-Pro",
}
DEVICE_COST = {"weak": 0.1, "medium": 0.4, "strong": 1.0}  # 成本系数（tokens 加权）
CAP_EMBED = {
    "weak": [0.2, 0.2, 0.2, 0.2],
    "medium": [0.6, 0.6, 0.6, 0.6],
    "strong": [1.0, 1.0, 1.0, 1.0],
}

TIERS = list(DEVICE_MODELS.keys())  # ["weak", "medium", "strong"]


def make_device_pool(online: bool = True) -> List[DeviceState]:
    """构造固定 3 档设备池（每档 1 台实例）。"""
    devices = []
    for tier in TIERS:
        devices.append(DeviceState(
            device_id=tier,
            cap_embed=list(CAP_EMBED[tier]),
            battery_pct=100.0,
            temp_c=35.0,
            mem_free_gb=8.0,
            network_bw_mbps=100.0,
            can_execute=True,
            online=online,
            last_heartbeat=0.0,
        ))
    return devices


def _client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key=API_KEY)


def execute_device(tier: str, messages: List[dict], mock: bool = False) -> tuple:
    """用 tier 档设备（模型）执行一步。

    返回 (output, cost, latency_ms)。
    cost = tokens × 档位成本系数（归一化）。
    """
    if mock:
        return f"[{tier} mock]", DEVICE_COST[tier], 1.0
    t0 = time.time()
    resp = _client().chat.completions.create(
        model=DEVICE_MODELS[tier], messages=messages, temperature=0.0
    )
    latency_ms = (time.time() - t0) * 1000.0
    output = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    cost = tokens * DEVICE_COST[tier] / 1000.0  # 归一化到 ~[0,1]
    return output, cost, latency_ms


# ---- 合成多步任务 benchmark（真实 benchmark 的替身，接口对齐） ----
@dataclass
class SyntheticStep:
    question: str
    answer: str            # 标准答案（ground truth）


@dataclass
class SyntheticTask:
    task_id: str
    steps: List[SyntheticStep] = field(default_factory=list)


# 简单的事实/算术题库（不同模型正确率不同，强模型更稳）
_QUESTION_BANK = [
    ("9 的平方是多少？", "81"),
    ("法国的首都是？", "巴黎"),
    ("一年有多少个月？", "12"),
    ("水的化学式是？", "H2O"),
    ("3 + 5 × 2 = ?", "13"),
    ("太阳从哪个方向升起？", "东方"),
    ("一周有几天？", "7"),
    ("100 的一半是多少？", "50"),
    ("质数 7 的下一个质数是？", "11"),
    ("正方形有几条边？", "4"),
]


class SyntheticBenchmark:
    """合成多步任务：每个 task = steps_per_task 个连续子问题。"""

    def __init__(self, steps_per_task: int = 3, seed: int = 42):
        self.steps_per_task = steps_per_task
        import random
        self.rng = random.Random(seed)

    def sample_task(self, task_id: str) -> SyntheticTask:
        steps = [SyntheticStep(q, a) for q, a in
                 self.rng.sample(_QUESTION_BANK, self.steps_per_task)]
        return SyntheticTask(task_id=task_id, steps=steps)
