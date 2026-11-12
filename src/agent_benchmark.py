"""自建多步 agent benchmark（不依赖外部下载，只用 LLM API）。

任务 = 3 步「难度递增的依赖链」：
  Step1(easy)  个位数/简单算术 —— 弱设备也能对
  Step2(medium) 两位数乘法 —— 中/强设备对，弱设备大概率错
  Step3(hard)   质数判断/复杂推理 —— 仅强设备稳对

成功 = 累计正确；ground truth 纯本地可算，无需网络。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


@dataclass
class AgentStep:
    question: str        # 完整问题（含上一步答案的上下文）
    answer: str          # ground truth
    difficulty: str      # easy / medium / hard


@dataclass
class AgentTask:
    task_id: str
    steps: List[AgentStep] = field(default_factory=list)


class AgentBenchmark:
    """生成难度递增的三步链任务。"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def sample_task(self, task_id: str) -> AgentTask:
        a = self.rng.randint(3, 9)     # 个位数
        b = self.rng.randint(2, 6)
        c = self.rng.randint(13, 47)   # 两位数
        x = a + b                      # easy 答案
        y = x * c                      # medium 答案
        z = y * 2 + 1                  # hard：质数判断对象（较大）
        steps = [
            AgentStep(
                question=f"{a} + {b} = ?（只输出数字）",
                answer=str(x),
                difficulty="easy",
            ),
            AgentStep(
                question=f"你已经知道上一步答案是 {x}，计算 {x} × {c} = ?（只输出数字）",
                answer=str(y),
                difficulty="medium",
            ),
            AgentStep(
                question=f"你已经知道上一步答案是 {y}，先算 {y} × 2 + 1，再判断结果是质数还是合数（只输出「质数」或「合数」）",
                answer=("质数" if is_prime(z) else "合数"),
                difficulty="hard",
            ),
        ]
        return AgentTask(task_id=task_id, steps=steps)