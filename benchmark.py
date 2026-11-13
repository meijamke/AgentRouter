"""自建 benchmark：执行类型 × 难度 二维任务生成（第一性原理版 M1 v5）。

两个轴：
- 执行类型轴（让策略学收尾/续步/调工具）
- 难度轴（让 weak/medium/strong 分化）：easy/medium/hard = 数字规模
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    exec_type: str      # finalize / continue / tool
    difficulty: str     # easy / medium / hard
    question: str
    answer: str


class Benchmark:
    """生成 step 级任务（每任务一道题 + GT 答案）。"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ranges(self, difficulty: str):
        if difficulty == "easy":
            return (1, 9), (1, 9), (2, 5)
        if difficulty == "medium":
            return (11, 99), (11, 99), (2, 9)
        return (100, 999), (100, 999), (3, 9)

    def sample_task(self, task_id: str) -> Task:
        exec_type = self.rng.choice(["finalize", "continue", "tool"])
        difficulty = self.rng.choice(["easy", "medium", "hard"])
        (a_lo, a_hi), (b_lo, b_hi), (c_lo, c_hi) = self._ranges(difficulty)
        a = self.rng.randint(a_lo, a_hi)
        b = self.rng.randint(b_lo, b_hi)
        c = self.rng.randint(c_lo, c_hi)

        if exec_type == "finalize":
            q = f"{a} + {b} = ?（只输出数字）"
            ans = str(a + b)
        elif exec_type == "continue":
            q = f"先算 {a} + {b}，再把结果乘以 {c}，最终是多少？（只输出数字）"
            ans = str((a + b) * c)
        else:  # tool
            q = f"用计算器计算 {a} × {b} = ?（只输出数字）"
            ans = str(a * b)
        return Task(task_id=task_id, exec_type=exec_type, difficulty=difficulty, question=q, answer=ans)