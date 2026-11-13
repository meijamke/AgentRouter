"""离线答案表：建表（设备×任务→对错/成本）+ 查表 + 最便宜够用标签 + 弱强分化验证。"""
from __future__ import annotations

from benchmark import Task
from device_env import execute_device
from schema import DEVICES


class AnswerTable:
    def __init__(self):
        self.table = {}   # task_id -> {tier: (correct, cost)}

    def build(self, tasks: list, mock: bool = True):
        for task in tasks:
            self.table[task.task_id] = {}
            for tier in DEVICES:
                out, cost, _ = execute_device(tier, [{"role": "user", "content": task.question}], mock=mock)
                correct = (out.strip() == task.answer.strip())
                self.table[task.task_id][tier] = (correct, cost)

    def lookup(self, task_id: str, tier: str):
        return self.table[task_id][tier]

    def cheapest_correct(self, task_id: str) -> str:
        """最便宜够用设备（Handoff target 的 GT 基础）。"""
        for tier in DEVICES:
            correct, _ = self.table[task_id][tier]
            if correct:
                return tier
        return "strong"   # 全错兜底 strong（让任务能推进）

    def differentiation_report(self) -> dict:
        """弱强分化：各设备正确率。"""
        n = len(self.table)
        acc = {t: 0 for t in DEVICES}
        for tid in self.table:
            for t in DEVICES:
                acc[t] += int(self.table[tid][t][0])
        return {t: acc[t] / max(n, 1) for t in DEVICES}