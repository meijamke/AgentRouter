"""M1 v5 全流程冒烟（mock）：建答案表 → 弱强分化 → IL → RL。"""
import torch

from answer_table import AnswerTable
from benchmark import Benchmark
from policy import Policy
from schema import DEVICES, DeviceState
from train_il import train_il
from train_rl import train_rl


def main():
    torch.manual_seed(0)
    bench = Benchmark(seed=42)
    tasks = [bench.sample_task(f"t{i}") for i in range(12)]

    print("=== ① 建离线答案表（mock）===")
    table = AnswerTable()
    table.build(tasks, mock=True)
    print(f"  弱强分化: {table.differentiation_report()}")

    print("=== ② IL 冷启动 ===")
    policy = Policy()
    train_il(policy, table, tasks, epochs=3)

    print("=== ③ cost-aware RL ===")
    devices = [DeviceState(d, [0.2] * 4) for d in DEVICES]
    train_rl(policy, table, tasks, devices, epochs=3)

    print("\nM1 v5 全流程跑通")


if __name__ == "__main__":
    main()