"""演示：自建 benchmark 的难度分层是否让弱/中/强设备正确率分化（真实 API）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.agent_benchmark import AgentBenchmark
from src.device_env import execute_device

bench = AgentBenchmark(seed=42)
print("=== 难度分层验证（真实 API，每步三档设备各答一次）===\n")
for t in range(2):
    task = bench.sample_task(f"task-{t}")
    for step in task.steps:
        results = {}
        for tier in ["weak", "medium", "strong"]:
            out, cost, _ = execute_device(tier, [{"role": "user", "content": step.question}], mock=False)
            results[tier] = (out.strip() == step.answer.strip())
        cheapest = next((x for x in ["weak", "medium", "strong"] if results[x]), "无设备答对")
        print(f"[{step.difficulty:6s}] 答={step.answer:6s}  weak={results['weak']} medium={results['medium']} strong={results['strong']}  -> 最便宜够用={cheapest}")
    print()
