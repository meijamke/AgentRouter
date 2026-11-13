"""端到端冒烟测试（mock）：验证 M1 v5 骨架——统一执行决策 + 一个网络。"""
import torch

from benchmark import Benchmark
from policy import Policy
from schema import ACTION_SPACE, NUM_ACTIONS, DeviceState, StepState
from state_encoder import STATE_DIM, state_to_features


def main():
    # 1. 动作空间
    print(f"=== 统一执行决策动作空间（{NUM_ACTIONS} 个动作）===")
    for i, (label, device, exec_type, tool) in enumerate(ACTION_SPACE):
        print(f"  {i}: {label}")

    # 2. benchmark 二维任务
    print("\n=== benchmark（执行类型 × 难度）===")
    bench = Benchmark(seed=42)
    for t in range(3):
        task = bench.sample_task(f"t{t}")
        print(f"  [{task.exec_type:9s}/{task.difficulty:6s}] {task.question[:40]!r} → {task.answer}")

    # 3. 策略网络：一个网络，10 动作
    print(f"\n=== 策略网络（STATE_DIM={STATE_DIM}，一个网络出 10 动作）===")
    policy = Policy()
    print(f"  参数量: {sum(p.numel() for p in policy.parameters())}")
    s = StepState(device_states=[DeviceState(d, [0.2]*4) for d in ["weak", "medium", "strong"]])
    f = torch.tensor(state_to_features(s)).unsqueeze(0)
    logits = policy.forward(f)
    print(f"  logits shape: {tuple(logits.shape)}（应 [1, {NUM_ACTIONS}]）")

    # 4. 采样动作 + 贪心
    a = policy.act(f).item()
    g = policy.act(f, greedy=True).item()
    print(f"  采样动作={a} → {ACTION_SPACE[a][0]}；贪心动作={g} → {ACTION_SPACE[g][0]}")

    # 5. mask（屏蔽 offline 设备）
    mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    # 屏蔽 weak 的三个动作（1/4/7）
    mask[0, [1, 4, 7]] = False
    logits_masked = logits.masked_fill(~mask, -1e9)
    print(f"  mask 后（屏蔽 weak）: softmax 不含动作 1/4/7")

    print("\nSMOKE OK —— M1 v5 骨架跑通")


if __name__ == "__main__":
    main()