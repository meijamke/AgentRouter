"""冒烟测试：mock 模式验证全链路（不调真实 API）。

运行：python smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from src.policy import Policy
from src.train_il import train_il
from src.train_rl import train_rl
from src.device_env import make_device_pool, SyntheticBenchmark
from src.state_encoder import state_to_features, STATE_DIM


def main():
    torch.manual_seed(0)

    # 1. 设备池 + 状态特征维度
    devices = make_device_pool()
    assert len(devices) == 3, "应有 3 档设备"
    print(f"设备池: {[d.device_id for d in devices]}")
    print(f"STATE_DIM = {STATE_DIM}")

    from src.schema import StepState
    s = StepState(
        trajectory_summary=[], progress=[0.0, 0.5], last_output="",
        p_fail=0.0, cost_accum=0.0, device_states=devices, step_idx=0,
    )
    f = state_to_features(s)
    assert f.shape == (STATE_DIM,), f"特征维度错误: {f.shape}"
    print(f"状态特征 shape: {f.shape}")

    # 2. 策略前向 + 参数量
    policy = Policy()
    n_params = sum(x.numel() for x in policy.parameters())
    print(f"Policy 参数量: {n_params}")

    import torch as _t
    dev_emb = _t.tensor([[0.2]*4, [0.6]*4, [1.0]*4], dtype=_t.float32).unsqueeze(0)
    mask = _t.ones(1, 3)
    pf = _t.tensor([[0.0]])
    logits, value = policy(f.unsqueeze(0), dev_emb, mask, pf)
    assert logits.shape == (1, 4), f"logits shape 错误: {logits.shape}"
    print(f"前向 logits shape: {logits.shape}, value: {value.item():.3f}")

    # 3. IL 冷启动（mock）
    print("\n=== IL 冷启动 ===")
    train_il(policy, epochs=2, mock=True)

    # 4. cost-aware RL（mock）
    print("\n=== cost-aware RL ===")
    train_rl(policy, epochs=3, mock=True)

    print("\nSMOKE OK —— 全链路（设备/编码/策略/IL/RL）跑通")


if __name__ == "__main__":
    main()