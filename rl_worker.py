"""RL worker（子进程）：加载策略权重，跑一个 AppWorld episode，输出 JSON。

独立子进程 = 绕开 AppWorld"多 env 顺序创建"的安全守卫 bug。
"""
import json
import os
import sys

os.chdir(r"D:\dsh_space\跨设备模型路由代码")
sys.path.insert(0, os.getcwd())

import torch


def main():
    task_id, weights_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    from src.appworld_env import AppworldEnv
    from src.device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
    from src.policy import Policy
    from src.schema import StepState
    from src.state_encoder import state_to_features

    policy = Policy()
    policy.load_state_dict(torch.load(weights_path, weights_only=True))
    policy.eval()
    devices = make_device_pool()
    env = AppworldEnv(task_id)
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))

    feats, actions = [], []
    cost = 0.0
    history = ""
    for step_idx in range(5):
        s = StepState(
            trajectory_summary=[], progress=[step_idx / 5, 0.5], last_output=history[-200:],
            p_fail=0.0, cost_accum=cost, device_states=devices, step_idx=step_idx,
        )
        f = state_to_features(s)
        with torch.no_grad():
            logits, _ = policy(f.unsqueeze(0), dev_emb, mask, torch.tensor([[0.0]]))
            a = logits.argmax().item()          # 贪心
        if a == 0:
            break
        feats.append(f.tolist())                 # 23 维特征
        actions.append(a)
        tier = TIERS[a - 1]
        code, c, _ = execute_device(tier, [{"role": "user", "content": env.prompt(history)}], mock=False)
        cost += c
        out = env.execute(code.strip())
        history += f"\n{out}\n"

    result = {"feats": feats, "actions": actions, "success": env.success(), "cost": cost}
    env.close()
    json.dump(result, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()