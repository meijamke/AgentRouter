"""AppWorld benchmark 适配器（代码式环境）。

任务 = 自然语言目标（如"找出我 Spotify 播放列表里最受欢迎的歌名"），
agent 每步生成 Python 代码（调用 App 的 API），world.execute(code) 执行并返回输出，
最后 world.evaluate().success 判成功。
"""
from __future__ import annotations

import torch

from .device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


class AppworldEnv:
    """封装 AppWorld 单任务环境。"""

    def __init__(self, task_id: str):
        from appworld import AppWorld
        self.task_id = task_id
        self.world = AppWorld(task_id)
        self.instruction = self.world.task.instruction
        self.app_descriptions = str(self.world.task.app_descriptions)
        self.api_docs = str(self.world.task.api_docs)[:3000]   # 截断

    def prompt(self, history: str = "") -> str:
        return (
            f"任务：{self.instruction}\n"
            f"可用 App 说明：{self.app_descriptions}\n"
            f"API 文档（节选）：{self.api_docs}\n"
            f"{history}"
            "请生成一段 Python 代码，调用上述 API 完成任务。只输出代码，不要解释："
        )

    def execute(self, code: str) -> str:
        return self.world.execute(code)

    def success(self) -> bool:
        try:
            return bool(self.world.evaluate().success)
        except Exception:
            return False

    def close(self):
        try:
            self.world.close()
        except Exception:
            pass


def run_episode(policy: Policy, env: AppworldEnv, devices, max_steps: int = 5,
                mock: bool = False, greedy: bool = True):
    """在 AppWorld 一个任务上跑一条带路由的轨迹，返回 (success, cost, n_steps)。"""
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))
    cost = 0.0
    history = ""

    for step_idx in range(max_steps):
        s = StepState(
            trajectory_summary=[], progress=[step_idx / max_steps, 0.5],
            last_output=history[-200:], p_fail=0.0, cost_accum=cost,
            device_states=devices, step_idx=step_idx,
        )
        f = state_to_features(s).unsqueeze(0)
        pf = torch.tensor([[s.p_fail]])
        a, logits, _ = policy.act(f, dev_emb, mask, pf, greedy=greedy)

        if a.item() == 0:   # stop：策略认为可以评估了
            break

        tier = TIERS[a.item() - 1]
        code, c, _ = execute_device(tier, [{"role": "user", "content": env.prompt(history)}], mock=mock)
        cost += c
        out = env.execute(code.strip())
        history += f"\n[代码]\n{code}\n[输出]\n{out}\n"

    return env.success(), cost, step_idx + 1


def run_episode_grad(policy: Policy, env: AppworldEnv, devices, max_steps: int = 5,
                     mock: bool = False):
    """带梯度的 rollout（RL 用）：返回 (log_probs, actions, success, cost, n_steps)。"""
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))
    cost = 0.0
    history = ""
    log_probs, actions = [], []

    for step_idx in range(max_steps):
        s = StepState(
            trajectory_summary=[], progress=[step_idx / max_steps, 0.5],
            last_output=history[-200:], p_fail=0.0, cost_accum=cost,
            device_states=devices, step_idx=step_idx,
        )
        f = state_to_features(s).unsqueeze(0)
        pf = torch.tensor([[s.p_fail]])
        logits, _ = policy.forward(f, dev_emb, mask, pf)   # 带梯度
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        actions.append(a.item())

        if a.item() == 0:   # stop
            break
        tier = TIERS[a.item() - 1]
        code, c, _ = execute_device(tier, [{"role": "user", "content": env.prompt(history)}], mock=mock)
        cost += c
        out = env.execute(code.strip())
        history += f"\n[代码]\n{code}\n[输出]\n{out}\n"

    return log_probs, actions, env.success(), cost, step_idx + 1


if __name__ == "__main__":
    from appworld import load_task_ids
    tid = load_task_ids("train")[0]
    env = AppworldEnv(tid)
    print("指令:", env.instruction)
    # 手动一步：用 strong 设备生成代码并执行
    code, cost, _ = execute_device("strong", [{"role": "user", "content": env.prompt()}], mock=False)
    print("生成代码:", code[:200])
    out = env.execute(code.strip())
    print("执行输出:", str(out)[:200])
    print("success:", env.success())
    env.close()