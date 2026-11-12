"""ALFWorld benchmark 适配器：把交互式 agent 任务接进路由循环。

ALFWorld 任务 = 目标导向的文本环境（"把杯子放到桌子上"），每步需模型从
admissible_commands 里选一个动作，环境执行并返回新 observation。
成功 = 目标达成（done & success），质量信号 = task success（非每步对错）。
"""
from __future__ import annotations

import torch

from .device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


class AlfworldEnv:
    """封装 ALFWorld 原始环境（懒加载，安装后才真正 import）。"""

    def __init__(self, train_eval: str = "train"):
        self._env = None
        self.train_eval = train_eval

    def _load(self):
        if self._env is None:
            import alfworld.agents.environment as environment
            import alfworld.agents.modules.generic as generic
            cfg = generic.load_config()
            env_type = cfg["env"]["type"]
            env = getattr(environment, env_type)(cfg, train_eval=self.train_eval)
            self._env = env.init_env(batch_size=1)
        return self._env

    def reset(self):
        env = self._load()
        obs, info = env.reset()
        text = obs[0] if isinstance(obs, (list, tuple)) else obs
        # 可执行动作列表（可能在 info 或 obs 里）
        admissible = info.get("admissible_commands", []) if isinstance(info, dict) else []
        return text, admissible

    def step(self, action: str):
        env = self._load()
        out = env.step([action])
        obs, scores, dones, infos = out[0], out[1], out[2], out[3]
        text = obs[0] if isinstance(obs, (list, tuple)) else obs
        done = bool(dones[0]) if not isinstance(dones, (bool, int)) else bool(dones)
        success = bool(infos[0].get("won", False)) if isinstance(infos[0], dict) else bool(done)
        return text, done, success


def _action_prompt(obs: str, admissible: list) -> str:
    acts = ", ".join(admissible) if admissible else "(free-form)"
    return (
        "你是一个家用环境中的智能体。\n"
        f"当前观察：{obs}\n"
        f"可执行动作：{acts}\n"
        "只输出上述可执行动作列表中的一个动作（不要解释、不要多余文字）："
    )


def run_episode(policy: Policy, env: AlfworldEnv, devices, max_steps: int = 50,
                mock: bool = False, greedy: bool = False):
    """在 ALFWorld 一个任务上跑一条带路由的轨迹，返回 (success, cost, n_steps)。"""
    obs, admissible = env.reset()
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))
    cost = 0.0
    n_steps = 0

    for step_idx in range(max_steps):
        s = StepState(
            trajectory_summary=[], progress=[step_idx / max_steps, 0.5], last_output=obs,
            p_fail=0.0, cost_accum=cost, device_states=devices, step_idx=step_idx,
        )
        f = state_to_features(s).unsqueeze(0)
        pf = torch.tensor([[s.p_fail]])
        a, logits, _ = policy.act(f, dev_emb, mask, pf, greedy=greedy)

        if a.item() == 0:   # stop：策略放弃该任务
            break

        tier = TIERS[a.item() - 1]
        prompt = _action_prompt(obs, admissible)
        action, c, _ = execute_device(tier, [{"role": "user", "content": prompt}], mock=mock)
        cost += c
        n_steps += 1

        obs, done, success = env.step(action.strip())
        admissible = []
        if done:
            return success, cost, n_steps
    return False, cost, n_steps


if __name__ == "__main__":
    # 手动冒烟：先构造策略再跑一个任务（需 ALFWorld 已安装 + 数据已下载）
    torch.manual_seed(0)
    policy = Policy()
    devices = make_device_pool()
    env = AlfworldEnv(train_eval="train")
    succ, cost, steps = run_episode(policy, env, devices, max_steps=10, mock=True)
    print(f"episode: success={succ} cost={cost:.3f} steps={steps}")