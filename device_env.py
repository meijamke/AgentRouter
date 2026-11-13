"""三档设备 + 真实 API 执行。"""
from __future__ import annotations

from openai import OpenAI

from config import API_BASE, API_KEY, TIERS

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    return _client


def model_of(tier: str) -> str:
    return TIERS[tier]["model"]


def cap_embed_of(tier: str) -> list:
    return TIERS[tier]["cap_embed"]


def cost_of(tier: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    """成本 = 能力档系数 × tokens（本机 kv=0；云最贵由 cost_coef 体现）。"""
    return TIERS[tier]["cost_coef"] * (prompt_tokens + completion_tokens)


def execute_device(tier: str, messages: list, mock: bool = False):
    """调用一台设备，返回 (output, cost, latency_ms)。"""
    if mock:
        return f"[{tier}] mock", cost_of(tier, prompt_tokens=20, completion_tokens=10), 0.0
    r = _get_client().chat.completions.create(
        model=model_of(tier), messages=messages, temperature=0,
    )
    out = r.choices[0].message.content or ""
    usage = r.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
    cost = cost_of(tier, usage.prompt_tokens, usage.completion_tokens)
    return out, cost, 0.0