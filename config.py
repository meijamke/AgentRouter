"""统一配置：LLM API 三档设备。"""
import os

API_KEY = os.environ.get("LLM_API_KEY", "")
API_BASE = os.environ.get("LLM_API_BASE", "http://113.46.219.251/v1")


# 三档设备：能力档 cap_embed（4 维：文本/代码/数学/多模态）+ 模型名 + 成本系数
TIERS = {
    "weak":   {"model": "Qwen3.8-Flash",   "cap_embed": [0.2, 0.2, 0.2, 0.2], "cost_coef": 1.0},
    "medium": {"model": "GLM-5.3-Flash",   "cap_embed": [0.6, 0.6, 0.6, 0.6], "cost_coef": 3.0},
    "strong": {"model": "DeepSeek-V4-Pro", "cap_embed": [1.0, 1.0, 1.0, 1.0], "cost_coef": 10.0},
}