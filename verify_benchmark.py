# -*- coding: utf-8 -*-
"""调硬 benchmark 弱强分化验证：8 题 × 3 设备，看 weak 是否真错、strong 是否真对。"""
import sys
sys.path.insert(0, "src")

from src.device_env import execute_device

# (难度, 问题, 标准答案)
TASKS = [
    ("easy",   "37 + 25 = ?（只输出数字）", "62"),
    ("easy",   "6 × 8 = ?（只输出数字）", "48"),
    ("medium", "846 × 73 = ?（只输出数字）", "61758"),
    ("medium", "(123 + 456) × 7 = ?（只输出数字）", "4053"),
    ("hard",   "9876 × 5432 = ?（只输出数字）", "53646432"),
    ("hard",   "计算 2^16 + 3^6 = ?（只输出数字）", "66265"),
    ("hard",   "一个数的平方是 144，另一个数的立方是 343，这两个数的和是多少？（只输出数字）", "19"),
    ("hard",   "写出 Python 代码：统计 1 到 1000 中有多少个质数。只输出代码。", "168"),
]

print(f"验证 {len(TASKS)} 题 × 3 设备（真实 API）\n", flush=True)
for difficulty, q, ans in TASKS:
    row = {}
    for tier in ["weak", "medium", "strong"]:
        try:
            out, cost, _ = execute_device(tier, [{"role": "user", "content": q}], mock=False)
            # 对代码题：提取 out 里的数字/执行判断；这里先按"答案子串"粗判
            text = out.strip()
            correct = (ans in text) or (text.replace(" ", "") == ans)
            row[tier] = correct
            print(f"  [{difficulty:6s}] weak={row.get('weak')} med={row.get('medium')} str={row.get('strong')} "
                  f"答={ans} | {tier}->{text[:40]!r}", flush=True)
        except Exception as e:
            row[tier] = False
            print(f"  [{difficulty:6s}] {tier} FAILED {str(e)[:60]}", flush=True)
    diff = sum(1 for t in ["weak", "medium", "strong"] if row.get(t))
    print(f"  => 正确设备数={diff}（weak={row.get('weak')} medium={row.get('medium')} strong={row.get('strong')}）\n", flush=True)