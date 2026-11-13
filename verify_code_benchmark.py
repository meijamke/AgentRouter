# -*- coding: utf-8 -*-
"""代码生成弱强分化验证：写函数 + 执行测试用例，看 weak 是否真有 bug、strong 是否正确。"""
import sys
sys.path.insert(0, "src")

from src.device_env import execute_device

# (难度, 题目, 函数名, 测试用例 [(args, expected)])
TASKS = [
    ("medium", "写 Python 函数 sum_digits(n)，返回 n 的各位数字之和。只输出函数定义（不要示例调用）。", "sum_digits",
     [((123,), 6), ((999,), 27)]),
    ("medium", "写 Python 函数 is_palindrome(s)，判断字符串 s 是否回文。只输出函数定义。", "is_palindrome",
     [(("aba",), True), (("abc",), False)]),
    ("hard", "写 Python 函数 count_primes(n)，返回 1 到 n（含）中质数的个数。只输出函数定义。", "count_primes",
     [((10,), 4), ((100,), 25)]),
    ("hard", "写 Python 函数 fib(n)，返回第 n 个斐波那契数（fib(0)=0，fib(1)=1）。只输出函数定义。", "fib",
     [((10,), 55), ((20,), 6765)]),
]


def check_code(code: str, fn_name: str, cases) -> bool:
    """受限 exec + 跑测试用例。"""
    ns = {}
    try:
        exec(code, {"__builtins__": {"range": range, "len": len, "int": int, "str": str}}, ns)
        fn = ns.get(fn_name)
        if fn is None:
            return False
        for args, expected in cases:
            if fn(*args) != expected:
                return False
        return True
    except Exception:
        return False


print(f"代码生成验证 {len(TASKS)} 题 × 3 设备\n", flush=True)
for difficulty, q, fn_name, cases in TASKS:
    row = {}
    for tier in ["weak", "medium", "strong"]:
        try:
            out, cost, _ = execute_device(tier, [{"role": "user", "content": q}], mock=False)
            # 提取代码块：去 markdown 围栏
            code = out.strip()
            for marker in ("```python", "```"):
                code = code.replace(marker, "")
            row[tier] = check_code(code, fn_name, cases)
            print(f"  [{difficulty:6s}] {fn_name} {tier}->{'✓' if row[tier] else '✗'}  ({out[:50]!r})", flush=True)
        except Exception as e:
            row[tier] = False
            print(f"  [{difficulty:6s}] {fn_name} {tier} FAILED {str(e)[:60]}", flush=True)
    diff = sum(1 for t in ["weak", "medium", "strong"] if row.get(t))
    print(f"  => {fn_name}: 正确设备数={diff}（weak={row.get('weak')} medium={row.get('medium')} strong={row.get('strong')}）\n", flush=True)