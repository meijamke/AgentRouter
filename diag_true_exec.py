"""真执行诊断：3 任务 × 3 设备，各自独立子进程跑 AppWorld 真执行，看弱/中/强真实 success-cost 分化。"""
import json
import os
import subprocess
import sys

os.chdir(r"D:\dsh_space\跨设备模型路由代码")
sys.path.insert(0, os.getcwd())
from appworld import load_task_ids

tasks = load_task_ids("train")[:3]
print(f"诊断 {len(tasks)} 任务 × 3 设备（真执行）", flush=True)
for task in tasks:
    row = {}
    for tier in ["weak", "medium", "strong"]:
        out = f"diag_{task}_{tier}.json"
        try:
            subprocess.run([sys.executable, "diag_worker.py", task, tier, out],
                           check=True, timeout=600)
            r = json.load(open(out, encoding="utf-8"))
            row[tier] = (r["success"], r["cost"], r["lat"])
            print(f"  {task} [{tier}] success={r['success']} cost={r['cost']:.4f} lat={r['lat']/1000:.0f}s", flush=True)
        except Exception as e:
            print(f"  {task} [{tier}] FAILED {str(e)[:80]}", flush=True)
            row[tier] = (False, 0, 0)
    # 最便宜够用设备
    cheapest = next((t for t in ["weak", "medium", "strong"] if row[t][0]), "无")
    print(f"  => {task} 最便宜够用设备 = {cheapest}", flush=True)