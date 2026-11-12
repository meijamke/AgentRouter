"""真执行诊断 worker（子进程）：单设备单次代码执行 → 真 success。"""
import json
import os
import sys

os.chdir(r"D:\dsh_space\跨设备模型路由代码")
sys.path.insert(0, os.getcwd())


def main():
    task_id, tier, out = sys.argv[1], sys.argv[2], sys.argv[3]
    from src.device_env import execute_device
    from appworld import AppWorld
    world = AppWorld(task_id)
    prompt = (
        f"任务：{world.task.instruction}\n"
        f"可用 App：{str(world.task.app_descriptions)}\n"
        f"API 文档（节选）：{str(world.task.api_docs)[:3000]}\n"
        "请生成 Python 代码完成任务，只输出代码："
    )
    code, cost, lat = execute_device(tier, [{"role": "user", "content": prompt}], mock=False)
    try:
        world.execute(code.strip())
        success = bool(world.evaluate().success)
    except Exception:
        success = False
    world.close()
    json.dump({"task": task_id, "tier": tier, "success": success,
               "cost": cost, "lat": lat, "code_head": code[:80]},
              open(out, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()