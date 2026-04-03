#!/usr/bin/env python3
"""
加载 Coze 项目环境变量（与 d:\\下载\\load_env.py 一致）
通过 coze_workload_identity.Client 获取项目环境变量并输出 export 语句。

云端 / Linux: eval $(python scripts/load_env.py)
Windows PowerShell: 需自行解析输出或使用 setx（建议仍在 WSL/Linux 任务里 eval）

依赖：coze_workload_identity（仅 Coze 运行环境或自行安装）
"""
import os
import sys

workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
app_dir = os.path.join(workspace_path, "src")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from coze_workload_identity import Client

    client = Client()
    env_vars = client.get_project_env_vars()
    client.close()

    for env_var in env_vars:
        value = env_var.value.replace("'", "'\\''")
        print(f"export {env_var.key}='{value}'")

    print(f"# Successfully loaded {len(env_vars)} environment variables", file=sys.stderr)

except Exception as e:
    print(f"# Error loading environment variables: {e}", file=sys.stderr)
    sys.exit(1)
