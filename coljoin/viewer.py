"""可选的 Tk 计划/账本只读查看器。

Tk 在部分发行版需额外安装 python3-tk；运行环境缺失时优雅降级为 CLI 文本输出，
不影响规划、执行与账本主链路。
"""

import argparse
import json


def _collect_summary(workspace: str):
    from .pipeline import Pipeline

    pipeline = Pipeline(workspace)
    _, evaluations, decision = pipeline.plan()
    return {
        "candidate_count": len(evaluations),
        "chosen_plan_id": decision["chosen_plan_id"],
        "error_influenced": decision["error_influenced"],
        "ranking_top5": decision["ranking"][:5],
    }


def run_cli(workspace: str):
    print(json.dumps(_collect_summary(workspace), ensure_ascii=False, indent=2))


def run_tk(workspace: str):
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - 环境相关
        print(f"Tk 不可用，降级为 CLI 输出（原因: {exc}）")
        run_cli(workspace)
        return
    summary = _collect_summary(workspace)
    root = tk.Tk()
    root.title("coljoin plan viewer")
    text = tk.Text(root, width=80, height=24)
    text.pack()
    text.insert(tk.END, json.dumps(summary, ensure_ascii=False, indent=2))
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--cli", action="store_true")
    args = parser.parse_args()
    if args.cli:
        run_cli(args.workspace)
    else:
        run_tk(args.workspace)


if __name__ == "__main__":
    main()
