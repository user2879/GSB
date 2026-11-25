"""Optional Tkinter review console.

Tk is imported lazily so CLI and tests remain usable on minimal WSL images.
Install the system dependency with ``sudo apt-get install -y python3-tk``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .coordinator import CommandCoordinator
from .engine import IncrementalEngine
from .snapshot import SnapshotStore
from .synth import write_demo_bundle
from .validation import load_package, parse_command_stream


def launch(data_dir: str = "examples/generated", workspace: str = "state/gui") -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise SystemExit(
            "Tkinter 不可用。请在 Ubuntu/WSL 执行：sudo apt-get update && sudo apt-get install -y python3-tk"
        ) from exc

    data_path = Path(data_dir)
    if not (data_path / "corpus.jsonl").exists():
        write_demo_bundle(data_path, RuntimeConfig(corpus_size=16))
    package_result = load_package(data_path / "model_params.json")
    if not package_result["ok"]:
        raise SystemExit(json.dumps(package_result["failures"], ensure_ascii=False, indent=2))

    from .validation import parse_corpus, parse_lexicon

    corpus = parse_corpus(data_path / "corpus.jsonl")
    lexicon = parse_lexicon(data_path / "lexicon.json")
    commands = parse_command_stream(data_path / "command_stream.json")
    store = SnapshotStore(workspace)
    engine, _recovery = store.recover_or_create(package_result["package"], RuntimeConfig(), corpus["samples"], lexicon["entries"])
    engine.run_incremental()
    store.save(engine, phase="gui-start")
    coordinator = CommandCoordinator(engine, store, budget_ms=40)

    root = tk.Tk()
    root.title("NER 复核账本")
    root.geometry("980x680")
    header = ttk.Label(root, text="本地确定性中文 NER 复核", font=("sans-serif", 16, "bold"))
    header.pack(anchor="w", padx=12, pady=8)
    status = tk.StringVar(value="就绪")
    ttk.Label(root, textvariable=status, foreground="#1f4e79").pack(anchor="w", padx=12)

    text = tk.Text(root, wrap="none", font=("Noto Sans Mono CJK SC", 10))
    text.pack(fill="both", expand=True, padx=12, pady=8)

    def render(payload: dict[str, Any]) -> None:
        text.delete("1.0", "end")
        text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

    def show_summary() -> None:
        with coordinator.lock:
            render(engine.summary())

    def poll_latest() -> None:
        final = coordinator.wait_for_latest(timeout=0)
        if final.get("status") != "fresh":
            root.after(120, poll_latest)
            return
        render({"replaced_stale_summary": True, "authoritative": final.get("summary"), "command": final.get("command")})
        status.set(f"最新代次 {engine.generation}；权威账本已替换陈旧摘要")

    def next_command() -> None:
        if engine.command_cursor >= len(commands):
            status.set("命令流已执行完，可重新打开或替换 command_stream.json")
            return
        command = commands[engine.command_cursor]
        response = coordinator.submit(command, budget_ms=1)
        if response.get("status") == "stale":
            summary = response["summary"]
            render(response)
            status.set(f"陈旧摘要：stale={response.get('stale_generation')} target={response.get('target_generation')} code={response.get('code')}")
            root.after(120, poll_latest)
        else:
            render(response)
            status.set(f"最新完成：generation={engine.generation}")

    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", padx=12)
    ttk.Button(toolbar, text="刷新权威摘要", command=show_summary).pack(side="left", padx=4)
    ttk.Button(toolbar, text="执行下一条命令", command=next_command).pack(side="left", padx=4)
    ttk.Button(toolbar, text="退出", command=root.destroy).pack(side="right", padx=4)
    show_summary()

    def on_close() -> None:
        coordinator.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0
