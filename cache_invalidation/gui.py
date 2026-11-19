"""Tkinter 诊断界面。

渲染数据先由纯函数 build_view_model 生成并稳定排序，因此在没有显示
环境的 CI/容器里仍可对界面内容做单元测试；只有真正创建窗口时才导入
tkinter，Windows 本地执行 ``python -m cache_invalidation.gui`` 即可。
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine
from .errors import BusinessError


@dataclass(frozen=True)
class ViewModel:
    inheritance_rows: tuple[tuple[str, str], ...]
    edge_rows: tuple[tuple[str, str], ...]
    round_rows: tuple[tuple[str, str, str, str, str], ...]
    version_rows: tuple[tuple[str, int, str], ...]
    format_version: int
    status_code: str


def build_view_model(engine: Engine,
                     status_code: str = "OK") -> ViewModel:
    inheritance_rows: list[tuple[str, str]] = []
    for name in sorted(engine.graph.namespaces):
        parents = engine.graph.namespace_parents[name]
        inheritance_rows.append(
            (name, " -> ".join(parents) if parents else "<root>"))

    edge_rows: list[tuple[str, str]] = []
    for source in engine.graph.entry_ids_sorted():
        for target in sorted(engine.graph.annotation_edges[source]):
            edge_rows.append((source.to_str(), target.to_str()))

    round_rows: list[tuple[str, str, str, str, str]] = []
    for record in engine.records:
        triggers = sorted(
            record.explicit_triggers + record.expired_triggers
            + record.refresh_triggers)
        round_rows.append((
            str(record.round_no),
            record.trigger_reason,
            ",".join(triggers),
            ",".join(record.affected),
            ",".join(
                f"{key}={value}"
                for key, value in sorted(
                    record.annotation_versions.items())),
        ))

    version_rows: list[tuple[str, int, str]] = []
    for entry_id in engine.graph.entry_ids_sorted():
        version_rows.append((
            entry_id.to_str(),
            engine.versions[entry_id],
            "alive" if entry_id in engine.alive else "invalidated",
        ))

    return ViewModel(
        inheritance_rows=tuple(inheritance_rows),
        edge_rows=tuple(edge_rows),
        round_rows=tuple(round_rows),
        version_rows=tuple(version_rows),
        format_version=engine.format_version,
        status_code=status_code,
    )


def launch(engine: Engine, status_code: str = "OK") -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境缺少 tkinter；Windows 官方安装包默认包含，"
            "Linux 需安装 python3-tk。核心功能与测试不依赖该模块。"
        ) from exc

    model = build_view_model(engine, status_code)
    root = tk.Tk()
    root.title("缓存失效传播诊断")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    def add_table(title: str, columns: tuple[str, ...],
                  rows: tuple[tuple, ...]) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="w", width=180)
        for row in rows:
            tree.insert("", "end", values=row)
        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                  command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    add_table("命名空间继承链", ("namespace", "parents"),
              model.inheritance_rows)
    add_table("注释依赖边", ("source", "downstream"), model.edge_rows)
    add_table("传播轮次",
              ("round", "reason", "triggers", "affected", "versions"),
              model.round_rows)
    add_table("注释版本", ("entry", "annotation_version", "state"),
              model.version_rows)

    status = tk.Label(
        root,
        text=(f"snapshot_format=v{model.format_version} "
              f"business_code={model.status_code}"),
        anchor="w",
    )
    status.pack(fill="x", padx=8, pady=(0, 8))
    root.mainloop()


def describe_error(error: BusinessError) -> str:
    return f"{error.code}: {error.message}"


def main() -> int:
    from pathlib import Path

    from .loader import load_declarations

    root = Path(__file__).resolve().parent.parent
    graph = load_declarations(
        root / "samples" / "namespaces.json",
        root / "samples" / "entries.json")
    engine = Engine(graph)
    launch(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
