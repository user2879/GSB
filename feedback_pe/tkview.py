from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

try:  # Import is testable on WSL once libtk/python3-tk are available.
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:  # pragma: no cover - environment dependent
    tk = None
    ttk = None
    messagebox = None

from .commands import apply_command
from .config import FRAC_BITS_CHOICES, QUANT_TIER_CHOICES
from .engine import Workspace
from .reporting import build_report
from .runner import run_batch


def view_model(report: dict[str, Any], scheme: str | None = None) -> dict[str, Any]:
    business = report["business"]
    schemes = [item["name"] for item in business["config"]["schemes"]]
    selected = scheme or business["config"].get("active_scheme") or schemes[0]

    layer_rows = []
    for layer, values in business["scheme_layer_stats"].get(selected, {}).items():
        layer_rows.append({
            "layer": layer,
            "sample_count": values["sample_count"],
            "point_count": values["point_count"],
            "score_mean": values["score_mean"],
            "weight_mean": values["weight_mean"],
            "max_score_deviation": values["max_score_deviation"],
            "max_weight_deviation": values["max_weight_deviation"],
        })

    error_rows = []
    for item in business["stage_results"]:
        if item["stage"] != "error" or item["scheme"] != selected or not item["available"]:
            continue
        trace = item["payload"]["trace"]
        for stage_name, stage in trace["stages"].items():
            error_rows.append({
                "sample_id": trace["sample_id"],
                "stage": stage_name,
                "exceeded": stage["exceeded"],
                "max_relative_error": stage["max_relative_error"],
                "max_absolute_error": stage["max_absolute_error"],
                "absolute_zero_reference_count": stage["absolute_zero_reference_count"],
                "first_metric": (
                    stage["first_exceedance"]["metric"]
                    if stage["first_exceedance"] else None
                ),
            })

    boundary_rows = [
        item for item in business["boundary_samples"] if item["scheme"] == selected
    ]
    ledger_rows = []
    latest = report.get("execution", {}).get("latest_run") or {}
    for item in latest.get("invalidations", []):
        ledger_rows.append({
            "kind": "invalidate",
            "sort_key": item["sort_key"],
            "detail": ",".join(item.get("basis", [])),
        })
    for item in latest.get("reuse", []):
        ledger_rows.append({
            "kind": "reuse",
            "sort_key": item["sort_key"],
            "detail": item["cache_key"][:12],
        })
    for item in latest.get("recompute", []):
        ledger_rows.append({
            "kind": "recompute",
            "sort_key": item["sort_key"],
            "detail": item["cache_key"][:12],
        })

    return {
        "schemes": schemes,
        "selected_scheme": selected,
        "layer_rows": layer_rows,
        "error_rows": error_rows,
        "boundary_rows": boundary_rows,
        "ledger_rows": ledger_rows,
        "code_counts": business["business_code_counts"],
        "scheduler": report.get("execution", {}).get("scheduler", {}),
    }


class DiagnosticView:
    def __init__(self, report: dict[str, Any],
                 on_edit: Callable[[str, float, int, str], dict[str, Any]] | None = None):
        if tk is None:
            raise RuntimeError("Tkinter is unavailable; install python3-tk and Tk runtime libraries")
        self.report = report
        self.on_edit = on_edit
        self.root = tk.Tk()
        self.root.title("PE fixed-point incremental diagnostics")
        self.root.geometry("1180x760")
        self._build_controls()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.trees: dict[str, ttk.Treeview] = {}
        self._build_tabs()
        self.status = tk.StringVar(value="ready")
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=4)
        self.refresh(report)

    def _build_controls(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Incremental edit")
        frame.pack(fill="x", padx=8, pady=8)
        model = view_model(self.report)
        self.scheme_var = tk.StringVar(value=model["selected_scheme"])
        self.base_var = tk.StringVar()
        self.frac_var = tk.StringVar(value=str(self.report["business"]["config"]["frac_bits"]))
        self.tier_var = tk.StringVar(value=self.report["business"]["config"]["quant_tier"])

        ttk.Label(frame, text="scheme").grid(row=0, column=0, padx=4, pady=6)
        self.scheme_menu = ttk.OptionMenu(frame, self.scheme_var, model["selected_scheme"],
                                          *model["schemes"], command=lambda _e: self._sync_base())
        self.scheme_menu.grid(row=0, column=1, padx=4)
        ttk.Label(frame, text="base_freq").grid(row=0, column=2, padx=4)
        ttk.Entry(frame, textvariable=self.base_var, width=12).grid(row=0, column=3, padx=4)
        ttk.Label(frame, text="frac_bits").grid(row=0, column=4, padx=4)
        ttk.OptionMenu(frame, self.frac_var, self.frac_var.get(),
                       *[str(x) for x in FRAC_BITS_CHOICES]).grid(row=0, column=5, padx=4)
        ttk.Label(frame, text="quant_tier").grid(row=0, column=6, padx=4)
        ttk.OptionMenu(frame, self.tier_var, self.tier_var.get(),
                       *QUANT_TIER_CHOICES).grid(row=0, column=7, padx=4)
        ttk.Button(frame, text="Apply incremental recompute",
                   command=self._apply).grid(row=0, column=8, padx=8)
        self._sync_base()

    def _sync_base(self) -> None:
        scheme_name = self.scheme_var.get()
        for scheme in self.report["business"]["config"]["schemes"]:
            if scheme["name"] == scheme_name:
                self.base_var.set(str(scheme["base_freq"]))
                break

    def _build_tabs(self) -> None:
        specs = {
            "layers": ("Layer stats", ("layer", "samples", "points", "score_mean",
                                       "weight_mean", "max_score_err", "max_weight_err")),
            "errors": ("Stage errors", ("sample", "stage", "exceeded", "max_rel",
                                        "max_abs", "zero_refs", "first_metric")),
            "boundary": ("Boundary samples", ("sample", "scheme", "distance_margin",
                                              "first_stage", "flips", "reasons")),
            "ledger": ("Invalidate/reuse", ("kind", "sort_key", "detail")),
            "codes": ("Business codes", ("code", "count")),
        }
        for key, (title, columns) in specs.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            tree = ttk.Treeview(frame, columns=columns, show="headings")
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=180 if column in {"sort_key", "reasons"} else 110,
                            anchor="w")
            yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=yscroll.set)
            tree.pack(side="left", fill="both", expand=True)
            yscroll.pack(side="right", fill="y")
            self.trees[key] = tree

    def refresh(self, report: dict[str, Any]) -> None:
        self.report = report
        model = view_model(report, self.scheme_var.get() if hasattr(self, "scheme_var") else None)
        for tree in self.trees.values():
            for item in tree.get_children():
                tree.delete(item)
        for row in model["layer_rows"]:
            self.trees["layers"].insert("", "end", values=(
                row["layer"], row["sample_count"], row["point_count"],
                _fmt(row["score_mean"]), _fmt(row["weight_mean"]),
                _fmt(row["max_score_deviation"]), _fmt(row["max_weight_deviation"]),
            ))
        for row in model["error_rows"]:
            self.trees["errors"].insert("", "end", values=(
                row["sample_id"], row["stage"], row["exceeded"],
                _fmt(row["max_relative_error"]), _fmt(row["max_absolute_error"]),
                row["absolute_zero_reference_count"], _fmt(row["first_metric"]),
            ))
        for row in model["boundary_rows"]:
            self.trees["boundary"].insert("", "end", values=(
                row["sample_id"], row["scheme"], row["distance_boundary_margin"],
                row["first_exceeded_stage"], row["flip_count"],
                ";".join(row["reasons"]),
            ))
        for row in model["ledger_rows"]:
            self.trees["ledger"].insert("", "end", values=(
                row["kind"], row["sort_key"], row["detail"],
            ))
        for code, count in sorted(model["code_counts"].items()):
            self.trees["codes"].insert("", "end", values=(code, count))
        scheduler = model["scheduler"].get("last_batch") or {}
        self.status.set(
            f"version={report.get('execution', {}).get('global_version')} "
            f"completed={scheduler.get('completed_count')} "
            f"pending={scheduler.get('pending_count')} "
            f"recovery={report.get('execution', {}).get('recovery')}"
        )

    def _apply(self) -> None:
        if self.on_edit is None:
            messagebox.showinfo("No controller", "This view is display-only.")
            return
        try:
            report = self.on_edit(
                self.scheme_var.get(),
                float(self.base_var.get()),
                int(self.frac_var.get()),
                self.tier_var.get(),
            )
            self.refresh(report)
        except Exception as exc:
            messagebox.showerror("Incremental edit failed", str(exc))

    def run(self) -> None:  # pragma: no cover - manual UI loop
        self.root.mainloop()

    def smoke(self) -> None:
        self.root.update_idletasks()
        self.root.update()
        self.root.destroy()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)



class TkController:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root)
        state = json.loads((self.root / "state" / "ui_state.json").read_text("utf-8"))
        self.config = state["config"]
        self.samples = {item["id"]: item for item in state["samples"]}
        self.failures = state["failures"]
        self.workspace = Workspace(self.root / state["workspace"])
        self.report_path = self.root / state["report"]

    def initial_report(self) -> dict[str, Any]:
        return json.loads(self.report_path.read_text("utf-8"))

    def edit(self, scheme: str, base_freq: float, frac_bits: int,
             quant_tier: str) -> dict[str, Any]:
        apply_command(self.config, self.samples, {
            "op": "set_base_freq", "scheme": scheme, "base_freq": base_freq,
        })
        apply_command(self.config, self.samples, {
            "op": "set_frac_bits", "frac_bits": frac_bits,
        })
        apply_command(self.config, self.samples, {
            "op": "set_quant_tier", "quant_tier": quant_tier,
        })
        summary = run_batch(
            self.workspace, self.config, list(self.samples.values()), self.failures,
            batch_budget=10**9, sample_quota=10**9,
        )
        report = build_report(
            self.config, list(self.samples.values()), self.failures,
            self.workspace, summary,
        )
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report


def main() -> None:  # pragma: no cover - manual launch
    controller = TkController(Path.cwd())
    view = DiagnosticView(controller.initial_report(), controller.edit)
    view.run()


if __name__ == "__main__":  # pragma: no cover
    main()
