from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .util import to_builtin


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    text = json.dumps(
        to_builtin(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(path, text + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GEMM/BMM precision observability report")
    lines.append("")
    lines.append(f"Schema: `{report['schema_version']}`")
    lines.append("")
    lines.append("## Injection evidence")
    lines.append("")
    lines.append("| Injection | Kind | Expected | Actual | Hit | Stage |")
    lines.append("|---|---|---|---|---|---|")
    for item in report.get("injections", []):
        lines.append(
            f"| {item['injection_id']} | {item['kind']} | {item['expected_code']} | "
            f"{item['actual_code']} | {item['hit']} | {item['stage']} |"
        )
    lines.append("")
    lines.append("## Comparison audit tables")
    for comparison in report.get("comparison_tables", []):
        lines.append("")
        lines.append(f"### {comparison['sample']}")
        lines.append("")
        lines.append("| Block | Order | Max abs | Max rel | Mean abs | Exceed | Rounding steps | Digest |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in comparison.get("audit_table", []):
            block = row["block_shape"]
            block_label = f"BM{block['BM']}/BN{block['BN']}/BK{block['BK']}"
            steps = ",".join(
                f"{key}:{value}"
                for key, value in row.get("rounding_step_count", {}).items()
            )
            lines.append(
                f"| {block_label} | {row['loop_order']} | "
                f"{row['max_absolute_error']:.6e} | {row['max_relative_error']:.6e} | "
                f"{row['mean_absolute_error']:.6e} | {row['exceed_count']} | "
                f"{steps} | `{row['audit_digest'][:12]}` |"
            )
        lines.append("")
        conclusions = comparison.get("conclusions", {})
        lines.append(f"- Block effect: {conclusions.get('block_size_effect', 'n/a')}")
        lines.append(f"- Loop effect: {conclusions.get('loop_order_effect', 'n/a')}")
        lines.append(f"- Attribution: {conclusions.get('dominant_error_source', 'n/a')}")
    lines.append("")
    lines.append("## Layer exceedances")
    lines.append("")
    exceedances = report.get("layer_exceedance_summary", [])
    if exceedances:
        lines.append("| Sample | Axis | Layer | Ratio | Count | Block | Order |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for item in exceedances:
            block = item["block_shape"]
            lines.append(
                f"| {item['sample']} | {item['axis']} | {item['layer_key']} | "
                f"{item['exceed_ratio']:.3f} | {item['exceed_count']} | "
                f"BM{block['BM']}/BN{block['BN']}/BK{block['BK']} | {item['loop_order']} |"
            )
    else:
        lines.append("No layer exceeded the configured ratio limit.")
    lines.append("")
    lines.append("## Concurrency consistency")
    lines.append("")
    checks = report.get("concurrency_checks", [])
    lines.append(f"- Checks recorded: {len(checks)}")
    lines.append(f"- All equal: {all(item['equal'] for item in checks) if checks else False}")
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    failures = report.get("failures", [])
    if failures:
        lines.append("| Case | Sample | Code | Stage | Shape | Batch |")
        lines.append("|---|---|---|---|---|---|")
        for item in failures:
            lines.append(
                f"| {item.get('case_id')} | {item.get('sample')} | {item.get('business_code')} | "
                f"{item.get('stage')} | `{item.get('shape')}` | {item.get('batch_index')} |"
            )
    else:
        lines.append("No failures recorded.")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    write_json_atomic(target, report)
    markdown_path = target.with_suffix(".md")
    atomic_write_text(markdown_path, render_markdown(report))
    return target
