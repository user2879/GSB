from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from affinity_scheduler.model import PLUGIN_NAMES
from affinity_scheduler.samples import sample_spec, write_sample
from affinity_scheduler.util import atomic_write_text


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare_sample(root: Path, *, port_base: int | None = None) -> dict[str, Any]:
    return write_sample(root, port_base=port_base)


def write_spec_variant(root: Path, mutate: Any, *, port_base: int | None = None) -> Path:
    value = sample_spec(1, port_base)
    mutate(value)
    path = root / "config" / "spec-variant.json"
    write_json(path, value)
    return path


def fake_result(
    pod_id: str,
    node_name: str,
    *,
    total_score: int = 50,
    parameter_version: int = 1,
    plugin_scores: dict[str, int] | None = None,
    plugin_contributions: dict[str, int] | None = None,
    error_bound: int = 1,
) -> dict[str, Any]:
    scores = plugin_scores or {name: 50 for name in PLUGIN_NAMES}
    contributions = plugin_contributions or {
        name: 5000 for name in PLUGIN_NAMES
    }
    return {
        "schema": "pod-schedule-conclusion-v1",
        "pod_id": pod_id,
        "node_name": node_name,
        "status": "bound" if node_name != "-" else "unbound",
        "veto_code": 0 if node_name != "-" else 50,
        "veto_name": "NONE" if node_name != "-" else "NO_FEASIBLE_NODE",
        "total_fixed": total_score * 1000,
        "total_score": total_score,
        "plugin_scores": scores,
        "plugin_contributions": contributions,
        "error_bound": error_bound,
        "tie_break": [*PLUGIN_NAMES, "node_name"],
        "filter_trace": [],
        "parameter_version": parameter_version,
        "scale_exponent": 3,
        "scale_factor": 1000,
        "receipts": {},
        "conclusion_digest": "pending",
    }


def run_cli(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, "-m", "affinity_scheduler", *args],
        cwd=str(cwd or REPO_ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_stdout_json(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"stdout is not JSON: {completed.stdout!r}; stderr={completed.stderr!r}"
        ) from exc
