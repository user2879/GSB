"""End-to-end local acceptance demonstration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .edits import apply_edit
from .engine import OperatorChainEngine, compare_execution_artifacts
from .util import read_json


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_demo(workspace: str | Path = ".opchain-demo") -> dict[str, Any]:
    root = _project_root()
    chain = read_json(root / "samples" / "chain.json")
    input_spec = read_json(root / "samples" / "input.json")
    config = read_json(root / "samples" / "config.json")
    workspace_root = Path(workspace).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="demo-", dir=workspace_root))

    incremental_workspace = run_root / "incremental"
    engine = OperatorChainEngine(chain, input_spec, config, incremental_workspace)
    initial = engine.run()

    edited_chain, edited_config = apply_edit(
        chain,
        config,
        {"kind": "set_params", "node_id": "gelu1", "params": {"approximation": "tanh"}},
    )
    engine = OperatorChainEngine(edited_chain, input_spec, edited_config, incremental_workspace)
    after_param = engine.run()
    full_workspace = run_root / "full-param"
    full_after_param = OperatorChainEngine(edited_chain, input_spec, edited_config, full_workspace).run()
    param_comparison = compare_execution_artifacts(after_param, full_after_param)

    precision_chain, precision_config = apply_edit(
        edited_chain,
        edited_config,
        {"kind": "set_precision", "precision_tier": "fp32"},
    )
    engine = OperatorChainEngine(precision_chain, input_spec, precision_config, incremental_workspace)
    after_precision = engine.run()
    full_after_precision = OperatorChainEngine(
        precision_chain,
        input_spec,
        precision_config,
        run_root / "full-precision",
    ).run()
    precision_comparison = compare_execution_artifacts(after_precision, full_after_precision)

    interrupt_config = {**precision_config, "snapshot_interval": 1}
    partial_engine = OperatorChainEngine(
        precision_chain,
        input_spec,
        interrupt_config,
        run_root / "budget",
    )
    partial = partial_engine.run(stop_after=2)
    resumed_engine = OperatorChainEngine(
        precision_chain,
        input_spec,
        interrupt_config,
        run_root / "budget",
    )
    resumed = resumed_engine.run(resume=True, strict_resume=True)
    budget_run = OperatorChainEngine(
        precision_chain,
        input_spec,
        {**precision_config, "recompute_budget": 2, "snapshot_interval": 1},
        run_root / "limited-budget",
    ).run()

    fused = initial
    plain_config = dict(config)
    plain_config["fusion_strategy"] = "none"
    plain = OperatorChainEngine(chain, input_spec, plain_config, run_root / "plain-fusion").run()
    difference = 0.0
    for left, right in zip(fused["nodes"], plain["nodes"]):
        if left["id"] == "perm1":
            import numpy as np

            with np.errstate(invalid="ignore"):
                difference = float(np.nanmax(np.abs(np.asarray(left["values"]) - np.asarray(right["values"]))))

    ok = (
        initial["status"] == "complete"
        and param_comparison["equal"]
        and precision_comparison["equal"]
        and after_precision["status"] == "complete"
        and difference > 0.0
        and any(item.get("code") == "E_RECOMPUTE_BUDGET_PENDING" for item in budget_run.get("diagnostics", []))
        and resumed["status"] == "complete"
    )
    return {
        "ok": ok,
        "workspace": str(run_root),
        "initial": {
            "status": initial["status"],
            "invalidated": initial["invalidated"],
            "canonical_signature": initial["canonical_signature"],
        },
        "parameter_edit": {
            "invalidated": after_param["invalidated"],
            "reused": after_param["reused"],
            "recomputed": after_param["recomputed"],
            "full_equality": param_comparison,
        },
        "precision_switch": {
            "invalidated": after_precision["invalidated"],
            "full_equality": precision_comparison,
        },
        "fusion_difference_perm1_max_abs": difference,
        "budget_resume": {
            "partial_status": partial["status"],
            "pending": partial["pending"],
            "interrupted_after_stop_hook": True,
            "resumed_status": resumed["status"],
            "resumed_diagnostics": resumed["diagnostics"],
            "limited_budget_pending": budget_run["pending"],
            "limited_budget_codes": sorted({item.get("code") for item in budget_run["diagnostics"]}),
        },
    }
