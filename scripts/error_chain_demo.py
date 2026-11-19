"""覆盖全部业务错误码的结构化诊断链路。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tactical_pathbench.astar import AStarSolver
from tactical_pathbench.budget import Budget, Scheduler
from tactical_pathbench.errors import (
    ErrorCode,
    TacticalError,
)
from tactical_pathbench.encoding import write_text_file
from tactical_pathbench.grid import RegionPartition, TacticalGrid
from tactical_pathbench.workbench import build_workbench


def capture(fn):
    try:
        fn()
    except TacticalError as exc:
        return exc.to_diagnostic()
    return {"code": "NO_ERROR", "message": "expected exception was not raised"}


def main() -> int:
    diagnostics = {}

    def e_map_shape():
        TacticalGrid(np.zeros((2,), dtype=np.uint8), np.ones((2,), bool),
                     np.ones((2,), bool), RegionPartition(1, 1))

    def e_terrain_range():
        wb = build_workbench(seed=2)
        wb.modify_terrain((0, 0), 99)

    def e_dependency_stale():
        wb = build_workbench(seed=2)
        wb.graph.get_node("path:DOES_NOT_EXIST")

    def e_closure_mismatch():
        wb = build_workbench(seed=2)
        cell = next(iter(wb.grid._cell_region))
        closure = wb.graph.tile_closure([cell])
        wb.graph.assert_closure_precision(closure, {123456})

    def e_budget_denied():
        Scheduler(Budget(thread_budget=0))

    def e_unit_invalid():
        wb = build_workbench(seed=2)
        unit = wb.sample.units[0]
        wb.move_unit(unit.unit_id, (-1, -1))

    def e_query_unreachable():
        wb = build_workbench(seed=2)
        target = (0, 0)
        wb.grid.set_passable(target, False)
        source = wb.sample.queries[0].source
        if source == target:
            source = wb.sample.queries[1].source
        AStarSolver(wb.grid).solve(source, target)

    diagnostics[ErrorCode.E_MAP_SHAPE] = capture(e_map_shape)
    diagnostics[ErrorCode.E_TERRAIN_RANGE] = capture(e_terrain_range)
    diagnostics[ErrorCode.E_DEPENDENCY_STALE] = capture(e_dependency_stale)
    diagnostics[ErrorCode.E_CLOSURE_MISMATCH] = capture(e_closure_mismatch)
    diagnostics[ErrorCode.E_BUDGET_DENIED] = capture(e_budget_denied)
    diagnostics[ErrorCode.E_UNIT_INVALID] = capture(e_unit_invalid)
    diagnostics[ErrorCode.E_QUERY_UNREACHABLE] = capture(e_query_unreachable)

    expected = {
        ErrorCode.E_MAP_SHAPE,
        ErrorCode.E_TERRAIN_RANGE,
        ErrorCode.E_DEPENDENCY_STALE,
        ErrorCode.E_CLOSURE_MISMATCH,
        ErrorCode.E_BUDGET_DENIED,
        ErrorCode.E_UNIT_INVALID,
        ErrorCode.E_QUERY_UNREACHABLE,
    }
    observed = set(diagnostics)
    ok = expected == observed and all(item["code"] in expected for item in diagnostics.values())
    payload = {"ok": ok, "diagnostics": diagnostics}
    reports = ROOT / "artifacts" / "reports"
    write_text_file(reports / "error-chain-report.json", json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
