"""端到端验收演示：生成样本 -> 双求解 -> 增量失效闭包 -> 一致性核对 -> 报告。

运行：
    .venv\\Scripts\\python.exe scripts\\run_demo.py --workspace artifacts/persist --reports artifacts/reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactical_pathbench.budget import Budget
from tactical_pathbench.encoding import windows_path, write_text_file
from tactical_pathbench.persistence import PersistenceStore
from tactical_pathbench.workbench import build_workbench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--workspace", default="artifacts/persist")
    parser.add_argument("--reports", default="artifacts/reports")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", type=int, default=8)
    args = parser.parse_args()

    workspace = ROOT / args.workspace
    reports = ROOT / args.reports
    store = PersistenceStore(workspace)

    wb = build_workbench(seed=args.seed, budget=Budget(thread_budget=args.threads, memory_budget=args.memory))
    initial_routes = wb.solve_batch()
    initial_dual = {q.key(): wb.solve_dual(q) for q in wb.sample.queries}

    interior = boundary = None
    for cell in wb.grid._cell_region:
        if wb.grid.is_boundary_cell(cell):
            boundary = cell
        else:
            interior = cell
        if interior and boundary:
            break

    changes = []
    changes.append(wb.modify_terrain(interior, 7).to_payload())
    routes_after_first = wb.solve_batch()
    consistency_first = wb.consistency_check().to_payload()

    changes.append(wb.modify_terrain(boundary, 9).to_payload())
    routes_after_second = wb.solve_batch()
    consistency_second = wb.consistency_check().to_payload()

    wb.move_unit(wb.sample.units[0].unit_id, wb.sample.queries[1].target)
    routes_after_move = wb.solve_batch()
    consistency_move = wb.consistency_check().to_payload()

    manifest_hash = wb.persist(store)
    exit_hash = store.write_exit_report(
        {
            "status": "clean",
            "map_version": wb.map_version,
            "manifest_hash": manifest_hash,
            "path_count": len(wb.path_hash_report()),
        }
    )

    def route_rows(routes):
        return {
            key: {
                "path_hash": route.path_hash(),
                "cost": route.cost,
                "expanded": route.expanded,
                "solver": route.solver,
                "cache_source": wb.reuse_sources.get(key, route.cache_source),
                "within_budget": route.within_budget,
            }
            for key, route in sorted(routes.items())
        }

    dual_report = {
        key: {
            "astar": route_pair["astar"].to_payload(),
            "flowfield": route_pair["flowfield"].to_payload(),
            "cost_equal": route_pair["astar"].cost == route_pair["flowfield"].cost,
            "sequence_equal": tuple(route_pair["astar"].cells)
            == tuple(route_pair["flowfield"].cells),
        }
        for key, route_pair in sorted(initial_dual.items())
    }

    report = {
        "seed": args.seed,
        "windows_workspace": windows_path(workspace),
        "windows_reports": windows_path(reports),
        "map_content_hash": wb.grid.content_hash(),
        "region_partition_signature": wb.grid.region_partition_signature(),
        "region_checksums": wb.grid.region_checksum_report(),
        "dual_solver": dual_report,
        "closure_history": changes,
        "consistency": {
            "after_first_change": consistency_first,
            "after_second_change": consistency_second,
            "after_unit_move": consistency_move,
        },
        "path_hashes": {
            "initial": route_rows(initial_routes),
            "after_first_change": route_rows(routes_after_first),
            "after_second_change": route_rows(routes_after_second),
            "after_unit_move": route_rows(routes_after_move),
        },
        "scheduler": wb.scheduler.report(),
        "persistence": {
            "manifest_path": windows_path(store.manifest_path),
            "manifest_hash": manifest_hash,
            "exit_report_hash": exit_hash,
            "journal_tail": store._read_journal_tail(),
        },
    }
    write_text_file(reports / "acceptance-report.json", json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))

    closure_lines = ["sequence\tkind\tcell\tinvalid_flows\tinvalid_jumps\tinvalid_paths\treused_flows"]
    for record in changes:
        closure = record["closure"]
        closure_lines.append(
            "\t".join(
                [
                    str(record["sequence"]),
                    record["kind"],
                    str(record["cell"]),
                    str(closure["invalidated_flow_regions"]),
                    str(len(closure["invalidated_jump_portals"])),
                    str(len(closure["invalidated_path_queries"])),
                    str(len(closure["reused_flows"])),
                ]
            )
        )
    write_text_file(reports / "closure-report.tsv", "\n".join(closure_lines) + "\n")

    all_consistent = all(
        item["consistent"]
        for item in (consistency_first, consistency_second, consistency_move)
    )
    print(json.dumps({
        "ok": all_consistent,
        "report": windows_path(reports / "acceptance-report.json"),
        "closure": windows_path(reports / "closure-report.tsv"),
        "map_hash": wb.grid.content_hash(),
    }, ensure_ascii=False, indent=2))
    return 0 if all_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
