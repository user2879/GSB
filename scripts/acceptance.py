"""One-command acceptance run with persisted evidence under artifacts/."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pageagg.canonical import sha256_bytes, write_json_atomic, write_text_atomic
from pageagg.catalog import canonical_catalog_text
from pageagg.models import ColumnDef
from pageagg.page_format import encode_page
from pageagg.sample import generate_sample
from pageagg.ui import Dashboard, load_dashboard_data


class Acceptance:
    def __init__(self) -> None:
        self.base = ROOT / "artifacts" / "acceptance"
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True)
        self.raw_dir = self.base / "raw"
        self.raw_dir.mkdir()
        self.log_path = self.base / "logs" / "business.jsonl"
        self.checks: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {name}: {detail}")
        if not passed:
            raise AssertionError(f"acceptance check failed: {name}")

    def run(
        self,
        name: str,
        *args: str,
        expected_exit: int = 0,
        parse_json: bool = True,
    ) -> dict[str, Any] | None:
        command = [sys.executable, "-m", "pageagg", *args]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        record = {
            "name": name,
            "command": command,
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        self.commands.append(record)
        write_text_atomic(
            self.raw_dir / f"{name}.stdout.json",
            process.stdout if process.stdout else "{}\n",
        )
        if process.returncode != expected_exit:
            raise AssertionError(
                f"{name} exited {process.returncode}, expected {expected_exit}: "
                f"{process.stdout} {process.stderr}"
            )
        if parse_json and process.stdout.strip():
            return json.loads(process.stdout)
        return None

    def write_evolution_dataset(self, root: Path, groups_by_page: dict[int, list[int]], values: dict[int, list[int]]) -> Path:
        rows = {len(group_list) for group_list in groups_by_page.values()}
        assert len(rows) == 1
        rows_per_page = rows.pop()
        columns = [ColumnDef("group_id", "int"), ColumnDef("value_int", "int")]
        pages_dir = root / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for page_id in sorted(groups_by_page):
            group_values = np.asarray(groups_by_page[page_id], dtype="<i8")
            values_array = np.asarray(values[page_id], dtype="<i8")
            null_maps = {
                "group_id": np.zeros(rows_per_page, dtype=np.uint8),
                "value_int": np.zeros(rows_per_page, dtype=np.uint8),
            }
            payload = encode_page(
                page_id,
                rows_per_page,
                columns,
                {"group_id": group_values, "value_int": values_array},
                null_maps,
            )
            relative = f"pages/page_{page_id:04d}.cpg"
            (root / relative).write_bytes(payload)
            entries.append(
                {
                    "page_id": page_id,
                    "file": relative,
                    "rows": rows_per_page,
                    "content_sha256": sha256_bytes(payload),
                }
            )
        catalog = {
            "format_version": 1,
            "dataset_id": "acceptance-evolution-v1",
            "rows_per_page": rows_per_page,
            "group_key": "group_id",
            "batch_size": 2,
            "overflow_ratio_threshold_ppm": 400000,
            "rows_reference": rows_per_page,
            "boundary_margin_ppm": 50000,
            "columns": [column.payload() for column in columns],
            "scan_sequence": sorted(groups_by_page),
            "pages": entries,
        }
        catalog_path = root / "catalog.json"
        write_text_atomic(catalog_path, canonical_catalog_text(catalog))
        return catalog_path

    def execute(self) -> dict[str, Any]:
        environment = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sqlite3": __import__("sqlite3").sqlite_version,
            "tkinter": tk.TkVersion,
        }
        self.check("environment_imports", True, environment)

        sample = generate_sample(self.base / "sample", force=True)
        self.check(
            "deterministic_sample_generated",
            Path(sample["catalog_path"]).exists(),
            {
                "catalog_digest": sample["catalog_digest"],
                "page_count": sample["page_count"],
                "access_count": sample["access_count"],
                "scan_sequence": sample["scan_sequence"],
            },
        )

        compare_dir = self.base / "route-compare-files"
        compare = self.run(
            "compare_routes",
            "compare-routes",
            "--catalog",
            sample["catalog_path"],
            "--budget",
            "4",
            "--out-dir",
            str(compare_dir),
        )
        assert compare is not None
        absolute_equal = (
            (compare_dir / "resident.result.json").read_bytes()
            == (compare_dir / "rescan.result.json").read_bytes()
        )
        eviction_equal = (
            (compare_dir / "eviction.events.json").read_bytes()
            == (compare_dir / "eviction.events.from-rescan.json").read_bytes()
        )
        self.check(
            "dual_route_result_and_eviction_bytes",
            compare["equivalence"]["result_bytes_equal"]
            and compare["equivalence"]["eviction_bytes_equal"]
            and absolute_equal
            and eviction_equal,
            {
                "result_digest": compare["equivalence"]["result_digest"],
                "result_bytes_sha256": compare["equivalence"]["result_bytes_sha256"],
                "eviction_digest": compare["equivalence"]["eviction_digest"],
                "resident_cache_hits": compare["resident"]["cache_hits"],
                "rescan_cache_hits": compare["rescan"]["cache_hits"],
                "raw_file_bytes_equal": absolute_equal,
                "raw_eviction_bytes_equal": eviction_equal,
            },
        )
        self.check(
            "route_boundary_decision",
            compare["auto_route"]["boundary"]["affected"] is True
            and compare["auto_route"]["selected_route"] == "RESIDENT",
            compare["auto_route"],
        )

        resident_db = self.base / "resident.db"
        rescan_db = self.base / "rescan.db"
        resident = self.run(
            "aggregate_resident",
            "aggregate",
            "--catalog",
            sample["catalog_path"],
            "--db",
            str(resident_db),
            "--budget",
            "4",
            "--route",
            "resident",
        )
        rescan = self.run(
            "aggregate_rescan",
            "aggregate",
            "--catalog",
            sample["catalog_path"],
            "--db",
            str(rescan_db),
            "--budget",
            "2",
            "--route",
            "rescan",
        )
        assert resident is not None and rescan is not None
        self.check(
            "persisted_dual_route_estimate_equivalence",
            resident["execution"]["result_digest"] == rescan["execution"]["result_digest"]
            and resident["execution"]["result_bytes_sha256"]
            == rescan["execution"]["result_bytes_sha256"],
            {
                "resident_result_digest": resident["execution"]["result_digest"],
                "rescan_result_digest": rescan["execution"]["result_digest"],
                "resident_cache_hits": resident["execution"]["cache_hits"],
                "rescan_cache_hits": rescan["execution"]["cache_hits"],
                "resident_evictions": resident["execution"]["eviction_count"],
                "rescan_events": rescan["execution"]["rescan_event_count"],
            },
        )

        replay = self.run("replay_resident", "replay", "--db", str(resident_db))
        validation = self.run("validate_resident", "validate", "--db", str(resident_db))
        rescan_validation = self.run("validate_rescan", "validate", "--db", str(rescan_db))
        assert replay is not None and validation is not None and rescan_validation is not None
        self.check(
            "replay_and_validator",
            replay["ok"] and validation["ok"] and rescan_validation["ok"],
            {
                "replay_checked_batches": replay["checked_batches"],
                "replay_checked_outcomes": replay["checked_page_outcomes"],
                "validator_checks": [item["name"] for item in validation["checks"]],
            },
        )

        evolve_db = self.base / "evolve.db"
        self.run(
            "aggregate_for_evolve",
            "aggregate",
            "--catalog",
            sample["catalog_path"],
            "--db",
            str(evolve_db),
            "--budget",
            "4",
        )
        evolved = self.run(
            "evolve_budget",
            "evolve",
            "--db",
            str(evolve_db),
            "--catalog",
            sample["catalog_path"],
            "--budget",
            "2",
        )
        evolve_validation = self.run("validate_evolved", "validate", "--db", str(evolve_db))
        assert evolved is not None and evolve_validation is not None
        self.check(
            "snapshot_version_and_whole_replacement",
            evolved["snapshot_action"]["plan_version"] == 2
            and evolved["snapshot_action"]["previous_plan_version"] == 1
            and evolved["snapshot_action"]["replaced"] is True
            and evolve_validation["ok"] is True,
            {
                "snapshot_action": evolved["snapshot_action"],
                "ledger_plan_version_counter": evolved["ledger"]["plan_version_counter"],
            },
        )
        self.check(
            "unaffected_group_bytes_preserved",
            evolved["propagation"]["unaffected_groups_byte_identical"]
            and all(
                item["old_digest"] == item["new_digest"]
                for item in evolved["propagation"]["unaffected_group_digests"]
            ),
            {
                "reasons": evolved["propagation"]["reasons"],
                "affected_pages": evolved["propagation"]["affected_pages"],
                "route_affected_groups": evolved["propagation"]["route_affected_groups"],
                "unaffected_groups": evolved["propagation"]["unaffected_groups"],
                "unaffected_group_digests": evolved["propagation"]["unaffected_group_digests"],
            },
        )

        base_catalog = self.write_evolution_dataset(
            self.base / "evolution-base",
            {1: [1, 1, 2, 2], 2: [3, 3, 4, 4]},
            {1: [0, 1, 2, 3], 2: [0, 1, 2, 3]},
        )
        changed_catalog = self.write_evolution_dataset(
            self.base / "evolution-changed",
            {1: [1, 1, 1, 5], 2: [3, 3, 4, 4]},
            {1: [10, 20, 30, 40], 2: [0, 1, 2, 3]},
        )
        group_db = self.base / "groups.db"
        self.run(
            "aggregate_group_baseline",
            "aggregate",
            "--catalog",
            str(base_catalog),
            "--db",
            str(group_db),
            "--budget",
            "1",
        )
        group_evolution = self.run(
            "evolve_group_membership",
            "evolve",
            "--db",
            str(group_db),
            "--catalog",
            str(changed_catalog),
            "--budget",
            "1",
        )
        assert group_evolution is not None
        propagation = group_evolution["propagation"]
        self.check(
            "group_view_replacement_diff",
            propagation["added_groups"] == [5]
            and propagation["removed_groups"] == [2]
            and propagation["changed_groups"] == [1]
            and propagation["unaffected_groups"] == [3, 4],
            {
                "added_groups": propagation["added_groups"],
                "removed_groups": propagation["removed_groups"],
                "changed_groups": propagation["changed_groups"],
                "unaffected_groups": propagation["unaffected_groups"],
                "unaffected_group_digests": propagation["unaffected_group_digests"],
                "propagation_steps": propagation["propagation_steps"],
            },
        )

        crash_db = self.base / "crash.db"
        crash_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pageagg",
                "aggregate",
                "--catalog",
                sample["catalog_path"],
                "--db",
                str(crash_db),
                "--budget",
                "4",
                "--crash-during-batch",
                "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.commands.append(
            {
                "name": "forced_termination",
                "exit_code": crash_process.returncode,
                "stdout": crash_process.stdout,
                "stderr": crash_process.stderr,
            }
        )
        recovery = self.run("recover_after_forced_exit", "recover", "--db", str(crash_db))
        resumed = self.run(
            "resume_after_forced_exit",
            "aggregate",
            "--catalog",
            sample["catalog_path"],
            "--db",
            str(crash_db),
            "--budget",
            "4",
        )
        crash_validation = self.run("validate_after_resume", "validate", "--db", str(crash_db))
        assert recovery is not None and resumed is not None and crash_validation is not None
        self.check(
            "forced_exit_recovery",
            crash_process.returncode == 137
            and recovery["forced_exit_report_available"] is False
            and recovery["snapshot"]["cursor"] == 3
            and recovery["unknown_batches_require_rerun"] is True
            and resumed["snapshot_action"]["action"] == "resumed"
            and crash_validation["ok"] is True,
            {
                "forced_exit_code": crash_process.returncode,
                "recover_cursor": recovery["snapshot"]["cursor"],
                "unknown_attempts": recovery["unknown_attempts"],
                "forced_exit_report": recovery["forced_exit_report"],
                "resume_verified": resumed["snapshot_action"]["verified"],
            },
        )

        catalog_fault = self.run(
            "inject_catalog_truncate",
            "inject",
            "--catalog",
            sample["catalog_path"],
            "--out",
            str(self.base / "fault-catalog"),
            "--kind",
            "catalog_truncate",
        )
        nullmap_fault = self.run(
            "inject_nullmap_misalign",
            "inject",
            "--catalog",
            sample["catalog_path"],
            "--out",
            str(self.base / "fault-nullmap"),
            "--kind",
            "nullmap_misalign",
        )
        assert catalog_fault is not None and nullmap_fault is not None
        e_page = self.run(
            "error_catalog_truncate",
            "aggregate",
            "--catalog",
            str(self.base / "fault-catalog" / "catalog.json"),
            "--db",
            str(self.base / "fault-catalog.db"),
            "--budget",
            "4",
            "--log-path",
            str(self.log_path),
            expected_exit=10,
        )
        e_nullmap = self.run(
            "error_nullmap_misalign",
            "aggregate",
            "--catalog",
            str(self.base / "fault-nullmap" / "catalog.json"),
            "--db",
            str(self.base / "fault-nullmap.db"),
            "--budget",
            "4",
            "--log-path",
            str(self.log_path),
            expected_exit=11,
        )
        e_budget = self.run(
            "error_zero_budget",
            "aggregate",
            "--catalog",
            sample["catalog_path"],
            "--db",
            str(self.base / "zero-budget.db"),
            "--budget",
            "0",
            "--log-path",
            str(self.log_path),
            expected_exit=12,
        )
        assert e_page is not None and e_nullmap is not None and e_budget is not None
        log_records = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        log_codes = [record["code"] for record in log_records]
        self.check(
            "three_anomaly_business_codes",
            e_page["code"] == "E_PAGE"
            and e_nullmap["code"] == "E_NULLMAP"
            and e_nullmap["category"] == "E_PAGE"
            and e_budget["code"] == "E_BUDGET"
            and {"E_PAGE", "E_NULLMAP", "E_BUDGET"}.issubset(set(log_codes)),
            {
                "catalog_truncate": e_page,
                "nullmap_misalign": e_nullmap,
                "zero_budget": e_budget,
                "structured_log_codes": log_codes,
            },
        )

        dashboard_data = load_dashboard_data(resident_db)
        root = tk.Tk()
        root.withdraw()
        try:
            Dashboard(root, dashboard_data, error_code="NONE")
            root.update_idletasks()
            root.update()
            ui_ok = True
        finally:
            root.destroy()
        self.check(
            "tkinter_dashboard_loads",
            ui_ok,
            {
                "page_outcomes": len(dashboard_data["page_outcomes"]),
                "slot_rows": len(dashboard_data["page_outcomes"]),
                "groups": len(dashboard_data["result"]["groups"]),
                "evictions": len(dashboard_data["evictions"]),
                "ledger_format_version": dashboard_data["ledger"]["format_version"],
            },
        )

        report = {
            "ok": all(check["passed"] for check in self.checks),
            "generated_at_local": __import__("datetime").datetime.now().astimezone().isoformat(),
            "workspace": str(ROOT),
            "environment": environment,
            "checks": self.checks,
            "commands": self.commands,
            "artifacts": {
                "sample": str(self.base / "sample"),
                "resident_db": str(resident_db),
                "rescan_db": str(rescan_db),
                "evolve_db": str(evolve_db),
                "crash_db": str(crash_db),
                "structured_log": str(self.log_path),
                "route_compare_files": str(compare_dir),
            },
        }
        write_json_atomic(self.base / "acceptance_report.json", report)
        write_json_atomic(ROOT / "artifacts" / "acceptance_report.json", report)
        markdown = self.write_markdown(report)
        write_text_atomic(ROOT / "artifacts" / "VALIDATION.md", markdown)
        print(f"ACCEPTANCE_OK checks={len(self.checks)} report={self.base / 'acceptance_report.json'}")
        return report

    def write_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# PageAgg Acceptance Report",
            "",
            f"- Overall: **{'PASS' if report['ok'] else 'FAIL'}**",
            f"- Generated: `{report['generated_at_local']}`",
            f"- Python: `{report['environment']['python']}`",
            f"- numpy: `{report['environment']['numpy']}`",
            f"- SQLite: `{report['environment']['sqlite3']}`",
            f"- Tk: `{report['environment']['tkinter']}`",
            "",
            "## Checks",
            "",
            "| Check | Result | Evidence |",
            "|---|---|---|",
        ]
        for check in report["checks"]:
            evidence = json.dumps(check["detail"], ensure_ascii=False, sort_keys=True)
            if len(evidence) > 500:
                evidence = evidence[:497] + "..."
            lines.append(
                f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {evidence.replace('|', '/')} |"
            )
        lines.extend(
            [
                "",
                "## Key Artifacts",
                "",
            ]
        )
        for name, path in report["artifacts"].items():
            lines.append(f"- `{name}`: `{path}`")
        lines.extend(
            [
                "",
                "The acceptance run used only local deterministic samples. No container runtime or external dataset was used.",
                "",
            ]
        )
        markdown = "\n".join(lines)
        write_text_atomic(self.base / "ACCEPTANCE.md", markdown)
        return markdown


def main() -> int:
    try:
        report = Acceptance().execute()
        return 0 if report["ok"] else 1
    except Exception as error:
        print(f"ACCEPTANCE_FAIL {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
