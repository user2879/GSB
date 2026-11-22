from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowdiag.attribution import attribute_conflicts
from slowdiag.canonical import canonical_json, sha256_canonical
from slowdiag.config import default_config_dict, load_config_data
from slowdiag.errors import E_REPLAY, SlowDiagError
from slowdiag.generator import check_repeatable_generation, generate_all
from slowdiag.merge import merge_evidence
from slowdiag.migration import migrate_conclusions
from slowdiag.pipeline import run_pipeline
from slowdiag.snapshots import create_snapshot, restore_snapshot
from slowdiag.storage import Ledger, read_all_sources


def _query_sha(view: dict, query_id: str) -> str:
    query = next(item for item in view["queries"] if item["query_id"] == query_id)
    return sha256_canonical(query)


def main() -> int:
    result: dict = {"checks": {}, "passed": True}

    def check(name: str, condition: bool, details: object) -> None:
        result["checks"][name] = {"passed": bool(condition), "details": details}
        result["passed"] = result["passed"] and bool(condition)

    with tempfile.TemporaryDirectory(prefix="slowdiag-acceptance-") as temp:
        workspace = Path(temp)
        config_path = workspace / "slow-sources.json"
        raw = default_config_dict()
        config_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        config = load_config_data(raw, workspace)
        generated = generate_all(config)
        repeatability = check_repeatable_generation(config)
        check("repeatable_generation", repeatability["repeatable"], repeatability)

        datasets = read_all_sources(config)
        merged_forward = merge_evidence(config, datasets)
        merged_reverse = merge_evidence(config, list(reversed(datasets)))
        check(
            "merge_order_independent",
            canonical_json(merged_forward) == canonical_json(merged_reverse),
            {"queries": len(merged_forward)},
        )
        migration = migrate_conclusions(config, datasets)
        attribution = attribute_conflicts(config, datasets, migration)
        attribution_repeat = attribute_conflicts(
            config, list(reversed(datasets)), migration
        )
        check(
            "attribution_repeatable",
            canonical_json(attribution) == canonical_json(attribution_repeat),
            attribution["counts"],
        )

        ledger = Ledger(workspace)
        first = run_pipeline(config, ledger)
        old_q7 = _query_sha(first.view, "q007")
        old_q8 = _query_sha(first.view, "q008")
        check(
            "plan_version_initial",
            first.view["plan_version"] == 1,
            {"plan_version": first.view["plan_version"]},
        )

        changed_raw = copy.deepcopy(raw)
        for source in changed_raw["sources"]:
            if source["id"] == "slow-canary":
                source["sample_period_ms"] = 1200
        config_path.write_text(
            json.dumps(changed_raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changed_config = load_config_data(changed_raw, workspace)
        generate_all(changed_config)
        second = run_pipeline(changed_config, ledger)
        invalidation = second.view["invalidation"]
        new_q7 = _query_sha(second.view, "q007")
        new_q8 = _query_sha(second.view, "q008")
        check(
            "invalidation_plan_monotonic",
            second.view["plan_version"] == 2,
            {"previous": 1, "current": second.view["plan_version"]},
        )
        check(
            "invalidation_scope",
            invalidation["affected_queries"]
            == ["q001", "q002", "q003", "q004", "q005", "q006"],
            {
                "changed_sources": invalidation["changed_sources"],
                "affected_queries": invalidation["affected_queries"],
            },
        )
        check(
            "unaffected_bytes_stable",
            invalidation["unaffected_byte_identical"]
            and old_q7 == new_q7
            and old_q8 == new_q8,
            invalidation["unaffected_checks"],
        )

        snapshot_path = create_snapshot(
            workspace, second.view, changed_config, "acceptance-snapshot.json"
        )
        restored = restore_snapshot(ledger, snapshot_path, workspace)
        check(
            "snapshot_restore",
            restored["snapshot_diff_count"] == 0
            and restored["view_sha256"] == second.view_sha256,
            restored,
        )

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        q003 = next(item for item in payload["merged"] if item["query_id"] == "q003")
        q003["metrics"]["scanned_rows"]["adopted"]["value"] += 1
        tampered = workspace / "snapshots" / "tampered.json"
        tampered.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        replay_error = None
        try:
            restore_snapshot(ledger, tampered, workspace)
        except SlowDiagError as exc:
            replay_error = exc
        check(
            "snapshot_replay_detection",
            replay_error is not None
            and replay_error.code == E_REPLAY
            and replay_error.query_id == "q003"
            and replay_error.metric == "scanned_rows"
            and replay_error.field == "adopted.value",
            replay_error.to_dict() if replay_error else None,
        )

        injections: dict[str, dict] = {}
        for kind, expected in (
            ("source", "E_SOURCE"),
            ("format", "E_FORMAT"),
            ("empty", "E_EMPTY"),
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "slowdiag",
                    "run",
                    "--workspace",
                    str(workspace),
                    "--inject",
                    kind,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            response = json.loads(completed.stdout)
            injections[kind] = response
            check(
                f"injection_{kind}",
                completed.returncode != 0 and response.get("business_code") == expected,
                response,
            )

        result["summary"] = {
            "generated_sources": [
                {key: value for key, value in item.items() if key != "path"}
                for item in generated
            ],
            "migration_counts": second.view["migration"]["counts"],
            "attribution_counts": second.view["attribution"]["counts"],
            "slow_queries": [
                item["query_id"]
                for item in second.view["queries"]
                if item["diagnosis"]["slow"]
            ],
            "injection_codes": {
                key: value.get("business_code") for key, value in injections.items()
            },
        }

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
