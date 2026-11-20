from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from synthml.generate import generate_dataset, replace_sample
from synthml.pipeline import RunOptions, run_pipeline
from synthml.util import read_json


STAGE_ORDER = {"config": 0, "load": 1, "feature": 2, "score": 3, "decision": 4, "metric": 5, "state": 6}


def _workspace(root: Path, sample_count: int = 24, seed: int = 909090) -> dict[str, Path]:
    samples = root / "samples"
    config = root / "experiment.yaml"
    plan = root / "plan.json"
    injections = root / "injections.yaml"
    serial = root / "serial.yaml"
    generate_dataset(
        samples,
        config,
        plan,
        sample_count=sample_count,
        seed=seed,
        concurrency=2,
        snapshot_interval=100,
        injection_path=injections,
        serial_config_path=serial,
    )
    return {
        "samples": samples,
        "config": config,
        "plan": plan,
        "injections": injections,
        "serial": serial,
    }


def _run(
    ws: dict[str, Path],
    root: Path,
    *,
    state: str,
    report: str,
    mode: str,
    plan: Path | None = None,
    config: Path | None = None,
    injections: Path | None = None,
    **overrides: Any,
):
    options = RunOptions(
        samples_dir=ws["samples"],
        config_path=config or ws["config"],
        plan_path=plan or ws["plan"],
        state_dir=root / state,
        report_path=root / report,
        mode=mode,
        injections_path=injections,
        **overrides,
    )
    return run_pipeline(options)


def _core(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "tiers": report["tiers"],
        "failures": report["failures"],
        "scope": report["scope"],
        "injection_hits": report["injection_hits"],
    }


def _stage_counts(report: dict[str, Any], stage: str) -> dict[str, int]:
    return {
        status: sum(
            1
            for entry in report["cache_audit"]
            if entry["stage"] == stage and entry["status"] == status
        )
        for status in ("reused", "recomputed", "failed")
    }


class IntegrationTests(unittest.TestCase):
    def test_edit_sequence_matches_blank_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthml-edit-") as temporary:
            root = Path(temporary)
            ws = _workspace(root)
            base = _run(ws, root, state="state_inc", report="base.json", mode="rebuild")
            self.assertEqual(base.exit_code, 0)
            self.assertEqual(len(base.report["tiers"]), 3)

            replacement = root / "replacement"
            replacement_config = root / "replacement.yaml"
            replacement_plan = root / "replacement_plan.json"
            generate_dataset(
                replacement,
                replacement_config,
                replacement_plan,
                sample_count=6,
                seed=777777,
                concurrency=1,
                snapshot_interval=100,
            )
            replace_sample(ws["samples"], "sample_000005.png", replacement / "sample_000005.png")
            replaced = _run(ws, root, state="state_inc", report="replaced.json", mode="incremental")
            self.assertEqual(replaced.exit_code, 0)
            self.assertEqual(_stage_counts(replaced.report, "load")["recomputed"], 1)
            self.assertEqual(_stage_counts(replaced.report, "feature")["recomputed"], 3)
            self.assertGreater(_stage_counts(replaced.report, "score")["recomputed"], 0)

            narrow_plan = json.loads(ws["plan"].read_text(encoding="utf-8"))
            narrow_plan["tiers"][0]["max_samples"] = 6
            narrow_plan["tiers"][0]["max_labels"] = 2
            narrow_plan["tiers"][1]["max_samples"] = 12
            narrow_plan["tiers"][1]["max_labels"] = 4
            narrow_plan["tiers"][2]["max_samples"] = 20
            narrow_path = root / "narrow_plan.json"
            narrow_path.write_text(json.dumps(narrow_plan, indent=2), encoding="utf-8")
            narrowed = _run(
                ws,
                root,
                state="state_inc",
                report="narrow.json",
                mode="incremental",
                plan=narrow_path,
            )
            self.assertEqual(_stage_counts(narrowed.report, "load")["recomputed"], 0)
            self.assertEqual(_stage_counts(narrowed.report, "feature")["recomputed"], 0)
            self.assertGreater(_stage_counts(narrowed.report, "score")["recomputed"], 0)
            narrowed_full = _run(
                ws,
                root,
                state="state_narrow_full",
                report="narrow_full.json",
                mode="rebuild",
                plan=narrow_path,
            )
            self.assertEqual(_core(narrowed.report), _core(narrowed_full.report))

            expanded = _run(ws, root, state="state_inc", report="expanded.json", mode="incremental")
            self.assertEqual(
                sum(_stage_counts(expanded.report, stage)["recomputed"] for stage in ("load", "feature", "score", "decision", "metric")),
                0,
            )

            threshold_report = _run(
                ws,
                root,
                state="state_inc",
                report="threshold.json",
                mode="incremental",
                force_threshold=0.55,
            )
            self.assertEqual(_stage_counts(threshold_report.report, "load")["recomputed"], 0)
            self.assertEqual(_stage_counts(threshold_report.report, "feature")["recomputed"], 0)
            self.assertEqual(_stage_counts(threshold_report.report, "score")["recomputed"], 0)
            self.assertGreater(_stage_counts(threshold_report.report, "decision")["recomputed"], 0)
            self.assertGreater(_stage_counts(threshold_report.report, "metric")["recomputed"], 0)

            hog_report = _run(
                ws,
                root,
                state="state_inc",
                report="hog.json",
                mode="incremental",
                force_feature_method="hog",
            )
            self.assertEqual(_stage_counts(hog_report.report, "load")["recomputed"], 0)
            self.assertGreater(_stage_counts(hog_report.report, "feature")["recomputed"], 0)
            self.assertTrue(all(tier["feature_metadata"]["feature_method"] == "hog" for tier in hog_report.report["tiers"]))

            knn_report = _run(
                ws,
                root,
                state="state_inc",
                report="knn.json",
                mode="incremental",
                force_feature_method="hog",
                force_classifier="knn",
            )
            self.assertEqual(_stage_counts(knn_report.report, "feature")["recomputed"], 0)
            self.assertGreater(_stage_counts(knn_report.report, "score")["recomputed"], 0)

            final_options = {
                "force_feature_method": "hog",
                "force_classifier": "knn",
                "force_threshold": 0.55,
            }
            incremental_final = _run(
                ws,
                root,
                state="state_inc",
                report="final_incremental.json",
                mode="incremental",
                **final_options,
            )
            self.assertEqual(_stage_counts(incremental_final.report, "score")["recomputed"], 0)
            self.assertGreater(_stage_counts(incremental_final.report, "decision")["recomputed"], 0)
            full_final = _run(
                ws,
                root,
                state="state_full_final",
                report="final_full.json",
                mode="rebuild",
                **final_options,
            )
            self.assertEqual(_core(incremental_final.report), _core(full_final.report))

    def test_concurrency_consistency_and_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthml-concurrency-") as temporary:
            root = Path(temporary)
            ws = _workspace(root, sample_count=24, seed=424242)
            concurrent = _run(
                ws, root, state="state_concurrent", report="concurrent.json", mode="rebuild"
            )
            serial = _run(
                ws,
                root,
                state="state_serial",
                report="serial.json",
                mode="rebuild",
                config=ws["serial"],
            )
            self.assertEqual(concurrent.exit_code, 0)
            self.assertEqual(serial.exit_code, 0)
            self.assertEqual(concurrent.report["tiers"], serial.report["tiers"])
            self.assertEqual(concurrent.report["failures"], serial.report["failures"])
            self.assertEqual(concurrent.report["cache_hit_counts"], serial.report["cache_hit_counts"])
            self.assertEqual(
                [entry["cache_key"] for entry in concurrent.report["cache_audit"]],
                [entry["cache_key"] for entry in serial.report["cache_audit"]],
            )
            self.assertEqual(
                [entry["status"] for entry in concurrent.report["cache_audit"]],
                [entry["status"] for entry in serial.report["cache_audit"]],
            )

            snapshot_dir = root / "state_concurrent" / "snapshots"
            in_progress = []
            for path in snapshot_dir.iterdir():
                manifest = read_json(path / "manifest.json")
                if manifest.get("in_progress"):
                    in_progress.append((int(manifest["snapshot_version"]), path.name))
            self.assertTrue(in_progress)
            checkpoint_name = sorted(in_progress)[-1][1]
            (root / "state_concurrent" / "CURRENT").write_text(
                checkpoint_name + "\n", encoding="utf-8"
            )
            resumed = _run(
                ws, root, state="state_concurrent", report="resumed.json", mode="resume"
            )
            self.assertEqual(resumed.exit_code, 0)
            self.assertEqual(_core(resumed.report), _core(concurrent.report))

    def test_injection_isolation_and_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthml-injection-") as temporary:
            root = Path(temporary)
            ws = _workspace(root, sample_count=24, seed=515151)
            injected = _run(
                ws,
                root,
                state="state_injected",
                report="injected.json",
                mode="rebuild",
                injections=ws["injections"],
            )
            self.assertEqual(injected.exit_code, 0)
            failures = injected.report["failures"]
            self.assertGreaterEqual(sum(item["stage"] == "load" for item in failures), 2)
            self.assertGreaterEqual(sum(item["code"] == "E_FEATURE_INVALID" for item in failures), 1)
            self.assertTrue(all(tier["samples"] for tier in injected.report["tiers"]))
            hits = {item["id"]: item for item in injected.report["injection_hits"]}
            self.assertTrue(hits["INJ-001"]["hit"])
            self.assertTrue(hits["INJ-002"]["hit"])
            self.assertTrue(hits["INJ-003"]["hit"])
            expected_order = sorted(
                failures,
                key=lambda item: (
                    item["sample"],
                    STAGE_ORDER[item["stage"]],
                    item.get("tier", ""),
                    item["code"],
                    item["message"],
                ),
            )
            self.assertEqual(failures, expected_order)

            resumed = _run(
                ws,
                root,
                state="state_injected",
                report="injected_resumed.json",
                mode="resume",
                injections=ws["injections"],
            )
            self.assertEqual(resumed.exit_code, 0)
            self.assertEqual(_core(resumed.report), _core(injected.report))

            identity_mismatch = _run(
                ws,
                root,
                state="state_injected",
                report="identity_mismatch.json",
                mode="resume",
            )
            self.assertEqual(identity_mismatch.exit_code, 3)
            self.assertEqual(identity_mismatch.report["error"]["code"], "E_CONFIG_MISMATCH")

            config_mismatch = _run(
                ws,
                root,
                state="state_injected",
                report="config_mismatch.json",
                mode="resume",
                config=ws["serial"],
                injections=ws["injections"],
            )
            self.assertEqual(config_mismatch.exit_code, 3)
            self.assertEqual(config_mismatch.report["error"]["code"], "E_CONFIG_MISMATCH")


if __name__ == "__main__":
    unittest.main()
