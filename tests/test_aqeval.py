from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aqeval.cache import ArtifactCache
from aqeval.config import DIMENSIONS, Config
from aqeval.conversion import convert_batch, convert_one
from aqeval.demo import compare_reports
from aqeval.diagnostics import consistency_report
from aqeval.edits import apply_edit
from aqeval.errors import InterruptedRun
from aqeval.formats import (
    parse_v1,
    read_exchange,
    write_format_bytes,
)
from aqeval.generator import (
    DEFAULT_SAMPLE_SPECS,
    SampleSpec,
    generate_sample_bytes,
    generate_workspace,
)
from aqeval.levels import LevelEntry, LevelMap
from aqeval.models import UnifiedResult
from aqeval.pipeline import EvaluationPipeline
from aqeval.scoring import (
    compute_frame_features,
    score_dimensions,
    summarize_frames,
)


class WorkspaceCase(unittest.TestCase):
    def make_workspace(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "workspace"
        generate_workspace(root)
        return root

    @staticmethod
    def ledger(report: dict) -> dict[tuple[str, str], dict]:
        return {
            (entry["file_id"], entry["stage"]): entry
            for entry in report["cache_ledger"]["entries"]
        }


class GeneratorTests(WorkspaceCase):
    def test_sample_generation_is_byte_deterministic(self) -> None:
        spec = DEFAULT_SAMPLE_SPECS[1]
        first = generate_sample_bytes(spec)
        second = generate_sample_bytes(spec)
        third = generate_sample_bytes(spec)
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(first[0:4], b"RIFF")
        self.assertEqual(first[8:12], b"WAVE")

    def test_generated_workspace_has_all_versions_and_manifest(self) -> None:
        root = self.make_workspace()
        self.assertTrue((root / "eval/v1/results_v1.txt").exists())
        self.assertTrue((root / "eval/v2/results_v2.json").exists())
        self.assertTrue((root / "eval/v3/results_v3.bin").exists())
        manifest = json.loads((root / "samples/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["specs"]), len(DEFAULT_SAMPLE_SPECS))
        for file_id, digest in manifest["sha256"].items():
            import hashlib

            actual = hashlib.sha256((root / "samples" / f"{file_id}.wav").read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


class FormatTests(WorkspaceCase):
    def setUp(self) -> None:
        self.root = self.make_workspace()
        self.level_map = LevelMap.load(self.root / "level_map.json")

    def test_all_three_versions_parse_to_same_ids_and_dimensions(self) -> None:
        outcomes = {
            version: read_exchange(
                version,
                self.root
                / {
                    1: "eval/v1/results_v1.txt",
                    2: "eval/v2/results_v2.json",
                    3: "eval/v3/results_v3.bin",
                }[version],
                self.level_map,
            )
            for version in (1, 2, 3)
        }
        for version, outcome in outcomes.items():
            self.assertEqual(len(outcome.records), 6, msg=f"v{version}")
            self.assertEqual(len(outcome.failures), 0)
            for record in outcome.records:
                self.assertEqual(set(record.dimensions_micro), set(DIMENSIONS))
                self.assertIn(record.level_code, {"0", "1", "2", "3"})
        ids = {version: [record.file_id for record in outcome.records] for version, outcome in outcomes.items()}
        self.assertEqual(ids[1], ids[2])
        self.assertEqual(ids[2], ids[3])
        self.assertTrue(all(record.frame_summaries for record in outcomes[3].records))

    def test_v1_fixed_width_format_and_precision(self) -> None:
        data = (self.root / "eval/v1/results_v1.txt").read_bytes()
        lines = data.splitlines()
        self.assertTrue(lines)
        self.assertTrue(all(len(line) == 46 for line in lines))
        outcome = parse_v1(data, "memory", self.level_map)
        self.assertEqual(outcome.failures, [])
        self.assertTrue(all(record.precision_tag == "v1_permille" for record in outcome.records))

    def test_v2_retains_precision_tag(self) -> None:
        outcome = read_exchange(2, self.root / "eval/v2/results_v2.json", self.level_map)
        self.assertEqual(outcome.failures, [])
        self.assertTrue(outcome.records)
        self.assertTrue(all(record.precision_tag == "reeval_float6" for record in outcome.records))

    def test_malformed_and_duplicate_v1_are_format_failures(self) -> None:
        line = (self.root / "eval/v1/results_v1.txt").read_text(encoding="utf-8").splitlines()[0]
        bad_width = (line[:-1] + "\n").encode("utf-8")
        failure_outcome = parse_v1(bad_width, "bad", self.level_map)
        self.assertEqual(failure_outcome.failures[0].code, "E_EVAL_FORMAT")
        duplicate = (line + "\n" + line + "\n").encode("utf-8")
        duplicate_outcome = parse_v1(duplicate, "dup", self.level_map)
        self.assertTrue(any(item.code == "E_EVAL_FORMAT" for item in duplicate_outcome.failures))

    def test_unmapped_code_reports_business_error(self) -> None:
        line = (self.root / "eval/v1/results_v1.txt").read_text(encoding="utf-8").splitlines()[0]
        line = line[:-2] + "9 "
        outcome = parse_v1((line + "\n").encode("utf-8"), "unmapped", self.level_map)
        self.assertEqual(outcome.failures[0].code, "E_LEVEL_UNMAPPED")

    def test_v3_checksum_detects_corruption(self) -> None:
        data = bytearray((self.root / "eval/v3/results_v3.bin").read_bytes())
        data[-1] ^= 0xFF
        temporary = self.root / "corrupt.bin"
        temporary.write_bytes(data)
        outcome = read_exchange(3, temporary, self.level_map)
        self.assertTrue(outcome.failures)
        self.assertEqual(outcome.failures[0].code, "E_EVAL_FORMAT")


class ConversionTests(WorkspaceCase):
    def setUp(self) -> None:
        self.root = self.make_workspace()
        self.level_map = LevelMap.load(self.root / "level_map.json")
        self.config = Config.load(self.root / "config.json")
        self.v1 = read_exchange(1, self.root / "eval/v1/results_v1.txt", self.level_map).records
        self.v2 = read_exchange(2, self.root / "eval/v2/results_v2.json", self.level_map).records
        self.v3 = read_exchange(3, self.root / "eval/v3/results_v3.bin", self.level_map).records

    def test_v1_to_v2_exact_precision_and_names(self) -> None:
        outcome = convert_batch(self.v1, 2, self.level_map, self.config.fixed_decimal_bits)
        self.assertEqual(outcome.failures, [])
        self.assertTrue(all(record.precision_tag == "permille_to_float6" for record in outcome.records))
        self.assertTrue(all(record.level_name is not None for record in outcome.records))
        for source, converted in zip(
            sorted(self.v1, key=lambda record: record.file_id),
            sorted(outcome.records, key=lambda record: record.file_id),
        ):
            self.assertEqual(source.total_micro, converted.total_micro)
            self.assertEqual(source.dimensions_micro, converted.dimensions_micro)

    def test_v2_to_v1_bypass_contains_residual_and_discarded_name(self) -> None:
        outcome = convert_batch(self.v2, 1, self.level_map, self.config.fixed_decimal_bits)
        self.assertEqual(len(outcome.bypass_entries), 6)
        for entry in outcome.bypass_entries:
            self.assertIn("total_residual_micro", entry)
            self.assertIn("dimension_residual_micro", entry)
            self.assertIn("discarded_level_name", entry)
            self.assertIn("discarded_precision_tag", entry)
        self.assertTrue(any(item.code == "DEG_V1_TRUNCATION" for item in outcome.degradations))

    def test_v1_to_v3_records_missing_frame_summary_degradation(self) -> None:
        outcome = convert_batch(self.v1, 3, self.level_map, self.config.fixed_decimal_bits)
        self.assertEqual(outcome.failures, [])
        self.assertTrue(
            any(item.code == "DEG_FRAME_SUMMARY_UNAVAILABLE" for item in outcome.degradations)
        )
        self.assertTrue(all(not record.frame_summaries for record in outcome.records))

    def test_v3_to_v1_roundtrip_preserves_total_at_permille(self) -> None:
        outcome = convert_batch(self.v3, 1, self.level_map, self.config.fixed_decimal_bits)
        self.assertEqual(len(outcome.records), 6)
        by_id = {record.file_id: record for record in self.v3}
        for record in outcome.records:
            self.assertEqual(record.total_permille, by_id[record.file_id].total_permille)

    def test_v3_to_v2_records_dropped_frame_summaries(self) -> None:
        outcome = convert_batch(self.v3, 2, self.level_map, self.config.fixed_decimal_bits)
        self.assertTrue(
            any(item.code == "DEG_FRAME_SUMMARY_DROPPED" for item in outcome.degradations)
        )
        self.assertTrue(all(not record.frame_summaries for record in outcome.records))
        self.assertTrue(
            all(entry["discarded_frame_summary_count"] > 0 for entry in outcome.bypass_entries)
        )

    def test_low_fixed_bits_emit_loss(self) -> None:
        outcome = convert_batch(self.v2, 3, self.level_map, 2)
        self.assertTrue(any(item.code == "DEG_V3_FIXED_WIDTH" for item in outcome.degradations))
        self.assertGreater(
            sum(item.residual_micro for item in outcome.degradations),
            0,
        )

    def test_missing_level_mapping_is_explicit(self) -> None:
        reduced = LevelMap([LevelEntry("2", "good", 0)])
        outcome = convert_batch(self.v2, 1, reduced, self.config.fixed_decimal_bits)
        self.assertTrue(outcome.failures)
        self.assertTrue(all(item.code == "E_LEVEL_UNMAPPED" for item in outcome.failures))


class ScoringTests(WorkspaceCase):
    def setUp(self) -> None:
        self.root = self.make_workspace()
        self.config = Config.load(self.root / "config.json")
        self.level_map = LevelMap.load(self.root / "level_map.json")

    def _score_sample(
        self, file_id: str, config: Config | None = None
    ) -> tuple[dict, dict]:
        from aqeval.audio import load_audio, frame_ranges

        config = config or self.config
        audio = load_audio(self.root / "samples" / f"{file_id}.wav", file_id, config)
        frames = [
            compute_frame_features(
                file_id,
                index,
                audio.samples[start:end],
                config.sample_rate,
                config.clipping_amplitude_threshold,
            )
            for index, (start, end) in enumerate(
                frame_ranges(
                    audio.sample_count,
                    config.frame_length_samples,
                    config.hop_samples,
                )
            )
        ]
        summary = summarize_frames(file_id, frames, config)
        return summary, score_dimensions(file_id, summary, config)

    def test_scoring_is_deterministic_and_quantized(self) -> None:
        first = self._score_sample("sample_clean")
        second = self._score_sample("sample_clean")
        self.assertEqual(first, second)
        for value in first[1]["dimensions_permille"].values():
            self.assertIn(value, self.config.quantization_levels)

    def test_boundary_sample_hits_exact_continuity_threshold(self) -> None:
        summary, scored = self._score_sample("sample_boundary")
        self.assertEqual(summary["continuity_metric"], 0.05)
        self.assertTrue(
            any(
                item["dimension"] == "continuity"
                and item["kind"] == "raw_band_threshold"
                and item["threshold"] == 0.05
                for item in scored["boundaries"]
            )
        )

    def test_noise_floor_uses_low_energy_frames_only(self) -> None:
        summary, scored = self._score_sample("sample_floor")
        self.assertGreater(summary["noise_floor_frame_count"], 0)
        self.assertLessEqual(
            summary["noise_floor_cutoff_dbfs"],
            summary["rms_median_dbfs"],
        )
        self.assertAlmostEqual(summary["noise_floor_dbfs"], -34.0, delta=3.0)
        self.assertGreater(scored["metrics"]["noise"], -40.0)

    def test_continuity_uses_configured_zero_run_threshold(self) -> None:
        summary, scored = self._score_sample("sample_dropout")
        self.assertEqual(summary["zero_run_threshold"], 20)
        self.assertGreater(summary["dropout_frame_count"], 0)
        self.assertGreater(summary["continuity_metric"], 0.0)
        raised = self.config.with_updates(consecutive_zero_samples=100)
        raised_summary, raised_scored = self._score_sample("sample_dropout", raised)
        self.assertEqual(raised_summary["dropout_frame_count"], 0)
        self.assertEqual(raised_summary["continuity_metric"], 0.0)
        self.assertEqual(raised_scored["dimensions_micro"]["continuity"], 1000_000)
        self.assertGreater(scored["metrics"]["continuity"], raised_scored["metrics"]["continuity"])

    def test_level_flip_causes_distinguish_permille_and_fixed_width(self) -> None:
        level_map = LevelMap(
            [LevelEntry("0", "poor", 0), LevelEntry("2", "good", 900)]
        )
        record = UnifiedResult(
            file_id="edge",
            source_version=0,
            total_micro=899_600,
            dimensions_micro={name: 899_600 for name in DIMENSIONS},
            level_code="0",
            level_name="poor",
            precision_tag="reeval_float6",
        )
        report = consistency_report({"edge": record}, level_map, 2)
        by_pair = {
            (item["left_version"], item["right_version"]): item
            for item in report["level_flips"]
        }
        self.assertEqual(by_pair[(1, 2)]["cause"], "permille_truncation")
        self.assertEqual(by_pair[(2, 3)]["cause"], "fixed_width")

    def test_level_mapping_missing_is_classified_in_consistency(self) -> None:
        level_map = LevelMap([LevelEntry("0", "poor", 0)])
        record = UnifiedResult(
            file_id="edge",
            source_version=0,
            total_micro=700_000,
            dimensions_micro={name: 700_000 for name in DIMENSIONS},
            level_code="9",
            level_name=None,
            precision_tag="reeval_float6",
        )
        report = consistency_report({"edge": record}, level_map, 5)
        self.assertEqual(len(report["level_flips"]), 2)
        self.assertTrue(
            all(item["cause"] == "level_mapping_missing" for item in report["level_flips"])
        )
        self.assertTrue(all(item["evidence"]["failure_codes"] for item in report["level_flips"]))


class PipelineTests(WorkspaceCase):
    def setUp(self) -> None:
        self.root = self.make_workspace()

    def test_initial_and_noop_runs_reuse_all_artifacts(self) -> None:
        first = EvaluationPipeline(self.root).run()
        self.assertEqual(first["summary"]["canonical_count"], 6)
        self.assertEqual(first["failures"], [])
        self.assertTrue(first["snapshot"]["complete"])
        second = EvaluationPipeline(self.root).run()
        self.assertEqual(second["cache_ledger"]["reuse_count"], 24)
        self.assertEqual(second["cache_ledger"]["recompute_count"], 0)
        self.assertEqual(first["outputs"]["sha256"], second["outputs"]["sha256"])

    def test_weight_change_only_recomputes_totals_and_serialization(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "set_weight", "dimension": "noise", "value": 0.4})
        report = EvaluationPipeline(self.root).run()
        ledger = self.ledger(report)
        for file_id in {key[0] for key in ledger}:
            self.assertEqual(ledger[(file_id, "features")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "dimensions")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "total")]["status"], "recompute")
            self.assertEqual(ledger[(file_id, "serialization")]["status"], "recompute")
        self.assertTrue(report["snapshot"]["stale"])
        self.assertEqual(report["snapshot"]["stale_code"], "E_SNAPSHOT_STALE")

    def test_audio_replacement_only_invalidates_one_file(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(
            self.root,
            {
                "op": "replace_audio",
                "file_id": "sample_snr6",
                "source": "replacements/sample_snr6.wav",
            },
        )
        report = EvaluationPipeline(self.root).run()
        ledger = self.ledger(report)
        for file_id in ("sample_clean", "sample_clip", "sample_dropout", "sample_floor", "sample_boundary"):
            self.assertEqual(ledger[(file_id, "features")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "dimensions")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "serialization")]["status"], "reuse")
        self.assertEqual(ledger[("sample_snr6", "features")]["status"], "recompute")
        self.assertEqual(ledger[("sample_snr6", "dimensions")]["status"], "recompute")
        self.assertEqual(ledger[("sample_snr6", "serialization")]["status"], "recompute")

    def test_serial_verification_does_not_recompute_reused_files(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(
            self.root,
            {
                "op": "replace_audio",
                "file_id": "sample_snr6",
                "source": "replacements/sample_snr6.wav",
            },
        )
        feature_calls: list[str] = []
        dimension_calls: list[str] = []

        def tracked_features(file_id, *args, **kwargs):
            feature_calls.append(file_id)
            return compute_frame_features(file_id, *args, **kwargs)

        def tracked_dimensions(file_id, *args, **kwargs):
            dimension_calls.append(file_id)
            return score_dimensions(file_id, *args, **kwargs)

        with patch("aqeval.pipeline.compute_frame_features", side_effect=tracked_features), patch(
            "aqeval.pipeline.score_dimensions", side_effect=tracked_dimensions
        ):
            report = EvaluationPipeline(self.root).run()
        self.assertEqual(set(feature_calls), {"sample_snr6"})
        self.assertEqual(set(dimension_calls), {"sample_snr6"})
        self.assertEqual(report["concurrency"]["feature_verification_mode"], "serial_recompute")
        self.assertEqual(report["concurrency"]["score_verification_mode"], "serial_recompute")

    def test_clipping_threshold_invalidates_feature_summary_and_downstream(self) -> None:
        initial = EvaluationPipeline(self.root).run()
        before = {
            item["file_id"]: item["dimensions_micro"]["clipping"]
            for item in initial["canonical_results"]
        }
        apply_edit(
            self.root,
            {"op": "set_threshold", "key": "clipping_amplitude_threshold", "value": 0.1},
        )
        report = EvaluationPipeline(self.root).run()
        ledger = self.ledger(report)
        for file_id in {key[0] for key in ledger}:
            self.assertEqual(ledger[(file_id, "features")]["status"], "recompute")
            self.assertEqual(ledger[(file_id, "dimensions")]["status"], "recompute")
            self.assertEqual(ledger[(file_id, "total")]["status"], "recompute")
        after = {
            item["file_id"]: item["dimensions_micro"]["clipping"]
            for item in report["canonical_results"]
        }
        self.assertLess(after["sample_clip"], before["sample_clip"])

    def test_version_switch_only_invalidates_serialization(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "switch_version", "value": 3})
        report = EvaluationPipeline(self.root).run()
        ledger = self.ledger(report)
        for file_id in {key[0] for key in ledger}:
            self.assertEqual(ledger[(file_id, "features")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "dimensions")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "total")]["status"], "reuse")
            self.assertEqual(ledger[(file_id, "serialization")]["status"], "recompute")
        self.assertEqual(report["outputs"]["version"], 3)
        self.assertTrue(Path(report["outputs"]["path"]).exists())

    def test_v3_bit_width_change_does_not_invalidate_active_v2(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "set_fixed_bits", "value": 2})
        report = EvaluationPipeline(self.root).run()
        self.assertEqual(report["outputs"]["version"], 2)
        self.assertEqual(report["cache_ledger"]["reuse_count"], 24)
        self.assertEqual(report["cache_ledger"]["recompute_count"], 0)

    def test_unrelated_level_addition_reuses_serialization(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(
            self.root,
            {
                "op": "add_level",
                "code": "4",
                "name": "outstanding",
                "min_total_permille": 950,
            },
        )
        report = EvaluationPipeline(self.root).run()
        self.assertEqual(report["cache_ledger"]["recompute_count"], 0)
        self.assertEqual(report["cache_ledger"]["reuse_count"], 24)

    def test_level_removal_reports_unmapped_and_only_affects_users(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "remove_level", "code": "3"})
        report = EvaluationPipeline(self.root).run()
        self.assertTrue(
            any(item["code"] == "E_LEVEL_UNMAPPED" for item in report["failures"])
        )
        self.assertIn("E_LEVEL_UNMAPPED", report["business_codes"])
        ledger = self.ledger(report)
        self.assertEqual(ledger[("sample_clean", "serialization")]["status"], "recompute")
        self.assertEqual(ledger[("sample_boundary", "serialization")]["status"], "reuse")
        self.assertEqual(ledger[("sample_clean", "features")]["status"], "reuse")
        self.assertEqual(ledger[("sample_clean", "dimensions")]["status"], "reuse")

    def test_snapshot_interrupt_and_resume(self) -> None:
        with self.assertRaises(InterruptedRun):
            EvaluationPipeline(self.root, stop_after_stage="parse").run()
        manifest = json.loads(
            (self.root / "snapshots" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["complete"])
        self.assertEqual(sorted(manifest["stages"]), ["parse"])
        report = EvaluationPipeline(self.root).run()
        self.assertTrue(report["stages"]["parse"]["resumed"])
        self.assertIn("parse", report["snapshot"]["resumed_stages"])
        self.assertTrue(report["snapshot"]["complete"])

    def test_serial_and_concurrent_runs_are_equal(self) -> None:
        concurrent_root = self.root
        serial_root = self.root.parent / "serial_workspace"
        shutil.copytree(concurrent_root, serial_root)
        concurrent = EvaluationPipeline(concurrent_root, rebuild_cache=True, rebuild_snapshot=True).run()
        serial = EvaluationPipeline(
            serial_root,
            rebuild_cache=True,
            rebuild_snapshot=True,
            serial_mode=True,
        ).run()
        comparison = compare_reports(concurrent, serial)
        self.assertTrue(comparison["equal"], msg=json.dumps(comparison, indent=2))
        self.assertTrue(serial["concurrency"]["serial_matches_concurrent"])
        self.assertEqual(serial["concurrency"]["workers"], 1)

    def test_incremental_and_cold_rebuild_products_are_equal(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "set_fixed_bits", "value": 2})
        apply_edit(self.root, {"op": "switch_version", "value": 3})
        incremental = EvaluationPipeline(self.root).run()
        cold_root = self.root.parent / "cold_workspace"
        cold_root.mkdir()
        for filename in ("config.json", "level_map.json", "state.json"):
            shutil.copyfile(self.root / filename, cold_root / filename)
        shutil.copytree(self.root / "samples", cold_root / "samples")
        shutil.copytree(self.root / "eval", cold_root / "eval")
        cold = EvaluationPipeline(cold_root).run()
        comparison = compare_reports(incremental, cold)
        self.assertTrue(comparison["equal"], msg=json.dumps(comparison, indent=2))
        self.assertGreater(incremental["cache_ledger"]["reuse_count"], 0)
        self.assertEqual(cold["cache_ledger"]["reuse_count"], 0)

    def test_report_contains_diagnostics_and_injections(self) -> None:
        EvaluationPipeline(self.root).run()
        apply_edit(self.root, {"op": "set_fixed_bits", "value": 2})
        report = EvaluationPipeline(self.root).run()
        self.assertIn("degradations", report)
        self.assertIn("boundary_samples", report)
        self.assertTrue(report["boundary_samples"])
        self.assertIn("statistics", report["degradations"])
        injection_map = {item["id"]: item for item in report["injections"]}
        self.assertTrue(injection_map["INJ-002"]["hit"])
        self.assertTrue(injection_map["INJ-004"]["hit"])
        self.assertTrue(injection_map["INJ-005"]["hit"])
        self.assertIn("stage_order", json.dumps(report["cache_ledger"]))
        self.assertIn("DEG_V3_FIXED_WIDTH", report["business_codes"])
        self.assertIn("E_SNAPSHOT_STALE", report["business_codes"])

    def test_worker_count_is_capped(self) -> None:
        raw = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        raw["max_concurrency"] = 100
        raw["concurrency_cap"] = 3
        (self.root / "config.json").write_text(json.dumps(raw), encoding="utf-8")
        report = EvaluationPipeline(self.root).run()
        self.assertEqual(report["concurrency"]["workers"], 3)
        self.assertTrue(report["concurrency"]["within_cap"])


if __name__ == "__main__":
    unittest.main()
