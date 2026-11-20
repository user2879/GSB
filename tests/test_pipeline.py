from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vibration_pipeline.config import parse_config
from vibration_pipeline.data import load_data, load_yaml
from vibration_pipeline.engine import run_pipeline
from vibration_pipeline.features import extract_features
from vibration_pipeline.util import hash_object


ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    return (
        load_yaml(ROOT / "config" / "pipeline.yaml"),
        load_yaml(ROOT / "data" / "devices.yaml"),
        load_yaml(ROOT / "config" / "injections.yaml"),
    )


class PipelineTests(unittest.TestCase):
    def assert_same_result(self, incremental, full):
        self.assertEqual(incremental["result_fingerprint"], full["result_fingerprint"])
        self.assertEqual(incremental["segments"], full["segments"])
        self.assertEqual(incremental["boundary_segments"], full["boundary_segments"])
        self.assertEqual(incremental["failures"], full["failures"])
        self.assertEqual(incremental["injection_hits"], full["injection_hits"])
        self.assertEqual(incremental["waveform_fingerprint"], full["waveform_fingerprint"])

    def test_deterministic_features_and_preprocessing(self):
        config_spec, data_spec, _ = fixtures()
        config = parse_config(config_spec)
        loaded = load_data(data_spec, config)
        segment = loaded.devices["pump-a"].segments[0]
        device = loaded.devices["pump-a"]
        self.assertTrue(np.all(np.isfinite(segment.samples)))
        self.assertTrue(any(device_segment.missing_indices for device_segment in device.segments))
        self.assertTrue(any(device_segment.resampled_indices for device_segment in device.segments))
        first = extract_features(
            segment.samples,
            segment_length=segment.segment_length,
            sample_rate_hz=segment.sample_rate_hz,
            band_edges_hz=device.band_edges_hz,
            need_time=True,
            need_band=True,
            waveform_fingerprint=segment.waveform_fingerprint,
            preprocessing_fingerprint=segment.preprocess_fingerprint,
            mode=config.features.mode,
        )
        second = extract_features(
            segment.samples,
            segment_length=segment.segment_length,
            sample_rate_hz=segment.sample_rate_hz,
            band_edges_hz=device.band_edges_hz,
            need_time=True,
            need_band=True,
            waveform_fingerprint=segment.waveform_fingerprint,
            preprocessing_fingerprint=segment.preprocess_fingerprint,
            mode=config.features.mode,
        )
        self.assertEqual(first.vector, second.vector)
        self.assertEqual(first.content_fingerprint, second.content_fingerprint)
        self.assertEqual(first.feature_names[:3], ("rms", "kurtosis", "peak"))
        self.assertEqual(first.feature_names[3:], tuple(f"band_energy_{i}" for i in range(4)))
        self.assertTrue(all(np.isfinite(first.vector)))

    def test_validation_errors_continue_other_devices(self):
        config_spec, data_spec, _ = fixtures()
        valid_device = copy.deepcopy(data_spec["devices"][2])
        unknown = copy.deepcopy(valid_device)
        unknown["id"] = "unknown-device"
        bad_length = copy.deepcopy(valid_device)
        bad_length["id"] = "bad-length"
        bad_length["segment_length"] = 0
        bad_mismatch = copy.deepcopy(valid_device)
        bad_mismatch["id"] = "bad-mismatch"
        bad_mismatch["declared_segment_count"] = 9
        bad_ratio = copy.deepcopy(valid_device)
        bad_ratio["id"] = "bad-ratio"
        bad_ratio["anomaly_ratio"] = 1.2
        bad_shape = copy.deepcopy(valid_device)
        bad_shape["id"] = "bad-shape"
        bad_shape["sample_count"] = 1000
        config_spec["known_devices"] = ["fan-c", "bad-length", "bad-mismatch", "bad-ratio", "bad-shape"]
        data = {"devices": [valid_device, unknown, bad_length, bad_mismatch, bad_ratio, bad_shape]}
        with tempfile.TemporaryDirectory() as tmp:
            report = run_pipeline(config_spec, data, None, snapshot_dir=Path(tmp) / "snap", mode="full")
        codes = {failure["code"] for failure in report["failures"]}
        self.assertTrue(
            {"E_CONFIG_INVALID", "E_SEGMENT_LEN_INVALID", "E_SEGMENT_MISMATCH", "E_RATIO_OUT_OF_RANGE"}.issubset(codes)
        )
        self.assertTrue(
            any(item["device_id"] == "bad-shape" and "sample_count" in item["detail"] for item in report["failures"])
        )
        self.assertGreater(report["segment_count"], 0)

    def test_global_config_errors_and_valid_override(self):
        config_spec, data_spec, _ = fixtures()
        config_spec["segment_length"] = 0
        config_spec["threshold"] = 3.0
        config_spec["concurrency"] = 0
        config_spec["features"]["band_edges_hz"] = [0.0, 100.0, 100.0]
        config_spec["quantization"]["quantum"] = 0
        config_spec["algorithms"]["unknown_algorithm"] = {"enabled": True}
        device = copy.deepcopy(data_spec["devices"][2])
        device["segment_length"] = 256
        device["band_edges_hz"] = [0.0, 250.0, 500.0]
        device["band_count"] = 2
        data = {"devices": [device]}
        with tempfile.TemporaryDirectory() as tmp:
            report = run_pipeline(config_spec, data, None, snapshot_dir=Path(tmp) / "snap", mode="full")
        issue_codes = {issue["code"] for issue in report["config_issues"]}
        self.assertTrue(
            {
                "E_SEGMENT_LEN_INVALID",
                "E_BAND_INVALID",
                "E_THRESHOLD_INVALID",
                "E_CONCURRENCY_INVALID",
                "E_QUANT_INVALID",
                "E_CONFIG_INVALID",
            }.issubset(issue_codes)
        )
        self.assertEqual(report["segment_count"], 4)
        self.assertLessEqual(report["run"]["max_observed_concurrency"], 1)

    def test_incremental_edit_sequence_matches_full_rebuild(self):
        base_config, base_data, _ = fixtures()
        append_edits = {
            "segment_edits": {"pump-a": {"append": [{"count": 1, "seed_delta": 991, "source": "test-append"}]}}
        }
        replace_edits = copy.deepcopy(append_edits)
        replace_edits["segment_edits"]["motor-b"] = {
            "replace": [{"segment_index": 3, "seed_delta": 443, "source": "test-replace"}]
        }
        cumulative_edits = copy.deepcopy(replace_edits)
        cumulative_edits["segment_edits"]["fan-c"] = {"delete": [1]}
        steps = [
            ("append", base_config, append_edits),
            ("replace", base_config, replace_edits),
            ("delete", base_config, cumulative_edits),
        ]
        param_config = copy.deepcopy(base_config)
        param_config["algorithms"]["robust_mahalanobis"]["candidate_index"] = 1
        steps.append(("parameter", param_config, cumulative_edits))
        band_config = copy.deepcopy(param_config)
        band_config["features"]["band_edges_hz"] = [0.0, 75.0, 150.0, 300.0, 500.0]
        steps.append(("bands", band_config, cumulative_edits))
        quant_config = copy.deepcopy(band_config)
        quant_config["quantization"]["decimal_places"] = 2
        quant_config["quantization"]["quantum"] = 2
        steps.append(("quantization", quant_config, cumulative_edits))
        threshold_config = copy.deepcopy(quant_config)
        threshold_config["threshold"] = 0.625
        steps.append(("threshold", threshold_config, cumulative_edits))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incremental = run_pipeline(
                base_config, base_data, None, snapshot_dir=root / "incremental", mode="incremental"
            )
            self.assertEqual(incremental["snapshot"]["stale"], False)
            for name, config_spec, edits in steps:
                incremental = run_pipeline(
                    config_spec,
                    base_data,
                    None,
                    snapshot_dir=root / "incremental",
                    mode="incremental",
                    edits_spec=edits,
                )
                full = run_pipeline(
                    config_spec,
                    base_data,
                    None,
                    snapshot_dir=root / f"full-{name}",
                    mode="full",
                    edits_spec=edits,
                )
                self.assert_same_result(incremental, full)
                if name in ("parameter", "bands", "quantization", "threshold"):
                    self.assertTrue(incremental["snapshot"]["stale"])
                    self.assertEqual(incremental["snapshot"]["stale_code"], "E_SNAPSHOT_STALE")
                if name == "threshold":
                    self.assertTrue(all(item["layer"] != "algorithm.raw" for item in incremental["recompute"]))
                    self.assertTrue(all(item["layer"] != "algorithm.quantized" for item in incremental["recompute"]))
                    self.assertTrue(any(item["layer"] == "decision" for item in incremental["recompute"]))
                if name == "quantization":
                    self.assertTrue(all(item["layer"] != "algorithm.raw" for item in incremental["recompute"]))
                    self.assertTrue(any(item["layer"] == "algorithm.quantized" for item in incremental["recompute"]))
                if name == "parameter":
                    recomputed_raw = {
                        item["algorithm"] for item in incremental["recompute"] if item["layer"] == "algorithm.raw"
                    }
                    self.assertEqual(recomputed_raw, {"robust_mahalanobis"})
                if name == "bands":
                    reused_raw = {item["algorithm"] for item in incremental["reuse"] if item["layer"] == "algorithm.raw"}
                    recomputed_raw = {
                        item["algorithm"] for item in incremental["recompute"] if item["layer"] == "algorithm.raw"
                    }
                    self.assertIn("crest_factor", reused_raw)
                    self.assertEqual(recomputed_raw, {"robust_mahalanobis", "spectral_residual"})

    def test_concurrency_matches_serial_and_respects_limit(self):
        config_spec, data_spec, injections = fixtures()
        serial_config = copy.deepcopy(config_spec)
        serial_config["concurrency"] = 1
        parallel_config = copy.deepcopy(config_spec)
        parallel_config["concurrency"] = 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            serial = run_pipeline(
                serial_config, data_spec, injections, snapshot_dir=root / "serial", mode="full"
            )
            parallel = run_pipeline(
                parallel_config, data_spec, injections, snapshot_dir=root / "parallel", mode="full"
            )
        self.assertEqual(serial["result_fingerprint"], parallel["result_fingerprint"])
        self.assertEqual(serial["segments"], parallel["segments"])
        self.assertEqual(serial["failures"], parallel["failures"])
        self.assertEqual(serial["boundary_segments"], parallel["boundary_segments"])
        self.assertEqual(parallel["run"]["max_observed_concurrency"], 4)
        self.assertEqual(serial["run"]["max_observed_concurrency"], 1)

    def test_injection_hits_and_boundary_diagnostics(self):
        config_spec, data_spec, injections = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            report = run_pipeline(
                config_spec, data_spec, injections, snapshot_dir=Path(tmp) / "snap", mode="full"
            )
        hits = {item["id"]: item for item in report["injection_hits"]}
        self.assertTrue(all(hits[key]["hit"] for key in hits))
        self.assertEqual(set(hits), {item["id"] for item in injections["injections"]})
        failures = report["failures"]
        self.assertTrue(any(item["code"] == "E_NONFINITE_SAMPLE" and item["injected_id"] == "inj-01-nonfinite" for item in failures))
        self.assertTrue(any(item["code"] == "E_BAND_INVALID" and item["injected_id"] == "inj-02-band-oob" for item in failures))
        self.assertTrue(any(item["code"] == "E_ALGO_PARAM_INVALID" and item["injected_id"] == "inj-03-crest-window-zero" for item in failures))
        target_boundary = [
            item
            for item in report["boundary_segments"]
            if item["device_id"] == "motor-b" and item["segment_index"] == 3
        ]
        self.assertEqual(len(target_boundary), 1)
        boundary = target_boundary[0]
        self.assertEqual(boundary["score_ticks"], boundary["threshold_ticks"])
        self.assertIn("tie_policy=anomaly", boundary["reason"])
        self.assertIn("inj-04-threshold-boundary", boundary["injection_ids"])

    def test_quantization_changes_integers_and_marks(self):
        config_spec, data_spec, _ = fixtures()
        quantized_config = copy.deepcopy(config_spec)
        quantized_config["quantization"]["quantum"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_pipeline(config_spec, data_spec, None, snapshot_dir=root / "a", mode="full")
            second = run_pipeline(quantized_config, data_spec, None, snapshot_dir=root / "b", mode="full")
        ints_first = [(segment["segment_total_score_ticks"], [score["normalized_ticks"] for score in segment["algorithm_scores"]]) for segment in first["segments"]]
        ints_second = [(segment["segment_total_score_ticks"], [score["normalized_ticks"] for score in segment["algorithm_scores"]]) for segment in second["segments"]]
        marks_first = [(segment["device_id"], segment["segment_index"]) for segment in first["segments"] if segment["segment_anomaly"]]
        marks_second = [(segment["device_id"], segment["segment_index"]) for segment in second["segments"] if segment["segment_anomaly"]]
        self.assertNotEqual(ints_first, ints_second)
        self.assertNotEqual(marks_first, marks_second)

    def test_snapshot_reuse_and_stale_rebuild(self):
        config_spec, data_spec, _ = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snap"
            initial = run_pipeline(config_spec, data_spec, None, snapshot_dir=snapshot, mode="incremental")
            same = run_pipeline(config_spec, data_spec, None, snapshot_dir=snapshot, mode="incremental")
            self.assertEqual(initial["result_fingerprint"], same["result_fingerprint"])
            self.assertEqual(same["recompute_count"], 0)
            self.assertGreater(same["reuse_count"], 0)
            changed = copy.deepcopy(config_spec)
            changed["threshold"] = 0.61
            incremental = run_pipeline(changed, data_spec, None, snapshot_dir=snapshot, mode="incremental")
            full = run_pipeline(changed, data_spec, None, snapshot_dir=root / "full", mode="full")
            self.assert_same_result(incremental, full)
            self.assertEqual(incremental["snapshot"]["stale_code"], "E_SNAPSHOT_STALE")
            self.assertTrue((snapshot / "manifest.json").exists())
            for stage in ("feature_extraction", "scoring", "threshold", "summary"):
                self.assertTrue((snapshot / "stages" / stage / "data.json").exists())
            manifest = incremental["snapshot"]["manifest"]
            self.assertEqual(manifest["snapshot_version"], 1)
            self.assertEqual(manifest["config_hash"], incremental["config_hash"])
            self.assertIn("segment_cursors", manifest)
            self.assertIn("stages", manifest)

    def test_pre_requested_stop_is_graceful(self):
        config_spec, data_spec, _ = fixtures()
        from vibration_pipeline.engine import StopController

        controller = StopController()
        controller.request(15)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_pipeline(
                config_spec,
                data_spec,
                None,
                snapshot_dir=Path(tmp) / "snap",
                mode="full",
                stop_controller=controller,
            )
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["run"]["exit_code"], 0)
        self.assertEqual(report["snapshot"]["manifest"]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
