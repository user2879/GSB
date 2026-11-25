from __future__ import annotations

import json
import unittest

from speaker_diag.errors import DiagnosticError
from speaker_diag.samples import default_workload
from speaker_diag.spec import load_workload, load_workload_data


class SpecValidationTests(unittest.TestCase):
    def test_sample_validates_and_derives_windows(self) -> None:
        spec = load_workload("samples/workload.json")
        self.assertEqual(spec.audio.sample_rate, 16000)
        self.assertEqual(len(spec.window_ids), 36)
        self.assertEqual(spec.window_starts_samples[0], 0)
        self.assertEqual(spec.window_starts_samples[1], 3200)
        self.assertLess(spec.window.hop_ms, spec.window.window_ms)
        self.assertEqual(len({worker.port for worker in spec.workers}), len(spec.workers))

    def test_hop_must_be_smaller_than_window(self) -> None:
        data = default_workload(1)
        data["window"]["hop_ms"] = data["window"]["window_ms"]
        text = json.dumps(data, ensure_ascii=False, indent=2)
        expected_line = next(
            index for index, line in enumerate(text.splitlines(), start=1) if '"hop_ms"' in line
        )
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(source_text=text, source_name="<bad-hop>")
        exc = caught.exception
        self.assertEqual(exc.code, "E_SPEC")
        self.assertEqual(exc.field, "window.hop_ms")
        self.assertEqual(exc.line, expected_line)

    def test_duplicate_port_reports_specific_worker_line(self) -> None:
        data = default_workload(1)
        data["workers"][1]["port"] = data["workers"][0]["port"]
        text = json.dumps(data, ensure_ascii=False, indent=2)
        port_lines = [
            index for index, line in enumerate(text.splitlines(), start=1) if '"port"' in line
        ]
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(source_text=text)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "workers[1].port")
        self.assertEqual(caught.exception.line, port_lines[1])

    def test_non_loopback_host_is_rejected(self) -> None:
        data = default_workload(1)
        data["workers"][0]["host"] = "0.0.0.0"
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(data)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "workers[0].host")

    def test_quantization_width_must_be_positive(self) -> None:
        data = default_workload(1)
        data["quantization"]["fractional_bits"] = 0
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(data)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "quantization.fractional_bits")

    def test_request_window_must_exist(self) -> None:
        data = default_workload(1)
        data["requests"][0]["window_ids"][0] = "W9999"
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(data)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "requests[0].window_ids[0]")

    def test_json_syntax_error_is_e_spec_with_line(self) -> None:
        text = '{\n  "spec_version": 1,\n  "audio": {oops}\n}\n'
        with self.assertRaises(DiagnosticError) as caught:
            load_workload_data(source_text=text)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.line, 3)

    def test_window_overlap_ratio_can_be_derived(self) -> None:
        data = default_workload(1)
        del data["window"]["overlap_bps"]
        spec = load_workload_data(data)
        self.assertEqual(spec.window.overlap_bps, 5000)

    def test_optional_audio_display_defaults_are_explicit(self) -> None:
        data = default_workload(1)
        del data["audio"]["speaker_count"]
        del data["audio"]["pause_ms"]
        del data["audio"]["overlap_bps"]
        spec = load_workload_data(data)
        self.assertEqual(spec.audio.speaker_count, 3)
        self.assertEqual(spec.audio.pause_min_ms, 70)
        self.assertEqual(spec.audio.overlap_bps, 1200)


if __name__ == "__main__":
    unittest.main()
