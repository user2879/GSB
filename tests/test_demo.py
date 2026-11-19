from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from acoustic_workbench.audio import decode_wav_24
from acoustic_workbench.deconvolution import recompute_channel_ir
from acoustic_workbench.demo import run_demo
from acoustic_workbench.models import MeasurementRun, config_from_mapping
from acoustic_workbench.simulation import estimate_noise_sigma
from acoustic_workbench.snapshot import SnapshotStore


class DemoIntegrationTests(unittest.TestCase):
    def test_fixed_demo_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-demo-test-") as temporary:
            output = Path(temporary) / "demo"
            manifest = run_demo(output, verify_determinism=True)
            check = manifest["determinism_check"]
            self.assertTrue(check["identical"])
            self.assertEqual(check["content_mismatches"], [])
            self.assertEqual(check["only_first"], [])
            self.assertEqual(check["only_second"], [])
            self.assertGreater(check["compared_files"], 20)
            self.assertEqual(len(manifest["steps"]), 5)
            self.assertEqual(
                {event["code"] for event in manifest["error_chain"]},
                {
                    "E_AUDIO_FORMAT",
                    "E_SYNC_MARK",
                    "E_DELAY_BIAS",
                    "E_CALIB_MISMATCH",
                    "E_SNAPSHOT_STALE",
                    "E_BAND_INVALID",
                    "E_FILTER_LEN",
                },
            )
            self.assertTrue(all(step["acceptance"]["calibration_ok"] for step in manifest["steps"]))

            snapshot = SnapshotStore(output / "state" / "current_snapshot.json").load()
            config = config_from_mapping(snapshot.data["config"])
            wav_bytes = (output / "measurement" / "measurement.wav").read_bytes()
            recorded, fmt = decode_wav_24(
                wav_bytes,
                expected_sample_rate=48_000,
                expected_channels=4,
                expected_bit_depth=24,
            )
            source = np.load(output / "measurement" / "source_reference.npy", allow_pickle=False)
            truth = snapshot.data["sync"]["truth"]
            run = MeasurementRun(
                config=config.measurement,
                source=source,
                recorded=recorded,
                wav_bytes=wav_bytes,
                sync_markers=tuple(dict(item) for item in truth),
                sweep_starts=tuple(int(item["sweep_start_sample"]) for item in truth),
                source_hash=snapshot.data["input_hashes"]["source"],
                recorded_hash=snapshot.data["input_hashes"]["recorded_samples"],
                wav_hash=snapshot.data["input_hashes"]["wav"],
                noise_sigma_estimate=estimate_noise_sigma(recorded, config.measurement),
            )
            self.assertEqual(fmt.total_frames, recorded.shape[0])
            recomputed = recompute_channel_ir(run, config, 2)
            self.assertEqual(
                recomputed.ir_hash,
                snapshot.data["deconvolution"]["channels"][2]["ir_hash"],
            )


if __name__ == "__main__":
    unittest.main()
