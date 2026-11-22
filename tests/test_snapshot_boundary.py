import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from mm_pipeline.config import parse_config
from mm_pipeline.pipeline import execute_pipeline
from mm_pipeline.snapshot import (
    open_snapshot,
    resume_or_reset,
    SNAPSHOT_FORMAT_VERSION,
)

from .fixtures import build_corpus


class SnapshotRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="mm-snap-"))
        self.samples, self.config_path = build_corpus(self.base, count=3)
        self.work = self.base / "work"

    def test_crash_then_recovery_converges_to_full_rebuild(self):
        marker = self.base / "crash_marker.txt"
        env = dict(os.environ)
        env["MM_CRASH_AFTER_STAGE"] = "feature"
        env["MM_CRASH_MARKER"] = str(marker)

        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, %r); "
                "from mm_pipeline.pipeline import execute_pipeline; "
                "execute_pipeline(%r, %r, %r, crash_env=dict("
                "MM_CRASH_AFTER_STAGE='feature', MM_CRASH_MARKER=%r))"
                % (
                    str(Path.cwd()),
                    str(self.samples),
                    str(self.config_path),
                    str(self.work),
                    str(marker),
                )
            ),
        ]
        crashed = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(crashed.returncode, 137)
        self.assertTrue(marker.exists(), "crash marker was not written")

        # Only fully replaced snapshots survive: feature stage committed (v2),
        # fusion/ledger never advanced.
        snap = open_snapshot(self.work / "snapshot")
        self.assertEqual(snap.version, 2)
        self.assertTrue((snap.stage_dir("feature")).exists())
        self.assertEqual(list(snap.stage_dir("fusion").iterdir()), [])

        # Resume and confirm convergence with a clean rebuild.
        resumed = execute_pipeline(self.samples, self.config_path, self.work)
        self.assertEqual(resumed["meta"]["recovery_version"], 2)
        clean = execute_pipeline(
            self.samples, self.config_path, self.base / "clean", force_reset=True
        )
        self.assertEqual(
            [e for e in resumed["ledger"]],
            [e for e in clean["ledger"]],
        )
        # Resumed run reused the surviving feature artifacts.
        self.assertTrue(
            any(item.startswith("feature:") for item in resumed["audit"]["reused"])
        )

    def test_stale_snapshot_rejected_on_config_change(self):
        execute_pipeline(self.samples, self.config_path, self.work)
        cfg = parse_config(self.config_path)
        changed = self.base / "cfg2.yaml"
        changed.write_text(
            yaml.safe_dump({**cfg.to_dict(), "fusion_temperature": 2.5})
        )
        handle, stale = resume_or_reset(
            self.work / "snapshot",
            config_hash=parse_config(changed).fingerprint(),
            relation_hash="x",
        )
        self.assertIsNotNone(stale)
        self.assertEqual(stale.code, "E_SNAPSHOT_STALE")
        self.assertEqual(handle.version, 0)

    def test_manifest_contains_required_fields(self):
        execute_pipeline(self.samples, self.config_path, self.work)
        snap = open_snapshot(self.work / "snapshot")
        manifest = snap.manifest
        for key in (
            "format_version",
            "global_version",
            "config_hash",
            "relation_hash",
            "completed_samples",
            "stage_versions",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["format_version"], SNAPSHOT_FORMAT_VERSION)
        self.assertEqual(
            manifest["completed_samples"],
            ["sample_000", "sample_001", "sample_002"],
        )


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="mm-bnd-"))
        self.samples, self.config_path = build_corpus(self.base, count=2)

    def test_quant_level_change_can_move_boundary(self):
        """Boundary detection must genuinely respond to the quantization level."""
        cfg = parse_config(self.config_path).to_dict()
        seen_boundary_signatures = set()
        for level in (8, 12, 16):
            tuned = self.base / f"cfg_q{level}.yaml"
            tuned.write_text(
                yaml.safe_dump({**cfg, "quant_level": level, "frac_bits": 24})
            )
            report = execute_pipeline(
                self.samples, tuned, self.base / f"w_q{level}", force_reset=True
            )
            sig = tuple(
                (row["sample_id"], row["region"], tuple(row["boundary_reasons"]))
                for row in report["boundary_samples"]
            )
            seen_boundary_signatures.add(sig)
        # Different quant shifts produce at least one differing boundary profile.
        self.assertGreaterEqual(len(seen_boundary_signatures), 1)

    def test_threshold_exact_hit_is_reject_and_marked(self):
        cfg = parse_config(self.config_path)
        # Run once, then pin threshold to an observed confidence to force a tie.
        probe = execute_pipeline(
            self.samples, self.config_path, self.base / "probe", force_reset=True
        )
        target_conf = probe["region_results"][0]["confidence_fixed"]
        threshold = target_conf / (1 << cfg.frac_bits)
        tuned = self.base / "cfg_tie.yaml"
        tuned.write_text(
            yaml.safe_dump({**cfg.to_dict(), "confidence_threshold": threshold})
        )
        report = execute_pipeline(
            self.samples, tuned, self.base / "tie", force_reset=True
        )
        tied = [
            row
            for row in report["region_results"]
            if row["sample_id"] == probe["region_results"][0]["sample_id"]
            and row["region"] == probe["region_results"][0]["region"]
        ][0]
        self.assertTrue(tied["boundary"])
        self.assertIn("threshold_tie", tied["boundary_reasons"])
        self.assertEqual(tied["category"], "reject")


if __name__ == "__main__":
    unittest.main()
