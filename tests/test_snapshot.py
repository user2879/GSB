import shutil
import tempfile
import unittest
from pathlib import Path

from quantcalib.data.config import load_config
from quantcalib.data.samples import generate_workspace
from quantcalib.engine import EngineContext, QuantCalibEngine
from quantcalib.runner import comparable_view, run_full_rebuild


class SnapshotTests(unittest.TestCase):
    def test_stale_tmp_transaction_is_reclaimed(self):
        workdir = Path(tempfile.mkdtemp(prefix="qc_snap_"))
        try:
            generate_workspace(workdir)
            cfg, _ = load_config(workdir / "config" / "quant.yaml")
            engine = QuantCalibEngine(EngineContext(workdir=workdir), cfg)
            engine.run(0)

            tmp = workdir / "out" / "snapshot.tmp"
            tmp.mkdir()
            (tmp / "orphan").write_text("partial", encoding="utf-8")

            resumed_engine = QuantCalibEngine(
                EngineContext(workdir=workdir), cfg)
            resumed = resumed_engine.run(0)
            self.assertFalse(tmp.exists())
            self.assertTrue(resumed["resumed"])

            isolated = Path(tempfile.mkdtemp()) / "snapshot"
            clean = run_full_rebuild(workdir, isolated_snapshot=isolated)
            self.assertEqual(
                comparable_view(resumed), comparable_view(clean))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
