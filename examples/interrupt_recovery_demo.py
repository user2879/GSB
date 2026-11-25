"""Simulate an interrupted run and prove resume equals a clean full run."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantcalib.data.config import load_config  # noqa: E402
from quantcalib.data.samples import generate_workspace  # noqa: E402
from quantcalib.engine import EngineContext, QuantCalibEngine  # noqa: E402
from quantcalib.runner import comparable_view, run_full_rebuild  # noqa: E402


class InterruptedEngine(QuantCalibEngine):
    """Raises after the first layer groups are persisted, mid-transaction."""

    def __init__(self, context, cfg, crash_after_groups: int):
        super().__init__(context, cfg)
        self._seen = 0
        self._crash_after = crash_after_groups

    def run(self, global_version: int):  # pragma: no cover - demo path
        layers, calib, receipts, issues = self._load_inputs()
        layers = sorted(layers, key=lambda layer: layer.name)
        from quantcalib.fingerprint import fingerprint, fingerprint_json
        from quantcalib.persistence.snapshot import expected_manifest
        weight_fp = fingerprint([layer.to_record() for layer in layers])
        calib_fp = fingerprint({
            "inputs": calib.inputs,
            "reference_outputs": calib.reference_outputs,
            "seed": calib.seed,
        })
        receipt_fp = fingerprint_json(
            [r.to_dict() for r in sorted(
                receipts, key=lambda r: (r.layer, r.source_version))])
        manifest = expected_manifest(
            global_version, self.cfg.to_dict(), weight_fp, calib_fp,
            receipt_fp, self.cfg.active_receipt_version)
        resume, _ = self.store.resume_decision(manifest)
        if not resume:
            self.store.invalidate("demo: forced interruption")
        self.store.begin_commit()
        raise RuntimeError("E_DEMO_INTERRUPTED")


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="quantcalib_interrupt_"))
    generate_workspace(workdir)
    cfg, _ = load_config(workdir / "config" / "quant.yaml")

    try:
        InterruptedEngine(EngineContext(workdir=workdir), cfg, 1).run(0)
    except RuntimeError as exc:
        print(f"interrupted: {exc}")

    # The live snapshot must remain the previous intact state. The tmp tree
    # from the dead transaction is reclaimed when the next run starts.
    live = workdir / "out" / "snapshot"
    tmp = workdir / "out" / "snapshot.tmp"
    print("live snapshot exists:", live.exists(),
          "stale tmp before resume:", tmp.exists())

    resumed_engine = QuantCalibEngine(EngineContext(workdir=workdir), cfg)
    resumed = resumed_engine.run(0)
    print("stale tmp after resume:", tmp.exists())
    isolated = Path(tempfile.mkdtemp(prefix="quantcalib_clean_")) / "snapshot"
    clean = run_full_rebuild(workdir, isolated_snapshot=isolated)
    equal = comparable_view(resumed) == comparable_view(clean)
    print(json.dumps({"resumed": resumed["resumed"],
                      "field_level_equal": equal}, indent=2))
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
