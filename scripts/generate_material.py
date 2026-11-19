"""Render the deterministic 5.1 material and independently verify alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loudness_workbench.audio_io import read_wav  # noqa: E402
from loudness_workbench.bs1770 import StreamingLoudnessMeter  # noqa: E402
from loudness_workbench.config import (  # noqa: E402
    ABSOLUTE_GATE_LKFS,
    RELATIVE_GATE_OFFSET_LU,
    ms_to_samples,
)
from loudness_workbench.synth import render_material  # noqa: E402


def verify_alignment(wav_path: str, label_path: str) -> dict:
    signal, sample_rate, _ = read_wav(wav_path)
    manifest = json.loads(Path(label_path).read_text(encoding="utf-8"))
    fade = ms_to_samples(manifest["crossfade_ms"], sample_rate)
    checks = {"segments": [], "all_in_bounds": True}
    for lab in manifest["segments"]:
        segment = signal[lab["start_sample"] : lab["end_sample"]]
        meter = StreamingLoudnessMeter(sample_rate)
        meter.process(segment)
        gated = meter.finalize(ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU)
        in_bounds = (
            0 <= lab["start_sample"] < lab["end_sample"] <= signal.shape[0]
        )
        checks["all_in_bounds"] &= in_bounds
        checks["segments"].append(
            {
                "segment_id": lab["segment_id"],
                "kind": lab["kind"],
                "target_lkfs": lab["target_integrated_lkfs"],
                "measured_lkfs": round(gated.integrated_lkfs, 3),
                "measured_lra_lu": round(gated.lra_lu, 3),
                "in_bounds": in_bounds,
            }
        )
    expected_length = sum(
        lab["end_sample"] - lab["start_sample"]
        for lab in manifest["segments"]
    ) - fade * (len(manifest["segments"]) - 1)
    checks["length_consistent"] = expected_length == signal.shape[0]
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render and verify synthetic 5.1 material")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "programme_5p1_48k_24bit.wav"
    labels = out / "labels.json"
    manifest = render_material(str(wav), str(labels), args.seed)
    verification = verify_alignment(str(wav), str(labels))
    (out / "alignment_check.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("alignment:", json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["all_in_bounds"] and verification["length_consistent"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
