#!/usr/bin/env python3
"""Run the local acceptance matrix and print one machine-readable summary.

The script is intentionally self-contained: it uses only the deterministic
sample generator, local snapshots, and a child process for the forced-kill test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_diag.audio import read_wav, write_pcm16_wav  # noqa: E402
from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions  # noqa: E402
from voice_diag.synthesize import generate_demo  # noqa: E402


def run(sample: Path, out: Path, **options) -> dict:
    options.setdefault("max_batches", 32)
    return DiagnosticPipeline(
        sample / "config.json",
        sample / "manifest.json",
        out,
        options=PipelineOptions(**options),
    ).run()


def mutate_audio(path: Path) -> None:
    wav = read_wav(path)
    pcm = wav.pcm.astype(np.int32)
    pcm = np.clip(pcm + ((np.arange(pcm.size) % 19) - 9) * 2, -32768, 32767).astype("<i2")
    write_pcm16_wav(path, pcm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", help="optional directory in which to keep verification outputs")
    parser.add_argument("--out-json", help="optional path for the resulting summary JSON")
    args = parser.parse_args()
    temporary = None
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="voice-diag-acceptance-")
        work = Path(temporary.name)

    base = work / "base"
    generate_demo(base, overwrite=True)
    baseline = run(base, work / "out-baseline", max_workers=4)
    reuse = run(base, work / "out-baseline", max_workers=4)

    inc = work / "incremental"
    shutil.copytree(base, inc, dirs_exist_ok=True)
    inc_out = work / "out-incremental"
    inc_before = run(inc, inc_out, max_workers=4)
    mutate_audio(inc / "audio" / "F002.wav")
    inc_after = run(inc, inc_out, max_workers=4)
    inc_full = run(inc, work / "out-incremental-full", max_workers=4)

    crash_out = work / "out-crash"
    script = (
        "from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions\n"
        f"DiagnosticPipeline({str(base / 'config.json')!r}, {str(base / 'manifest.json')!r}, "
        f"{str(crash_out)!r}, options=PipelineOptions(max_batches=1, crash_at='after_batch')).run()\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT, check=False)
    recovered = run(base, crash_out, max_workers=4)

    q8 = work / "q8"
    shutil.copytree(base, q8, dirs_exist_ok=True)
    q8_config_path = q8 / "config.json"
    q8_config = json.loads(q8_config_path.read_text(encoding="utf-8"))
    q8_config["analysis"]["frac_bits"] = 8
    q8_config["global_version"] = 8
    q8_config_path.write_text(json.dumps(q8_config, ensure_ascii=False, indent=2), encoding="utf-8")
    q8_result = run(q8, work / "out-q8", max_workers=4)

    arrival_out = work / "out-arrival"
    arrival_partial = run(base, arrival_out, max_workers=4, arrival_limit=2)
    arrival_complete = run(base, arrival_out, max_workers=4)

    summary = {
        "baseline": {
            "status": baseline["status"],
            "final_digest": baseline["final_digest"],
            "frames_digest": baseline["report"]["frames_digest"],
            "receipts": baseline["report"]["receipts"],
            "overlap_frame_count": baseline["report"]["overlap_frame_count"],
            "boundary_frame_count": baseline["report"]["boundary_frame_count"],
            "multi_source_window_count": baseline["report"]["multi_source_window_count"],
            "interval_truth_match_count": baseline["report"]["interval_truth_match_count"],
        },
        "quiet_rerun": {
            "final_digest": reuse["final_digest"],
            "cache_reuse_count": reuse["report"]["cache"]["reuse_count"],
            "cache_invalidation_count": reuse["report"]["cache"]["invalidation_count"],
        },
        "incremental_audio": {
            "baseline_digest": inc_before["final_digest"],
            "incremental_digest": inc_after["final_digest"],
            "independent_full_digest": inc_full["final_digest"],
            "equal_to_full": inc_after["final_digest"] == inc_full["final_digest"],
            "cache_reuse_count": inc_after["report"]["cache"]["reuse_count"],
            "cache_invalidation_count": inc_after["report"]["cache"]["invalidation_count"],
            "invalidation_reasons": sorted(
                {
                    reason
                    for entry in inc_after["report"]["cache"]["invalidation_entries"]
                    for reason in entry["reasons"]
                }
            ),
        },
        "forced_termination": {
            "child_exit_code": child.returncode,
            "resumed_status": recovered["status"],
            "resumed_digest": recovered["final_digest"],
            "equal_to_full": recovered["final_digest"] == baseline["final_digest"],
        },
        "quantization_swap": {
            "q8_status": q8_result["status"],
            "q6_frames_digest": baseline["report"]["frames_digest"],
            "q8_frames_digest": q8_result["report"]["frames_digest"],
            "frames_changed": baseline["report"]["frames_digest"] != q8_result["report"]["frames_digest"],
        },
        "rolling_arrival": {
            "partial_status": arrival_partial["status"],
            "partial_complete": arrival_partial["complete"],
            "partial_cursor": arrival_partial["fragment_cursor"],
            "complete_status": arrival_complete["status"],
            "complete_cursor": arrival_complete["fragment_cursor"],
            "equal_to_full": arrival_complete["final_digest"] == baseline["final_digest"],
        },
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        Path(args.out_json).write_text(payload, encoding="utf-8")
    print(payload, end="")
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
