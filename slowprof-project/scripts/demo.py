"""Reproducible end-to-end demonstration of every profiler capability.

The script never modifies library source code: fault injection only creates a
corrupted *input* copy or passes illegal interval parameters. All output is
teed to ``state/demo_output.txt`` for offline review.

Usage:
    python scripts/demo.py            # full demo (large run included)
    python scripts/demo.py --quick    # skip the 150k-row scale run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowprof import cli  # noqa: E402
from slowprof.diagnostics import tkinter_available  # noqa: E402
from slowprof.folding import cross_check, fold_full  # noqa: E402
from slowprof.generator import generate  # noqa: E402
from slowprof.samples import load_samples, make_interval, summarize_input  # noqa: E402
from slowprof.session import ForcedKill, ProfileSession, SessionConfig  # noqa: E402


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_cli(argv):
    print(f"$ python -m slowprof.cli {' '.join(argv)}")
    code = cli.main(argv)
    print(f"(exit code {code})")
    return code


def main(quick: bool) -> int:
    data = ROOT / "data"
    state = ROOT / "state"
    if state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True)
    data.mkdir(exist_ok=True)

    section("0. environment")
    print(f"python       : {sys.version.split()[0]}")
    import sqlite3
    print(f"sqlite3      : {sqlite3.sqlite_version}")
    print(f"tkinter gui  : {tkinter_available()} "
          "(text fallback used when False)")
    try:
        import numpy
        print(f"numpy        : {numpy.__version__}")
    except Exception:
        print("numpy        : not installed (optional)")

    section("1. deterministic sample generation (5,000 rows)")
    small = data / "samples.ndjson"
    _, summary = generate(
        rows=5_000, services=3, min_depth=3, max_depth=8,
        weight_mode="skewed", seed=20261122, output=str(small))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("input summary:", json.dumps(
        summarize_input(small), indent=2, sort_keys=True))

    section("2. strict validation + E_SAMPLE fault injection (input only)")
    corrupted = data / "samples_corrupt.ndjson"
    run_cli([
        "inject-sample", "--samples", str(small),
        "--output", str(corrupted), "--kind", "bad_weight",
        "--target-line", "17",
    ])
    bad_ledger = state / "bad.sqlite"
    bad_snap = state / "bad_snap"
    run_cli([
        "profile", "--samples", str(corrupted),
        "--ledger", str(bad_ledger), "--snapshot-dir", str(bad_snap),
    ])

    section("3. E_RANGE illegal interval parameter injection")
    run_cli([
        "profile", "--samples", str(small),
        "--ledger", str(state / "range.sqlite"),
        "--snapshot-dir", str(state / "range_snap"),
        "--start-ms", "500", "--end-ms", "100",
    ])

    section("4. dual-path byte-identical folding")
    samples = load_samples(small)
    same, message = cross_check(samples)
    print(f"cross-check identical: {same} ({message})")
    result = fold_full(samples)
    print(f"path reason: {result.path_reason}")
    print("first 5 canonical lines:")
    print("\n".join(result.canonical_bytes().decode().splitlines()[:5]))

    section("5. full session: ranks, contexts, fixed-point annotations")
    cfg = SessionConfig(
        sample_path=str(small),
        ledger_path=str(state / "ledger.sqlite"),
        snapshot_dir=str(state / "snap"),
        scale=10_000, top_k=8, batch_size=1_000,
    )
    session = ProfileSession(cfg)
    session.run()
    from slowprof.diagnostics import render_from_ledger
    print(render_from_ledger(cfg, session))
    session.close()

    section("6. forced-kill durability and recovery")
    crash_cfg = SessionConfig(
        sample_path=str(small),
        ledger_path=str(state / "crash.sqlite"),
        snapshot_dir=str(state / "crash_snap"),
        scale=10_000, top_k=8, batch_size=1_000,
    )
    victim = ProfileSession(crash_cfg)
    try:
        victim.run(crash_after_batches=2)
    except ForcedKill as exc:
        print("simulated kill:", exc)
    info = victim.recover()
    print("recovery info:", json.dumps(info.__dict__, indent=2))
    victim.close()

    resumed = ProfileSession(crash_cfg)
    print("resume summary:", json.dumps(resumed.recover().__dict__, indent=2))
    resume_outcome = resumed.resume()
    print("after resume:", json.dumps(resume_outcome, indent=2,
                                      sort_keys=True))
    resumed.close()

    section("7. historical replay verification")
    run_cli([
        "replay", "--samples", str(small),
        "--ledger", str(state / "ledger.sqlite"),
        "--snapshot-dir", str(state / "snap"),
    ])

    section("8. interval windowing and snapshot replacement on version change")
    windowed = SessionConfig(
        sample_path=str(small),
        ledger_path=str(state / "window.sqlite"),
        snapshot_dir=str(state / "window_snap"),
        start_ms=1_700_000_000_500, end_ms=1_700_000_010_000,
    )
    win_session = ProfileSession(windowed)
    win_summary = win_session.run()
    print("windowed run:", json.dumps(win_summary, indent=2,
                                      sort_keys=True))
    win_session.close()

    if not quick:
        section("9. scale run: 150,000 rows, streaming path, reproducibility")
        big = data / "samples_150k.ndjson"
        t0 = time.perf_counter()
        _, big_summary = generate(
            rows=150_000, services=4, min_depth=4, max_depth=12,
            weight_mode="skewed", seed=99, output=str(big))
        gen_elapsed = time.perf_counter() - t0
        print("generation:", json.dumps(big_summary, sort_keys=True))
        print(f"generation seconds: {gen_elapsed:.2f}")

        big_cfg = SessionConfig(
            sample_path=str(big),
            ledger_path=str(state / "big.sqlite"),
            snapshot_dir=str(state / "big_snap"),
            scale=10_000, top_k=10, batch_size=20_000,
            max_unique_stacks=5_000, max_total_weight=200_000,
        )
        big_session = ProfileSession(big_cfg)
        t1 = time.perf_counter()
        big_session.run()
        run_elapsed = time.perf_counter() - t1
        print(f"profile seconds: {run_elapsed:.2f}")
        print("chosen path:", big_session.result.path_used)
        print("path reason:", big_session.result.path_reason)
        print("unique stacks:", big_session.result.unique_stack_count,
              "total weight:", big_session.result.total_weight)
        same_big, msg_big = cross_check(
            load_samples(big),
            big_cfg.max_unique_stacks, big_cfg.max_total_weight)
        print(f"big dual-path identical: {same_big} ({msg_big})")
        replay = big_session.replay()
        print("big replay:", json.dumps(replay, sort_keys=True))
        big_session.close()

        # Reproducibility: regenerate and compare digest.
        big_copy = data / "samples_150k_copy.ndjson"
        generate(rows=150_000, services=4, min_depth=4, max_depth=12,
                 weight_mode="skewed", seed=99, output=str(big_copy))
        from slowprof.samples import sample_stream_digest
        print("digest stable across regeneration:",
              sample_stream_digest(big) == sample_stream_digest(big_copy))

    section("diagnostics rendering mode")
    print("tkinter available:", tkinter_available(),
          "-> use 'python -m slowprof.cli diagnose ... "
          "--text' here; a real window opens on a tk-enabled Windows Python")
    print("\nAll demo stages completed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    sink = io.StringIO()
    with contextlib.redirect_stdout(
            _TeeStream(sys.stdout, sink)):
        code = main(args.quick)
    out = ROOT / "state" / "demo_output.txt"
    out.write_text(sink.getvalue(), encoding="utf-8")
    print(f"\n[tee] full output saved to {out}")
    sys.exit(code)
