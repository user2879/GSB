"""Command-line entry point for the deterministic workbench demo.

Usage (from the project root, virtualenv activated)::

    python scripts/run_demo.py --output-dir artifacts

Running it twice must produce byte-identical WAV, labels and report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loudness_workbench.demo import run_demo  # noqa: E402
from loudness_workbench.errors import WorkbenchError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loudness workbench demo")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.output_dir, args.seed)
    except WorkbenchError as exc:
        print(f"BUSINESS ERROR {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    verdict = report["tolerance_verdict"]
    print("=== Deterministic workbench demo ===")
    print(f"integrated loudness : streaming={report['streaming']['conclusions']['integrated_lkfs']} "
          f"offline={report['offline']['integrated_lkfs']} LKFS")
    print(f"true peak           : {report['offline']['true_peak_dbtp']:.3f} dBTP")
    print(f"loudness range      : {report['offline']['loudness_range_lu']:.3f} LU")
    print(f"tolerance verdict   : {verdict}")
    print(f"degraded frames     : {report['budget']['frames_degraded']}")
    print(f"stale snapshot      : {report['stale_snapshot_rejected']}")
    return 0 if verdict["all_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
