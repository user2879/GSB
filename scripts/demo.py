from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ossdiag.cli import main as cli_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="一键运行本地诊断演示")
    parser.add_argument("--spec", default=str(ROOT / "samples" / "small_spec.json"))
    parser.add_argument("--workdir", default=str(ROOT / ".demo-work"))
    parser.add_argument("--large", action="store_true", help="使用 289 MiB 规模样本")
    parser.add_argument("--fault", choices=("lease", "digest", "snapshot"), help="只演示一类异常")
    parser.add_argument("--keep", action="store_true", help="保留已有运行目录")
    args = parser.parse_args()
    spec = str(ROOT / "samples" / "demo_spec.json") if args.large else args.spec
    workdir = Path(args.workdir)
    if workdir.exists() and not args.keep:
        shutil.rmtree(workdir)
    argv = ["run", "--spec", spec, "--workdir", str(workdir)]
    if args.fault == "lease":
        argv.append("--inject-lease-overflow")
    elif args.fault == "digest":
        argv.append("--inject-digest-mismatch")
    elif args.fault == "snapshot":
        argv.append("--inject-snapshot-conflict")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
