"""Run the deterministic three-scenario demo using samples/demo_profile.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlstrust.demo import run_demo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="data/demo")
    parser.add_argument("--profile", default=str(PROJECT_ROOT / "samples" / "demo_profile.json"))
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--key-bits", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    family = profile["family"]
    summary = run_demo(
        PROJECT_ROOT / args.workdir,
        fresh=not args.reuse,
        key_bits=args.key_bits or int(family["key_bits"]),
        port=family["port"] if args.port is None else args.port,
        roots=int(family["roots"]),
        intermediate_levels=int(family["intermediate_levels"]),
        leaves=int(family["leaves"]),
        validity_days=int(family["validity_days"]),
        seed=str(family["seed"]),
        not_before_ms=int(family["not_before_ms"]),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
