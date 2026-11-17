"""Convenience entry point for the deterministic local demonstration."""

from pathlib import Path

from autoscale_diag.demo import run_demo


if __name__ == "__main__":
    import json
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/demo")
    result = run_demo(target)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result["ok"] else 1)

