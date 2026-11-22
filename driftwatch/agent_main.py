from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_runtime import AgentRuntime, spec_from_resolved_dict
from .errors import DriftError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback observation agent")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        with Path(args.spec).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        spec = spec_from_resolved_dict(raw)
        runtime = AgentRuntime(spec=spec, parent_pid=args.parent_pid)
        runtime.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, DriftError) else "E_PORT"
        print(json.dumps({"ok": False, "code": code, "message": str(exc)}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
