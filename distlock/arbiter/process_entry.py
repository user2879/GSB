"""Entry point executed inside each per-domain arbiter subprocess."""
from __future__ import annotations

import json
import sys

from ..clock import MonotonicClock
from ..models import Domain, ScaleFactor
from ..protocol import serve_arbiter
from .state_machine import DomainArbiter


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Single JSON argument describing the domain + bind information.
    config = json.loads(argv[0])
    domain = Domain(
        domain_id=config["domain_id"],
        capacity=int(config["capacity"]),
        lease_duration_ms=int(config["lease_duration_ms"]),
        renew_interval_ms=int(config["renew_interval_ms"]),
        queue_limit=int(config["queue_limit"]),
        version=int(config.get("version", 1)),
    )
    scale = ScaleFactor(int(config["scale_factor"]))
    arbiter = DomainArbiter(domain, scale, MonotonicClock())
    serve_arbiter(
        arbiter,
        host=config.get("host", "127.0.0.1"),
        port=int(config["port"]),
        domain_set_version=int(config["domain_set_version"]),
        ready_path=config.get("ready_path"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
