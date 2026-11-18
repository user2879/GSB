"""A local coordinator node subprocess driven entirely by the TCP protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .client import request_json
from .util import canonical_json


def emit(event: str, **fields: Any) -> None:
    print(canonical_json({"event": event, **fields}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local coordinator node worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28765)
    parser.add_argument("--member", required=True)
    parser.add_argument("--heartbeat-ms", type=int, default=300)
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        emit("node_rejected", code="E_PROTO", reason="non-loopback host", host=args.host)
        return 2
    if args.heartbeat_ms <= 0:
        emit("node_rejected", code="E_PROTO", reason="heartbeat-ms must be positive")
        return 2

    deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None
    emit(
        "node_started",
        member=args.member,
        host=args.host,
        port=args.port,
        recovery_source="tcp_persisted_state",
        forced_exit_report_available=False,
    )
    last_vote_key: tuple[int, int] | None = None
    last_leader: str | None = None
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                state = request_json(
                    args.host,
                    args.port,
                    {"op": "query_state"},
                    timeout=1.5,
                )
                if not state.get("ok"):
                    emit("node_query_error", member=args.member, response=state)
                    time.sleep(args.heartbeat_ms / 1000)
                    continue
                member = state.get("members", {}).get(args.member)
                if member is None:
                    emit("member_absent", member=args.member, state_version=state.get("state_version"))
                    time.sleep(args.heartbeat_ms / 1000)
                    continue
                term = int(state["term"])
                lease_version = int(member["lease_version"])
                heartbeat = request_json(
                    args.host,
                    args.port,
                    {
                        "op": "heartbeat",
                        "term": term,
                        "member": args.member,
                        "lease_version": lease_version,
                    },
                    timeout=1.5,
                )
                if heartbeat.get("ok"):
                    emit(
                        "heartbeat",
                        member=args.member,
                        term=term,
                        lease_version=heartbeat["lease_version"],
                        state_version=heartbeat["state_version"],
                    )
                else:
                    emit(
                        "heartbeat_denied",
                        member=args.member,
                        code=heartbeat.get("code"),
                        details=heartbeat.get("details", {}),
                    )

                vote = request_json(
                    args.host,
                    args.port,
                    {
                        "op": "vote",
                        "term": term,
                        "member": args.member,
                        "candidate": args.member,
                    },
                    timeout=1.5,
                )
                vote_key = (term, int(vote.get("state_version", -1)))
                if vote.get("ok") and (last_vote_key is None or vote_key != last_vote_key):
                    emit(
                        "vote",
                        member=args.member,
                        term=term,
                        candidate=args.member,
                        granted=vote.get("granted"),
                        idempotent=vote.get("idempotent"),
                    )
                    last_vote_key = vote_key
                elif not vote.get("ok"):
                    emit(
                        "vote_denied",
                        member=args.member,
                        code=vote.get("code"),
                        details=vote.get("details", {}),
                    )

                leader = request_json(
                    args.host,
                    args.port,
                    {"op": "query_leader"},
                    timeout=1.5,
                )
                current_leader = leader.get("leader") if leader.get("ok") else None
                if current_leader != last_leader or leader.get("requires_rerun"):
                    emit(
                        "leader_view",
                        member=args.member,
                        term=leader.get("term"),
                        leader=current_leader,
                        election=leader.get("election"),
                        requires_rerun=leader.get("requires_rerun"),
                    )
                    last_leader = current_leader
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                emit("node_transport_error", member=args.member, message=str(exc))
            if args.once:
                break
            time.sleep(args.heartbeat_ms / 1000)
    except KeyboardInterrupt:
        emit("node_graceful_interrupt", member=args.member)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

