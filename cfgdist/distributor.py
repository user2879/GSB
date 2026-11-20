"""Parallel per-node dispatch, retry handling and convergence verdict."""

from __future__ import annotations

import socket
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import STATE_DISTRIBUTING, STATE_FAILED
from .errors import BizError, ErrorCode
from .logging_setup import StructuredLogger
from .planner import expected_digest_map
from .protocol import request
from .session import SessionStore


@dataclass
class NodeDispatchResult:
    node: str
    ok: bool
    attempts: int = 0
    last_receipt: dict | None = None
    history: list[dict] = field(default_factory=list)
    error: str | None = None


def _dispatch_one_node(
    node_spec: dict,
    slice_payload: dict,
    derivations: list[dict],
    config_version: int,
    logger: StructuredLogger | None,
) -> NodeDispatchResult:
    node_id = node_spec["id"]
    host = node_spec["host"]
    port = node_spec["port"]
    entries = slice_payload["entries"]
    strategies = [item["strategy"] for item in entries]
    timeout_ms = max(item["timeout_ms"] for item in strategies) if strategies else 2000
    retries = max(item["retries"] for item in strategies) if strategies else 2

    result = NodeDispatchResult(node=node_id, ok=False)
    for attempt in range(1, retries + 2):
        message = {
            "kind": "dispatch",
            "node": node_id,
            "config_version": config_version,
            "attempt": attempt,
            "entries": entries,
            "derivations": derivations,
        }
        try:
            receipt = request(host, port, message, timeout_ms=timeout_ms)
        except (socket.timeout, TimeoutError, OSError) as exc:
            timeout_record = {
                "ok": False,
                "node": node_id,
                "config_version": config_version,
                "attempt": attempt,
                "retry": attempt > 1,
                "error": f"timeout/unreachable: {exc.__class__.__name__}",
            }
            result.history.append(timeout_record)
            result.attempts = attempt
            result.error = timeout_record["error"]
            if logger is not None:
                logger.log(
                    "dispatch.attempt_failed",
                    node=node_id,
                    attempt=attempt,
                    reason="timeout",
                )
            if attempt > retries:
                return result
            continue

        receipt.setdefault("retry", attempt > 1)
        receipt.setdefault("duplicate", False)
        result.history.append(receipt)
        result.attempts = attempt
        result.last_receipt = receipt
        if logger is not None:
            logger.log(
                "dispatch.attempt",
                node=node_id,
                attempt=attempt,
                ok=bool(receipt.get("ok")),
                retry=bool(receipt.get("retry")),
            )
        if receipt.get("ok"):
            result.ok = True
            return result
        result.error = receipt.get("error", "node reported failure")
        if attempt > retries:
            return result
    return result


def distribute(
    plan: dict,
    node_specs: list[dict],
    store: SessionStore,
    logger: StructuredLogger | None = None,
    max_concurrency: int = 4,
    session_id: str | None = None,
) -> dict:
    """Distribute a plan and record every attempt/receipt in the session DB."""
    node_by_id = {spec["id"]: spec for spec in node_specs}
    expected = expected_digest_map(plan)
    config_version = plan["config_version"]

    existing = store.is_committed(config_version)
    if existing is not None:
        if logger is not None:
            logger.biz_error(
                "dispatch.idempotent",
                ErrorCode.E_IDEMPOTENT,
                "config version already committed; no second dispatch",
                config_version=config_version,
            )
        raise BizError(
            ErrorCode.E_IDEMPOTENT,
            f"config version {config_version} is already committed",
            {
                "config_version": config_version,
                "session_id": existing["session_id"],
                "committed_at": existing["committed_at"],
            },
        )

    session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    store.create_session(session_id, config_version, plan)
    store.transition(session_id, STATE_DISTRIBUTING, [])

    derivations = plan["derivations"]
    results: dict[str, NodeDispatchResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        futures = {
            pool.submit(
                _dispatch_one_node,
                node_by_id[slice_payload["node"]],
                slice_payload,
                derivations,
                config_version,
                logger,
            ): slice_payload["node"]
            for slice_payload in plan["slices"]
        }
        for future in as_completed(futures):
            node_id = futures[future]
            outcome = future.result()
            results[node_id] = outcome
            for attempt_record in outcome.history:
                store.record_receipt(session_id, node_id, attempt_record.get("attempt", 1), attempt_record)

    confirmed = sorted(node_id for node_id, outcome in results.items() if outcome.ok)
    convergence = evaluate_convergence(plan, results)
    if not convergence["converged"]:
        store.transition(session_id, STATE_FAILED, confirmed)
        if logger is not None:
            logger.biz_error(
                "dispatch.converge_failed",
                ErrorCode.E_CONVERGE,
                "receipts did not converge to expected derived digests",
                deviations=convergence["deviations"],
            )
        raise BizError(
            ErrorCode.E_CONVERGE,
            "distribution failed to converge",
            {"session_id": session_id, "deviations": convergence["deviations"]},
        )

    commit_outcome = store.commit(session_id)
    if logger is not None:
        logger.log(
            "dispatch.committed",
            session_id=session_id,
            config_version=config_version,
            confirmed_nodes=confirmed,
        )
    return {
        "session_id": session_id,
        "config_version": config_version,
        "converged": True,
        "confirmed_nodes": confirmed,
        "results": {
            node_id: {
                "ok": outcome.ok,
                "attempts": outcome.attempts,
                "retry_used": outcome.attempts > 1,
                "receipt": outcome.last_receipt,
            }
            for node_id, outcome in sorted(results.items())
        },
        "commit": {
            "idempotent": commit_outcome["idempotent"],
            "committed_at": commit_outcome["committed"]["committed_at"],
        },
    }


def evaluate_convergence(plan: dict, results: dict[str, NodeDispatchResult]) -> dict:
    """Order-independent stable convergence verdict."""
    expected = expected_digest_map(plan)
    deviations = []
    # Stable node ordering: use plan slice order, not future completion order.
    for slice_payload in plan["slices"]:
        node_id = slice_payload["node"]
        want = expected[node_id]
        outcome = results.get(node_id)
        if outcome is None or not outcome.ok or outcome.last_receipt is None:
            deviations.append(
                {
                    "node": node_id,
                    "expected_digest": want,
                    "actual_digest": None,
                    "reason": "no successful receipt",
                }
            )
            continue
        actual = outcome.last_receipt.get("digest")
        if actual != want:
            deviations.append(
                {
                    "node": node_id,
                    "expected_digest": want,
                    "actual_digest": actual,
                    "reason": "digest mismatch",
                }
            )
    return {
        "converged": not deviations,
        "deviations": deviations,
    }
