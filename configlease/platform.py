"""Configuration center platform.

Responsibilities:
  * launch registered nodes as independent local OS child processes, each on a
    unique 127.0.0.1 high port, and complete a real socket handshake;
  * issue leases (TTL + revision window + covered keys) and arbitrate parallel
    lease requests under a global dispatch budget and per-node revision budget;
  * serialize same-key concurrent claims by stable (node_id, rev_seq) order;
  * verify every node apply result against a recomputed expected view;
  * roll back revisions, revoke affected leases and replace node views whole;
  * persist revisions/leases/cursors/budgets/events/snapshots in SQLite and
    recover from forced termination using only fully committed records.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clock import deadline_from_ttl, is_expired, now_ms
from .engine import (
    SNAPSHOT_VERSION,
    BudgetPool,
    KeyConflictSerializer,
    assert_snapshot_compatible,
    canonical_json,
    content_hash,
    expected_node_view,
    replay_lease_records,
    validate_request,
    verify_apply_result,
)
from .errors import (
    ApplyMismatchError,
    HandshakeTimeoutError,
    LeaseError,
    NodeStartError,
    PortInUseError,
    RecoverError,
    RollbackRevisionError,
    VersionConflictError,
)
from .manifest import Revision, fold_view
from .protocol import read_message, write_message
from .sample import summary_for_revision
from .storage import Store
from .structured_log import StructuredLogger

NODE_MODULE = "configlease.node"


def _wire_error_to_exception(payload: dict[str, Any]) -> LeaseError:
    from .errors import (
        KeyNotCoveredError,
        LeaseExpiredError,
        RevisionOutOfRangeError,
        RollbackRevisionError,
    )
    code = payload.get("code", "E_UNKNOWN")
    mapping = {
        "E_KEY": KeyNotCoveredError,
        "E_REV": RevisionOutOfRangeError,
        "E_ROLLBACK": RollbackRevisionError,
        "E_EXPIRED": LeaseExpiredError,
    }
    cls = mapping.get(code, LeaseError)
    exc = cls(payload.get("message", code),
              **{k: v for k, v in payload.items() if k not in {"type", "code", "message"}})
    return exc

"configlease.node"


# --------------------------------------------------------------------------- #
# Node child process + JSON-lines client
# --------------------------------------------------------------------------- #
@dataclass
class NodeSpec:
    node_id: str
    port: int
    host: str = "127.0.0.1"


class NodeProcess:
    """A node running as an independent child process on a loopback port."""

    def __init__(self, spec: NodeSpec, python_exe: str | None = None):
        self.spec = spec
        self.python_exe = python_exe or sys.executable
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.bound_port: int | None = None
        self._request_lock = asyncio.Lock()

    async def start(self, handshake_timeout: float = 5.0,
                    startup_timeout: float = 10.0) -> dict[str, Any]:
        port = self.spec.port
        if await self._port_in_use(port):
            raise PortInUseError(
                "node port already occupied", node_id=self.spec.node_id, port=port,
            )
        cmd = [
            self.python_exe, "-u", "-m", NODE_MODULE,
            "--node-id", self.spec.node_id,
            "--host", self.spec.host,
            "--port", str(port),
        ]
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise NodeStartError(
                "failed to spawn node process", node_id=self.spec.node_id,
                detail=str(exc),
            ) from exc

        # A bind failure surfaces as a BIND_ERROR stdout line and exit code 2.
        async def _watch_startup() -> str:
            assert self.proc is not None and self.proc.stdout is not None
            line = await self.proc.stdout.readline()
            return line.decode("utf-8", errors="replace").strip()

        try:
            first_line = await asyncio.wait_for(_watch_startup(), timeout=startup_timeout)
        except asyncio.TimeoutError as exc:
            await self.kill()
            raise NodeStartError(
                "node produced no startup line", node_id=self.spec.node_id,
            ) from exc

        if first_line.startswith("BIND_ERROR") or self.proc.returncode is not None:
            stderr = await self._drain_stderr()
            await self.kill()
            if "ADDRINUSE" in first_line or "in use" in first_line.lower():
                raise PortInUseError(
                    "node reported bind failure (port in use)",
                    node_id=self.spec.node_id, port=port, detail=first_line,
                )
            raise NodeStartError(
                "node exited during startup",
                node_id=self.spec.node_id, detail=first_line or stderr,
            )
        if not first_line.startswith("LISTENING"):
            await self.kill()
            raise NodeStartError(
                "unexpected node startup line",
                node_id=self.spec.node_id, detail=first_line,
            )
        self.bound_port = int(first_line.split()[1])
        await self._handshake(handshake_timeout)
        return {"node_id": self.spec.node_id, "port": self.bound_port, "pid": self.proc.pid}

    async def _handshake(self, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        # Real socket timeout: connect/read deadlines come from the event loop,
        # not from an artificial sleep.
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.spec.host, self.bound_port),
                timeout=max(0.01, deadline - asyncio.get_running_loop().time()),
            )
            ready = await asyncio.wait_for(
                read_message(self.reader),
                timeout=max(0.01, deadline - asyncio.get_running_loop().time()),
            )
        except (asyncio.TimeoutError, TimeoutError, OSError) as exc:
            await self.kill()
            raise HandshakeTimeoutError(
                "real socket handshake timeout",
                node_id=self.spec.node_id, port=self.bound_port,
                timeout_s=timeout,
            ) from exc
        if ready.get("type") != "ready" or ready.get("node_id") != self.spec.node_id:
            await self.kill()
            raise NodeStartError(
                "invalid handshake payload", node_id=self.spec.node_id, got=ready,
            )

    async def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.writer is None or self.reader is None:
            raise NodeStartError("node not connected", node_id=self.spec.node_id)
        # One persistent connection per node: serialize request/response pairs
        # so concurrent arbitrations never interleave reads on the same socket.
        async with self._request_lock:
            await write_message(self.writer, message)
            return await read_message(self.reader)

    async def ephemeral_request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Open a fresh connection for one request/response pair.

        Parallel arbitrations use independent connections so a node never blocks
        another arbitration while waiting on the shared serializer, removing the
        nested-lock deadlock.  The node keeps one shared state across sockets.
        """
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.spec.host, self.bound_port),
            timeout=5.0,
        )
        try:
            await read_message(reader)  # consume the per-connection ready frame
            await write_message(writer, message)
            return await read_message(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def kill(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    async def _drain_stderr(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            raw = await asyncio.wait_for(self.proc.stderr.read(), timeout=1)
            return raw.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            return ""

    @staticmethod
    async def _port_in_use(port: int) -> bool:
        if port <= 0:
            return False
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=0.5,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, TimeoutError, ConnectionRefusedError, OSError):
            return False


# --------------------------------------------------------------------------- #
# Platform
# --------------------------------------------------------------------------- #
@dataclass
class PlatformConfig:
    db_path: str
    log_dir: str
    session_id: str
    global_budget: int = 8
    node_budgets: dict[str, int] = field(default_factory=dict)
    lease_ttl_ms: int = 10_000
    arbitration_timeout_s: float = 3.0
    arbitration_retries: int = 2
    handshake_timeout_s: float = 5.0


class Platform:
    def __init__(self, config: PlatformConfig,
                 revisions: list[Revision],
                 node_specs: list[NodeSpec],
                 store: Store | None = None,
                 logger: StructuredLogger | None = None):
        self.config = config
        self.revisions = {r.rev_seq: r for r in revisions}
        self.revision_order = [r.rev_seq for r in revisions]
        self.node_specs = {n.node_id: n for n in node_specs}
        self.store = store or Store(config.db_path)
        self.logger = logger or StructuredLogger(config.log_dir, config.session_id)
        self.nodes: dict[str, NodeProcess] = {}
        self.superseded: set[int] = set(self.store.superseded_sequences())
        self.cursors: dict[str, int] = {}
        self.views: dict[str, dict[str, int]] = {}
        self.budgets = BudgetPool(
            config.global_budget,
            config.node_budgets or {n.node_id: 4 for n in node_specs},
        )
        self.serializer = KeyConflictSerializer()
        self._request_seq = 0
        self._restore_runtime_state()
        self._mirror_log_to_store()

    # ---- lifecycle -------------------------------------------------------
    async def start_nodes(self) -> list[dict[str, Any]]:
        started = []
        for spec in self.node_specs.values():
            node = NodeProcess(spec)
            info = await node.start(handshake_timeout=self.config.handshake_timeout_s)
            self.nodes[spec.node_id] = node
            started.append(info)
            self._log("node_started", **info)
        return started

    async def stop_nodes(self) -> None:
        for node in self.nodes.values():
            await node.kill()
        self.nodes.clear()

    async def close(self) -> None:
        await self.stop_nodes()
        self.logger.close()
        self.store.close()

    # ---- internal state --------------------------------------------------
    def _restore_runtime_state(self) -> None:
        for row in self.store.list_cursors().values():
            self.cursors[row["node_id"]] = row["applied_rev"]
            self.views[row["node_id"]] = dict(row["view"])
        for budget_row in self.store.list_budgets():
            if budget_row["kind"] == "global":
                self.budgets.global_budget.limit = budget_row["limit_value"]
                self.budgets.global_budget.consumed = budget_row["consumed"]
            else:
                node_id = budget_row["name"].split(":", 1)[1]
                budget = self.budgets.node_budget(node_id)
                budget.limit = budget_row["limit_value"]
                budget.consumed = budget_row["consumed"]

    def _mirror_log_to_store(self) -> None:
        original = self.logger.event

        def event_and_persist(event_type: str, **fields: Any) -> dict[str, Any]:
            record = original(event_type, **fields)
            try:
                self.store.insert_event(record)
            except Exception:  # pragma: no cover - logging must not crash flow
                pass
            return record

        self.logger.event = event_and_persist  # type: ignore[method-assign]

    def _log(self, event_type: str, **fields: Any) -> dict[str, Any]:
        return self.logger.event(event_type, **fields)

    def _log_error(self, exc: LeaseError, **extra: Any) -> dict[str, Any]:
        fields = dict(exc.context)
        fields.update(extra)
        return self.logger.error(exc.code, message=exc.message, **fields)

    def _next_request_id(self) -> str:
        self._request_seq += 1
        return f"req-{self.config.session_id}-{self._request_seq:06d}"

    def _revision(self, rev_seq: int) -> Revision:
        if rev_seq not in self.revisions:
            raise RollbackRevisionError("unknown revision", rev_seq=rev_seq)
        return self.revisions[rev_seq]

    # ---- lease issuance --------------------------------------------------
    def issue_lease(self, node_id: str, rev_seq: int, low_seq: int, high_seq: int,
                    keys: set[str], ttl_ms: int | None = None,
                    request_id: str | None = None) -> dict[str, Any]:
        if node_id not in self.node_specs:
            raise NodeStartError("unknown node", node_id=node_id)
        ttl = self.config.lease_ttl_ms if ttl_ms is None else ttl_ms
        lease = {
            "lease_id": f"lease-{uuid.uuid4().hex[:12]}",
            "request_id": request_id or self._next_request_id(),
            "node_id": node_id,
            "rev_seq": int(rev_seq),
            "low_seq": int(low_seq),
            "high_seq": int(high_seq),
            "keys": sorted(set(keys)),
            "issued_ms": now_ms(),
            "deadline_ms": deadline_from_ttl(ttl),
            "ttl_ms": int(ttl),
        }
        self.store.put_lease(lease, status="active")
        self._log("lease_issued", node_id=node_id, rev_seq=rev_seq,
                  lease_id=lease["lease_id"], request_id=lease["request_id"],
                  low_seq=lease["low_seq"], high_seq=lease["high_seq"],
                  keys=lease["keys"], deadline_ms=lease["deadline_ms"])
        return lease

    # ---- parallel arbitration -------------------------------------------
    async def dispatch_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Arbitrate many lease applications in parallel."""
        tasks = [self.dispatch(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """One full lease request: issue, arbitrate, apply and verify.

        Retries only occur after a genuine arbitration timeout.  The original
        ``request_id`` and ``rev_seq`` are preserved across every retry, so a
        retry never silently becomes a different change.
        """
        node_id = request["node_id"]
        rev_seq = int(request["rev_seq"])
        request_id = request.get("request_id") or self._next_request_id()
        request = dict(request)
        request["request_id"] = request_id
        revision = self._revision(rev_seq)
        lease = self.issue_lease(
            node_id=node_id,
            rev_seq=rev_seq,
            low_seq=int(request.get("low_seq", rev_seq)),
            high_seq=int(request.get("high_seq", rev_seq)),
            keys=set(request.get("keys", revision.changes.keys())),
            ttl_ms=request.get("ttl_ms"),
            request_id=request_id,
        )
        last_error: LeaseError | None = None
        for attempt in range(self.config.arbitration_retries + 1):
            try:
                return await self._arbitrate_once(request, lease, revision, attempt)
            except asyncio.TimeoutError as exc:
                last_error = LeaseError("arbitration timeout", code="E_TIMEOUT")
                last_error.code = "E_TIMEOUT"
                self._log(
                    "arbitration_timeout", node_id=node_id, rev_seq=rev_seq,
                    request_id=request_id, lease_id=lease["lease_id"], attempt=attempt,
                )
                if attempt >= self.config.arbitration_retries:
                    self.store.update_lease_status(lease["lease_id"], "rejected", "E_TIMEOUT")
                    raise
                # Retry keeps the same request id, revision and lease id.
                await asyncio.sleep(0)
        # Unreachable, kept for type completeness.
        assert last_error is not None
        raise last_error

    async def _arbitrate_once(self, request: dict[str, Any], lease: dict[str, Any],
                              revision: Revision, attempt: int) -> dict[str, Any]:
        node_id = request["node_id"]
        rev_seq = int(request["rev_seq"])
        keys = set(lease["keys"])
        try:
            # Rule checks happen before any budget is spent.
            validate_request(
                {"rev_seq": rev_seq, "changes": revision.changes},
                lease, self.superseded,
            )

            # Dual budget: global dispatch token first, then per-node token.
            # Both queues are FIFO, so exhausted budgets cause real queuing and
            # no request can jump the queue.
            await self.budgets.global_budget.acquire()
            node_budget = self.budgets.node_budget(node_id)
            await node_budget.acquire()
            self._persist_budgets()

            # Same-key conflicts serialize on the stable (node_id, rev_seq)
            # rule, bounded by a real timeout.
            try:
                await asyncio.wait_for(
                    self.serializer.claim(node_id, rev_seq, keys),
                    timeout=self.config.arbitration_timeout_s,
                )
            except asyncio.TimeoutError:
                await self.budgets.global_budget.refund()
                await node_budget.refund()
                self._persist_budgets()
                raise

            try:
                result = await self._grant_and_apply(request, lease, revision)
                # The global dispatch budget bounds in-flight dispatches, so its
                # token is released once the lease reaches a terminal state.
                # The per-node revision budget is cumulative: an applied
                # revision keeps its token and counts toward the node's cap.
                await self.budgets.global_budget.refund()
                self._persist_budgets()
                return result
            except LeaseError:
                # Rule/lease rejection: the revision was not applied, refund
                # both tokens and surface the structured E_* rejection.
                await self.budgets.global_budget.refund()
                await node_budget.refund()
                self._persist_budgets()
                raise
            except Exception:
                # Runtime failure: nothing was consumed, refund both tokens.
                await self.budgets.global_budget.refund()
                await node_budget.refund()
                self._persist_budgets()
                raise
            finally:
                await self.serializer.release(node_id, rev_seq, keys)
        except LeaseError as exc:
            # Rule rejections never move the node cursor and never consume
            # budget.  Record the structured reason for the audit chain.
            self.store.update_lease_status(lease["lease_id"], "rejected", exc.code)
            self._log_error(exc, node_id=node_id, rev_seq=rev_seq,
                            request_id=request["request_id"],
                            lease_id=lease["lease_id"], attempt=attempt)
            raise

    async def _grant_and_apply(self, request: dict[str, Any], lease: dict[str, Any],
                               revision: Revision) -> dict[str, Any]:
        node_id = request["node_id"]
        node = self.nodes[node_id]
        grant = await node.ephemeral_request({"type": "lease_grant", "lease": lease})
        if grant.get("type") == "error":
            exc = _wire_error_to_exception(grant)
            exc.context.update({"node_id": node_id, "rev_seq": revision.rev_seq,
                                "lease_id": lease["lease_id"],
                                "request_id": lease.get("request_id")})
            raise exc

        apply_msg = {
            "type": "apply",
            "lease_id": lease["lease_id"],
            "rev_seq": revision.rev_seq,
            "changes": revision.changes,
        }
        node_response = await node.ephemeral_request(apply_msg)
        if node_response.get("type") == "error":
            exc = _wire_error_to_exception(node_response)
            exc.context.update({"node_id": node_id, "rev_seq": revision.rev_seq,
                                "lease_id": lease["lease_id"],
                                "request_id": lease.get("request_id")})
            raise exc

        prior_view = dict(self.views.get(node_id, {}))
        prior_cursor = self.cursors.get(node_id, 0)
        verify_request = {"rev_seq": revision.rev_seq,
                          "changes": revision.changes,
                          "node_id": node_id}
        verified = verify_apply_result(verify_request, lease, node_response,
                                       prior_view, prior_cursor)

        # Commit the whole transition atomically: lease terminal state, cursor,
        # view and the per-node budget consumption.
        lease_record = dict(lease)
        lease_record["summary"] = verified["summary"]
        self.store.update_lease_status(lease["lease_id"], "applied")
        self.store.put_cursor(node_id, verified["applied_rev"], verified["view"])
        self.cursors[node_id] = verified["applied_rev"]
        self.views[node_id] = verified["view"]
        self._persist_budgets()
        self._log(
            "revision_applied", node_id=node_id, rev_seq=revision.rev_seq,
            lease_id=lease["lease_id"], request_id=lease["request_id"],
            applied_rev=verified["applied_rev"], summary=verified["summary"],
            global_budget=self.budgets.global_budget.snapshot(),
            node_budget=self.budgets.node_budget(node_id).snapshot(),
        )
        return {
            "status": "applied",
            "node_id": node_id,
            "rev_seq": revision.rev_seq,
            "lease_id": lease["lease_id"],
            "request_id": lease["request_id"],
            "applied_rev": verified["applied_rev"],
            "view": verified["view"],
            "summary": verified["summary"],
        }

    def _persist_budgets(self) -> None:
        g = self.budgets.global_budget
        self.store.put_budget(g.name, g.kind, g.limit, g.consumed)
        for node_id, budget in self.budgets.node_budgets.items():
            self.store.put_budget(budget.name, budget.kind, budget.limit, budget.consumed)

    # ---- rollback --------------------------------------------------------
    def next_rev_seq(self) -> int:
        return (max(self.revision_order) if self.revision_order else 0) + 1

    async def rollback(self, target_seq: int) -> dict[str, Any]:
        """Roll back to a historical revision.

        A brand new revision with a monotonically increasing sequence is
        created; the revisions up to the target are the rebuild basis.  All
        unexpired leases whose window intersects the rolled-back range are
        revoked whole and marked E_ROLLBACK; every node view is replaced, never
        incrementally merged.
        """
        if target_seq not in self.revisions:
            raise RollbackRevisionError(
                "rollback target unknown", rev_seq=target_seq,
            )
        new_seq = self.next_rev_seq()
        basis = [self.revisions[s] for s in self.revision_order if s <= target_seq]
        rebuilt_view = fold_view(basis)
        rollback_rev = Revision(
            rev_seq=new_seq,
            parent=self.revision_order[-1] if self.revision_order else 0,
            changes=dict(rebuilt_view),
            kind="rollback",
            rollback_target=target_seq,
        )
        self.revisions[new_seq] = rollback_rev
        self.revision_order.append(new_seq)
        self.store.upsert_revision(rollback_rev.to_dict())

        # Mark every revision strictly after the rollback target as superseded.
        rolled_back_seqs = [s for s in self.revision_order
                            if target_seq < s < new_seq]
        for seq in rolled_back_seqs:
            self.superseded.add(seq)
            self.store.mark_revision_superseded(seq)

        affected_leases: list[dict[str, Any]] = []
        now = now_ms()
        for lease in self.store.active_leases():
            window = set(range(lease["low_seq"], lease["high_seq"] + 1))
            if window & set(rolled_back_seqs) and lease["deadline_ms"] > now:
                self.store.update_lease_status(
                    lease["lease_id"], "revoked", "E_ROLLBACK",
                )
                affected_leases.append(lease)
                self._log(
                    "lease_revoked", code="E_ROLLBACK", node_id=lease["node_id"],
                    rev_seq=lease["rev_seq"], lease_id=lease["lease_id"],
                    rolled_back_seqs=rolled_back_seqs,
                )

        # Whole-view replacement for every registered node.
        for node_id in self.node_specs:
            self.cursors[node_id] = target_seq
            self.views[node_id] = dict(rebuilt_view)
            self.store.put_cursor(node_id, target_seq, rebuilt_view)
        for node_id, node in self.nodes.items():
            ack = await node.request({
                "type": "replace_view",
                "view": rebuilt_view,
                "applied_rev": target_seq,
                "clear_lease": True,
            })
            if ack.get("type") != "replace_ack":
                raise ApplyMismatchError(
                    "node rejected whole-view replacement",
                    node_id=node_id, got=ack,
                )

        result = {
            "status": "rolled_back",
            "new_rev_seq": new_seq,
            "target_seq": target_seq,
            "rolled_back_seqs": rolled_back_seqs,
            "basis_seqs": [r.rev_seq for r in basis],
            "rebuilt_view": rebuilt_view,
            "revoked_lease_ids": [l["lease_id"] for l in affected_leases],
            "revoked_leases": affected_leases,
        }
        self._log("rollback_committed", rev_seq=new_seq, **{
            k: v for k, v in result.items() if k != "revoked_leases"
        })
        return result

    # ---- forced-termination recovery ------------------------------------
    def recover(self) -> dict[str, Any]:
        """Recover using only fully committed SQLite records.

        Active leases that never reached a terminal state are marked
        ``unknown`` (E_RECOVER) and must be re-requested; the platform never
        fabricates an exit report for them.  Cursors and budget consumption are
        recomputed from applied records and corrected against stored rows.
        """
        discrepancies: list[dict[str, Any]] = []
        leases = self.store.list_leases()
        unknown_leases: list[dict[str, Any]] = []
        for lease in leases:
            if lease["status"] == "active":
                self.store.update_lease_status(
                    lease["lease_id"], "unknown", "E_RECOVER",
                )
                unknown_leases.append(lease)
                self._log(
                    "lease_unknown_after_crash", code="E_RECOVER",
                    node_id=lease["node_id"], rev_seq=lease["rev_seq"],
                    lease_id=lease["lease_id"],
                    request_id=lease["request_id"],
                )
                discrepancies.append({
                    "kind": "unknown_lease",
                    "lease_id": lease["lease_id"],
                    "node_id": lease["node_id"],
                    "rev_seq": lease["rev_seq"],
                })

        # Recompute authoritative cursors/views from applied lease records.
        applied = [l for l in leases if l["status"] == "applied"]
        expected_cursors = {node_id: 0 for node_id in self.node_specs}
        expected_views = {node_id: {} for node_id in self.node_specs}
        node_consumed = {node_id: 0 for node_id in self.node_specs}
        for lease in sorted(applied, key=lambda l: (l["node_id"], l["rev_seq"])):
            node_id = lease["node_id"]
            if node_id not in expected_cursors:
                continue
            revision = self.revisions.get(lease["rev_seq"])
            if revision is None:
                continue
            for key in sorted(revision.changes):
                expected_views[node_id][key] = revision.changes[key]
            expected_cursors[node_id] = lease["rev_seq"]
            node_consumed[node_id] += 1

        stored_cursors = self.store.list_cursors()
        for node_id in self.node_specs:
            stored = stored_cursors.get(node_id, {"applied_rev": 0, "view": {}})
            if stored.get("applied_rev") != expected_cursors[node_id] or \
                    stored.get("view") != expected_views[node_id]:
                discrepancies.append({
                    "kind": "cursor",
                    "node_id": node_id,
                    "stored_rev": stored.get("applied_rev"),
                    "expected_rev": expected_cursors[node_id],
                })
                self._log(
                    "cursor_corrected", code="E_RECOVER", node_id=node_id,
                    stored_rev=stored.get("applied_rev"),
                    expected_rev=expected_cursors[node_id],
                )
            # Correct the persisted row to the recomputed authoritative value.
            self.store.put_cursor(
                node_id, expected_cursors[node_id], expected_views[node_id],
            )
            self.cursors[node_id] = expected_cursors[node_id]
            self.views[node_id] = expected_views[node_id]

        # Recompute budgets: global = granted/applied leases; node = applied.
        granted = [l for l in leases if l["status"] in ("granted", "applied")]
        expected_global = len(granted)
        stored_budgets = {row["name"]: row for row in self.store.list_budgets()}
        global_row = stored_budgets.get("global")
        if global_row is None or global_row["consumed"] != expected_global:
            discrepancies.append({
                "kind": "global_budget",
                "stored": None if global_row is None else global_row["consumed"],
                "expected": expected_global,
            })
            self._log("budget_corrected", code="E_RECOVER",
                      stored=None if global_row is None else global_row["consumed"],
                      expected=expected_global)
        self.budgets.global_budget.consumed = expected_global
        self.store.put_budget(
            "global", "global",
            self.budgets.global_budget.limit, expected_global,
        )
        for node_id, budget in self.budgets.node_budgets.items():
            expected = node_consumed.get(node_id, 0)
            row = stored_budgets.get(budget.name)
            if row is None or row["consumed"] != expected:
                discrepancies.append({
                    "kind": "node_budget", "node_id": node_id,
                    "stored": None if row is None else row["consumed"],
                    "expected": expected,
                })
                self._log("budget_corrected", code="E_RECOVER", node_id=node_id,
                          stored=None if row is None else row["consumed"],
                          expected=expected)
            budget.consumed = expected
            self.store.put_budget(budget.name, "node", budget.limit, expected)

        result = {
            "status": "recovered",
            "unknown_lease_ids": [l["lease_id"] for l in unknown_leases],
            "discrepancies": discrepancies,
            "cursors": dict(self.cursors),
            "budgets": self.budgets.snapshots(),
            "exit_report_allowed": not unknown_leases,
        }
        self._log("recovery_completed", **{
            k: v for k, v in result.items() if k not in {"discrepancies"}
        }, discrepancy_count=len(discrepancies))
        return result

    async def rebuild_nodes_after_recover(self) -> list[dict[str, Any]]:
        """Push the recovered whole view to freshly started node processes."""
        acks = []
        for node_id, node in self.nodes.items():
            ack = await node.request({
                "type": "replace_view",
                "view": self.views.get(node_id, {}),
                "applied_rev": self.cursors.get(node_id, 0),
                "clear_lease": True,
            })
            if ack.get("type") != "replace_ack":
                raise RecoverError("node rebuild failed", node_id=node_id, got=ack)
            acks.append(ack)
        return acks

    def build_exit_report(self) -> dict[str, Any]:
        unknown = [l for l in self.store.list_leases() if l["status"] == "unknown"]
        if unknown:
            raise RecoverError(
                "cannot produce exit report while unknown leases exist; "
                "they must be re-requested",
                unknown_lease_ids=[l["lease_id"] for l in unknown],
            )
        return {
            "session_id": self.config.session_id,
            "revisions": self.revision_order,
            "cursors": dict(self.cursors),
            "budgets": self.budgets.snapshots(),
            "leases": self.store.list_leases(),
            "generated_ms": now_ms(),
        }

    # ---- snapshots & replay ---------------------------------------------
    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "revisions": [self.revisions[s].to_dict() for s in self.revision_order],
            "superseded": sorted(self.superseded),
            "leases": self.store.list_leases(),
            "cursors": {
                node_id: {"applied_rev": self.cursors.get(node_id, 0),
                          "view": self.views.get(node_id, {})}
                for node_id in sorted(self.node_specs)
            },
            "views": {node_id: self.views.get(node_id, {})
                      for node_id in sorted(self.node_specs)},
            "budgets": self.budgets.snapshots(),
        }

    def save_snapshot(self, name: str = "session") -> dict[str, Any]:
        state = self._snapshot_state()
        digest = content_hash(state)
        self.store.save_snapshot(name, SNAPSHOT_VERSION, digest, state, now_ms())
        self._log("snapshot_saved", snapshot=name, version=SNAPSHOT_VERSION,
                  content_hash=digest)
        return {"name": name, "version": SNAPSHOT_VERSION,
                "content_hash": digest, "state": state}

    def reopen_with_snapshot(self, name: str = "session",
                             force_version: int | None = None,
                             force_hash: str | None = None) -> dict[str, Any]:
        """Re-open a session and reconcile against its snapshot.

        A version/hash mismatch (E_VERSION) triggers a whole-state replacement
        of lease and cursor state, never an incremental merge.  The committed
        lease sequence is then replayed and compared field by field.
        """
        snapshot = self.store.load_snapshot(name)
        if snapshot is None:
            raise VersionConflictError("snapshot not found", snapshot=name)
        current = self._snapshot_state()
        compare_snapshot = dict(snapshot)
        if force_version is not None:
            compare_snapshot["version"] = force_version
        if force_hash is not None:
            compare_snapshot["content_hash"] = force_hash

        replaced = False
        mismatch: LeaseError | None = None
        try:
            assert_snapshot_compatible(compare_snapshot, current)
        except VersionConflictError as exc:
            mismatch = exc
            replaced = True
            self._replace_state_from_snapshot(snapshot["state"])
            self._log_error(exc, snapshot=name, action="whole_state_replace")

        replay_result = self.replay_snapshot(name)
        result = {
            "status": "reopened",
            "snapshot": name,
            "version_conflict": replaced,
            "conflict": None if mismatch is None else mismatch.to_dict(),
            "replay": replay_result,
        }
        self._log("snapshot_reopened", **result)
        return result

    def _replace_state_from_snapshot(self, state: dict[str, Any]) -> None:
        # Whole replacement: discard live lease/cursor state and adopt snapshot.
        self.superseded = set(state.get("superseded", []))
        self.cursors = {
            node_id: row["applied_rev"]
            for node_id, row in state.get("cursors", {}).items()
        }
        self.views = {node_id: dict(view)
                      for node_id, view in state.get("views", {}).items()}
        for node_id, row in state.get("cursors", {}).items():
            self.store.put_cursor(node_id, row["applied_rev"], row["view"])
        for lease in state.get("leases", []):
            self.store.put_lease(lease, lease["status"], lease.get("detail"))
        budgets = state.get("budgets", {})
        g = budgets.get("global", {})
        self.budgets.global_budget.limit = g.get("limit", self.budgets.global_budget.limit)
        self.budgets.global_budget.consumed = g.get("consumed", 0)
        self.store.put_budget("global", "global",
                              self.budgets.global_budget.limit,
                              self.budgets.global_budget.consumed)
        for node_id, row in budgets.get("nodes", {}).items():
            budget = self.budgets.node_budget(node_id)
            budget.limit = row.get("limit", budget.limit)
            budget.consumed = row.get("consumed", 0)
            self.store.put_budget(budget.name, "node", budget.limit, budget.consumed)

    def replay_snapshot(self, name: str = "session") -> dict[str, Any]:
        snapshot = self.store.load_snapshot(name)
        if snapshot is None:
            raise VersionConflictError("snapshot not found", snapshot=name)
        revisions = [Revision(r["rev_seq"], r["parent"], r["changes"],
                              r.get("kind", "normal"), r.get("rollback_target"))
                     for r in snapshot["state"]["revisions"]]
        return replay_lease_records(
            snapshot["state"]["leases"], revisions, snapshot["state"],
        )

    def query_logs(self, node_id: str | None = None, rev_seq: int | None = None,
                   code: str | None = None) -> list[dict[str, Any]]:
        return self.store.query_events(node_id=node_id, rev_seq=rev_seq, code=code)
