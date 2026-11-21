"""Gateway orchestration: routing, batching, hot swap and snapshot rebuild.

The gateway keeps one persistent connection per backend. Requests are routed
with a stable hash bucket and accumulated per (rule, destination) until the
declared batch size is reached. A hot swap observed while a batch is pending
splits the batch at the parameter-version boundary; the two halves are sent
separately and every split is recorded with its explicit reason, never mixed.
"""

import itertools
import threading
import uuid
from datetime import datetime, timezone

from . import determinism as det
from .backend import BackendProcess
from .hashing import route_decision, stable_bucket
from .snapshot import build_snapshot
from .spec import load_declaration
from .storage import LedgerStore


class Gateway:
    def __init__(self, declaration_path, db_path, python_executable=None):
        self.declaration = load_declaration(declaration_path)
        self.store = LedgerStore(db_path)
        self.backends = {}
        self._backend_seq = {}
        self._batch_counter = itertools.count(self._next_batch_index())
        self._buffers = {}
        self._buffer_lock = threading.RLock()
        self._rebuild_lock = threading.RLock()
        self.rebuild_gate = None
        self.rebuild_events = []
        for decl in self.declaration["backends"]:
            controller = BackendProcess(decl, python_executable)
            self.backends[decl["backend_id"]] = controller
            self._backend_seq[decl["backend_id"]] = 0

    def _next_batch_index(self):
        row = self.store.conn.execute(
            "SELECT COALESCE(MAX(rowid), 0) + 1 AS n FROM batches"
        ).fetchone()
        return row["n"]

    def start(self):
        for controller in self.backends.values():
            controller.start()
            self._record_param_version(controller, 0)

    def _record_param_version(self, controller, seq):
        decl = controller.decl
        seed = det.next_hotswap_seed(decl["seed"], seq) if seq else decl["seed"]
        self.store.register_param_version({
            "backend_id": controller.backend_id,
            "backend_seq": seq,
            "model_version": decl["model_version"],
            "quant_version": decl["quant_version"],
            "seed": seed,
            "model_dim": decl["model_dim"],
            "hidden_dim": decl["hidden_dim"],
            "param_digest": controller.version_digests[seq],
        })

    def hotswap_backend(self, backend_id):
        """Replace parameters on the same connection and bump the seq chain."""
        controller = self.backends[backend_id]
        new_seq = controller.active_seq + 1
        digest = controller.hotswap(new_seq)
        self._record_param_version(controller, new_seq)
        self._backend_seq[backend_id] = new_seq
        return {"backend_id": backend_id, "new_seq": new_seq,
                "param_digest": digest}

    def hotswap_gray_backends(self):
        results = []
        for rule in self.declaration["rules"]:
            gray = rule["gray_backend"]
            if not any(r["backend_id"] == gray for r in results):
                results.append(self.hotswap_backend(gray))
        return results

    def drop_param_version(self, backend_id, seq):
        controller = self.backends[backend_id]
        controller.drop_version(seq)

    def admit_request(self, request):
        rule = self._rule_for(request)
        decision = route_decision(rule["rule_id"], request["client_key"],
                                  rule["gray_weight_bp"])
        backend_id = (rule["stable_backend"] if decision == "stable"
                      else rule["gray_backend"])
        controller = self.backends[backend_id]
        backend = controller.decl
        vector = det.build_input_vector(
            rule["rule_id"], request["sample_id"], backend["model_dim"]
        )
        entry = {
            "rule_id": rule["rule_id"],
            "sample_id": request["sample_id"],
            "client_key": request["client_key"],
            "destination": decision,
            "bucket": stable_bucket(rule["rule_id"], request["client_key"]),
            "weight_bp": rule["gray_weight_bp"],
            "backend_id": backend_id,
            "backend_seq": controller.active_seq,
            "param_digest": controller.version_digests[controller.active_seq],
            "vector": vector,
            "input_digest": det.input_digest(
                rule["rule_id"], request["sample_id"],
                backend["model_dim"], vector),
        }
        key = (rule["rule_id"], decision)
        with self._buffer_lock:
            self._buffers.setdefault(key, []).append(entry)
            if len(self._buffers[key]) >= rule["batch_size"]:
                pending = self._buffers.pop(key)
                self._flush_batch(rule, decision, pending)
        return entry

    def flush_all(self):
        flushed = []
        with self._buffer_lock:
            keys = list(self._buffers)
            for key in keys:
                rule_id, decision = key
                pending = self._buffers.pop(key)
                rule = next(r for r in self.declaration["rules"]
                            if r["rule_id"] == rule_id)
                flushed.append(self._flush_batch(rule, decision, pending))
        return [f for f in flushed if f]

    def _rule_for(self, request):
        for rule in self.declaration["rules"]:
            if rule["rule_id"] == request["rule_id"]:
                return rule
        raise KeyError(f"unknown rule {request['rule_id']}")

    def _flush_batch(self, rule, decision, pending):
        if not pending:
            return None
        batch_id = f"batch-{next(self._batch_counter):05d}"
        backend_id = (rule["stable_backend"] if decision == "stable"
                      else rule["gray_backend"])
        self.store.mark_batch_intent(batch_id, rule["rule_id"], decision)

        groups = self._split_by_version(pending)
        hits, split_reasons = [], {}
        for sub_batch, (seq, rows, reason) in enumerate(groups):
            split_reasons[sub_batch] = reason
            controller = self.backends[backend_id]
            import numpy as np

            inputs = np.vstack([row["vector"] for row in rows])
            reply = controller.infer_batch(
                rule["rule_id"], seq, [r["sample_id"] for r in rows], inputs
            )
            for row in rows:
                hits.append({
                    "sub_batch": sub_batch,
                    "rule_id": row["rule_id"],
                    "sample_id": row["sample_id"],
                    "client_key": row["client_key"],
                    "destination": row["destination"],
                    "bucket": row["bucket"],
                    "weight_bp": row["weight_bp"],
                    "backend_id": backend_id,
                    "backend_seq": seq,
                    "param_digest": reply["param_digest"],
                    "output_digest": reply["digest_by_sample"][row["sample_id"]],
                    "input_digest": row["input_digest"],
                    "elapsed_ms": reply["elapsed_ms"],
                })
        self.store.commit_batch(batch_id, rule["rule_id"], decision,
                                hits, split_reasons)
        return {"batch_id": batch_id, "rule_id": rule["rule_id"],
                "destination": decision, "sub_batches": len(groups),
                "hits": len(hits)}

    @staticmethod
    def _split_by_version(pending):
        """Split explicitly at backend_seq boundaries (version-chain order)."""
        groups = []
        current_seq = pending[0]["backend_seq"]
        current_rows = []
        for row in pending:
            if row["backend_seq"] != current_seq:
                reason = (f"param version boundary: seq {current_seq} -> "
                          f"{row['backend_seq']} mid-batch; split to avoid "
                          "mixing parameter digest versions")
                groups.append((current_seq, current_rows, reason))
                current_seq = row["backend_seq"]
                current_rows = []
            current_rows.append(row)
        reason = ("single parameter version" if not groups else
                  f"tail after version boundary, seq {current_seq}")
        groups.append((current_seq, current_rows, reason))
        return groups

    def rebuild_snapshot(self, wait=True):
        """Rebuild on a worker thread; new requests keep being admitted."""
        event = {"rebuild_id": f"rebuild-{uuid.uuid4().hex[:12]}",
                 "started_upper_bound": self.store.max_ledger_seq(),
                 "snapshot": None, "error": None, "thread": None}

        def run():
            try:
                with self._rebuild_lock:
                    if self.rebuild_gate is not None:
                        self.rebuild_gate.wait(timeout=10)
                    upper = event["started_upper_bound"]
                    self.store.start_rebuild(event["rebuild_id"], upper)
                    snapshot = build_snapshot(
                        self.store, event["rebuild_id"], upper
                    )
                    current_max = self.store.max_ledger_seq()
                    incoming = max(0, current_max - upper)
                    snapshot["incoming_during_rebuild"] = incoming
                    self.store.finish_rebuild(event["rebuild_id"], incoming)
                    self.store.save_snapshot(snapshot)
                    event["snapshot"] = snapshot
            except Exception as exc:  # surfaced via event and wait()
                event["error"] = exc

        thread = threading.Thread(target=run, name="snapshot-rebuild",
                                  daemon=True)
        event["thread"] = thread
        self.rebuild_events.append(event)
        thread.start()
        if wait:
            thread.join()
            if event["error"]:
                raise event["error"]
        return event

    def recover(self):
        """Recover committed state and mark interrupted batches as unknown."""
        unknown = []
        for intent in self.store.pending_intents():
            self.store.mark_intent_unknown(intent)
            unknown.append(intent["batch_id"])
        snapshot = self.store.load_latest_snapshot()
        return {
            "format_version": 1,
            "progress_cursor": self.store.max_ledger_seq(),
            "committed_hits": self.store.count_hits(),
            "unknown_batches": unknown + [
                b["batch_id"] for b in self.store.unknown_batches()
                if b["batch_id"] not in unknown
            ],
            "snapshot_id": snapshot["snapshot_id"] if snapshot else None,
        }

    def verify_replay(self):
        from .snapshot import verify_snapshot

        return verify_snapshot(self.declaration, self.store)

    def stop(self):
        for controller in self.backends.values():
            controller.stop()
        self.store.close()
