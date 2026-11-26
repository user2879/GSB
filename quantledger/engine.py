"""Parent-side orchestrator for workers, command replay, cache and receipts."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import (
    BatchSpec,
    LoadedConfig,
    RuntimeState,
    apply_command,
    load_base_config,
    validate_commands,
)
from .errors import (
    E_COMMAND,
    E_LEDGER_CORRUPT,
    E_LEDGER_VERSION,
    E_PORT_IN_USE,
    E_PROTOCOL,
    E_RECONCILE_PENDING,
    E_WORKER_START,
    BusinessIssue,
    QLabError,
)
from .ledger import (
    LedgerStore,
    empty_ledger,
    recovery_report,
    semantic_snapshot,
    snapshot_digest,
    state_to_dict,
)
from .protocol import JsonLineClient
from .quant import layer_cache_keys, quant_summary
from .receipts import (
    Receipt,
    arbitrate_receipts,
    downgrade_q3_to_q1,
    load_receipts,
    materialize_receipt_versions,
    parse_receipt,
    receipt_roundtrip,
    upgrade_to_q3,
)
from .util import canonical_json, digest_array, digest_json, read_json


@dataclass
class WorkerProcessHandle:
    spec: Any
    process: subprocess.Popen[bytes]
    client: JsonLineClient
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any


class LedgerEngine:
    def __init__(
        self,
        config_path: str | Path,
        *,
        commands_path: str | Path | None = None,
        runtime_dir: str | Path = "runtime",
        ledger_path: str | Path | None = None,
        lease_ttl_ms: int = 5000,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.loaded: LoadedConfig = load_base_config(self.config_path)
        self.base_state = self.loaded.state.clone()
        self.lease_ttl_ms = int(lease_ttl_ms)
        self.request_timeout_s = float(request_timeout_s)
        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir = self.runtime_dir / "receipts"
        self.logs_dir = self.runtime_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.store = LedgerStore(ledger_path or (self.runtime_dir / "ledger.json"))
        self._handles: dict[str, WorkerProcessHandle] = {}
        self._lease_events: list[dict[str, Any]] = []

        self.issues: list[BusinessIssue] = list(self.loaded.issues)
        if commands_path is not None:
            raw = read_json(commands_path)
            commands, issues = validate_commands(raw, self.base_state)
            self.commands = commands
            self.issues.extend(issues)
        else:
            self.commands = list(self.loaded.commands)

        self.receipt_paths = self._resolve_receipt_paths()
        self.issue_dicts = [item.as_dict() for item in self.issues]

    def _resolve_receipt_paths(self) -> list[Path]:
        paths: list[Path] = []
        for raw in self.loaded.receipt_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = self.config_path.parent / path
            paths.append(path.resolve())
        return sorted(paths)

    def _replay(self, commands: list[dict[str, Any]]) -> RuntimeState:
        state = self.base_state.clone()
        for command in commands:
            try:
                apply_command(state, command)
            except (ValueError, KeyError) as exc:
                raise QLabError(E_LEDGER_CORRUPT, f"cannot replay accepted command {command.get('seq')}: {exc}") from exc
        return state

    def _check_ports(self) -> None:
        for worker in self.base_state.workers:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((worker.host, worker.port))
            except OSError as exc:
                raise QLabError(
                    E_PORT_IN_USE,
                    f"worker {worker.worker_id} loopback port {worker.port} is unavailable: {exc}",
                    worker_id=worker.worker_id,
                    port=worker.port,
                ) from exc
            finally:
                sock.close()

    @contextmanager
    def workers(self) -> Iterable[dict[str, WorkerProcessHandle]]:
        self._check_ports()
        project_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            for worker in sorted(self.base_state.workers, key=lambda item: item.worker_id):
                stdout_path = self.logs_dir / f"{worker.worker_id}.stdout.log"
                stderr_path = self.logs_dir / f"{worker.worker_id}.stderr.log"
                stdout_handle = stdout_path.open("ab", buffering=0)
                stderr_handle = stderr_path.open("ab", buffering=0)
                command = [
                    sys.executable,
                    "-m",
                    "quantledger.worker_main",
                    "--config",
                    str(self.config_path),
                    "--worker-id",
                    worker.worker_id,
                ]
                process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                client = JsonLineClient(worker.host, worker.port, timeout=self.request_timeout_s)
                handle = WorkerProcessHandle(
                    spec=worker,
                    process=process,
                    client=client,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                )
                self._handles[worker.worker_id] = handle
                try:
                    client.wait_ready(timeout=10.0)
                except Exception as exc:
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                    raise QLabError(
                        E_WORKER_START,
                        f"worker {worker.worker_id} failed to start: {exc}; stderr={stderr[-1000:]}",
                        worker_id=worker.worker_id,
                    ) from exc
            yield self._handles
        finally:
            for handle in self._handles.values():
                if handle.process.poll() is None:
                    handle.process.terminate()
            for handle in self._handles.values():
                try:
                    handle.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=2.0)
                handle.stdout_handle.close()
                handle.stderr_handle.close()
            self._handles.clear()

    def _set_worker_limits(self, state: RuntimeState, handles: dict[str, WorkerProcessHandle]) -> None:
        for worker in sorted(state.workers, key=lambda item: item.worker_id):
            handle = handles.get(worker.worker_id)
            if handle is None:
                continue
            response = handle.client.request(
                {"op": "set_lease_limit", "value": worker.lease_limit, "request_id": f"limit-{worker.worker_id}"}
            )
            if not response.get("ok"):
                raise QLabError(
                    E_WORKER_START,
                    f"cannot set lease limit for {worker.worker_id}: {response.get('message')}",
                )

    def _event(self, ledger: dict[str, Any], **payload: Any) -> None:
        event = {"event_seq": int(ledger["next_event_seq"]), **payload}
        ledger["next_event_seq"] = int(ledger["next_event_seq"]) + 1
        ledger["events"].append(event)

    def _ensure_historical_deliveries(self, ledger: dict[str, Any]) -> None:
        if not self.receipt_paths:
            return
        sidecar_candidates = [path.parent / "q3_to_q1_sidecar.json" for path in self.receipt_paths]
        sidecar = next((path for path in sidecar_candidates if path.exists()), None)
        receipts, issues = load_receipts(self.receipt_paths, sidecar_path=sidecar)
        for item in issues:
            ledger["command_issues"].append(item.as_dict())
        for receipt in receipts:
            self._append_delivery(
                ledger,
                upgrade_to_q3(receipt),
                origin="configured_history",
                command_seq=None,
            )

    def _delivery_key(self, receipt: Receipt | dict[str, Any]) -> str:
        if isinstance(receipt, Receipt):
            payload = receipt.to_q3()
        else:
            payload = receipt
        return digest_json({"q3_delivery": payload})

    def _append_delivery(
        self,
        ledger: dict[str, Any],
        receipt: Receipt,
        *,
        origin: str,
        command_seq: int | None,
        worker_pid: int | None = None,
    ) -> bool:
        normalized = upgrade_to_q3(receipt)
        key = self._delivery_key(normalized)
        if any(item.get("delivery_key") == key for item in ledger["deliveries"]):
            return False
        ledger["deliveries"].append(
            {
                "delivery_key": key,
                "origin": origin,
                "command_seq": command_seq,
                "worker_pid": worker_pid,
                "receipt": normalized.to_q3(),
            }
        )
        ledger["deliveries"].sort(
            key=lambda item: (
                item["receipt"].get("batch_id", ""),
                item["receipt"].get("layer_name", ""),
                item["receipt"].get("output_summary", ""),
                int(item["receipt"].get("revision", -1)) if isinstance(item["receipt"].get("revision"), int) else -1,
                str(item["receipt"].get("worker_id", "")),
                item["delivery_key"],
            )
        )
        return True

    def _layer_payload(self, layers: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": layer.name,
                "rows": layer.rows,
                "cols": layer.cols,
                "group_size": layer.group_size,
                "weight": layer.weight.astype(float).tolist(),
                "weight_seed": layer.weight_seed,
                **({"bias": layer.bias.astype(float).tolist()} if layer.bias is not None else {}),
            }
            for layer in layers
        ]

    def _worker_for_batch(self, batch_id: str, handles: dict[str, WorkerProcessHandle]) -> WorkerProcessHandle:
        worker_ids = sorted(handles)
        stable_bucket = int(digest_json({"batch_id": batch_id})[:16], 16)
        return handles[worker_ids[stable_bucket % len(worker_ids)]]

    def _submit_and_collect(
        self,
        handle: WorkerProcessHandle,
        state: RuntimeState,
        batch: BatchSpec,
        start_layer: int,
        activation: np.ndarray,
        command_seq: int,
        ttl_ms: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "op": "submit",
            "request_id": f"submit-{batch.batch_id}-{command_seq}",
            "batch_id": batch.batch_id,
            "samples": activation.astype(float).tolist(),
            "layers": self._layer_payload(state.layers[start_layer:]),
            "revision": state.revision,
            "frac_bits": state.frac_bits,
            "quant_level": state.quant_level,
            "ttl_ms": int(ttl_ms or self.lease_ttl_ms),
        }
        deadline = time.monotonic() + self.request_timeout_s
        while True:
            response = handle.client.request(payload)
            if response.get("ok"):
                break
            if response.get("code") != "E_LEASE_CAPACITY":
                raise QLabError(
                    str(response.get("code", E_PROTOCOL)),
                    str(response.get("message", "dispatch rejected")),
                    batch_id=batch.batch_id,
                )
            if time.monotonic() >= deadline:
                raise QLabError("E_LEASE_CAPACITY", "worker lease capacity was never available", batch_id=batch.batch_id)
            time.sleep(0.005)
        lease = response["lease"]
        lease_id = str(lease["lease_id"])
        generation = int(lease["generation"])
        result: dict[str, Any] | None = None
        try:
            while time.monotonic() < deadline:
                status = handle.client.request({"op": "fetch", "lease_id": lease_id})
                if not status.get("ok"):
                    code = str(status.get("code", E_PROTOCOL))
                    if code == E_RECONCILE_PENDING:
                        self._lease_events.append(
                            {
                                "code": E_RECONCILE_PENDING,
                                "worker_id": handle.spec.worker_id,
                                "batch_id": batch.batch_id,
                                "lease_id": lease_id,
                                "generation": generation,
                            }
                        )
                    raise QLabError(code, str(status.get("message", "fetch failed")), batch_id=batch.batch_id)
                if status.get("status") == "completed":
                    result = status.get("result")
                    break
                time.sleep(0.003)
        finally:
            try:
                release = handle.client.request(
                    {"op": "release", "lease_id": lease_id, "generation": generation, "revision": state.revision}
                )
                if not release.get("ok"):
                    self._lease_events.append(
                        {
                            "code": str(release.get("code", E_RECONCILE_PENDING)),
                            "worker_id": handle.spec.worker_id,
                            "batch_id": batch.batch_id,
                            "lease_id": lease_id,
                            "generation": generation,
                        }
                    )
            except Exception as exc:
                self._lease_events.append(
                    {
                        "code": E_RECONCILE_PENDING,
                        "worker_id": handle.spec.worker_id,
                        "batch_id": batch.batch_id,
                        "lease_id": lease_id,
                        "generation": generation,
                        "message": str(exc),
                    }
                )
        if result is None:
            raise QLabError("E_TIMEOUT", "timed out waiting for worker result", batch_id=batch.batch_id)
        if not result.get("ok"):
            raise QLabError(
                str(result.get("code", "E_WORKER_INTERNAL")),
                str(result.get("message", "worker job failed")),
                batch_id=batch.batch_id,
            )
        return result

    def _submit_with_retry(
        self,
        handle: WorkerProcessHandle,
        state: RuntimeState,
        batch: BatchSpec,
        start_layer: int,
        activation: np.ndarray,
        command_seq: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.request_timeout_s
        last_error: QLabError | None = None
        for attempt in range(3):
            try:
                return self._submit_and_collect(
                    handle,
                    state,
                    batch,
                    start_layer,
                    activation,
                    command_seq,
                    ttl_ms=self.lease_ttl_ms if attempt == 0 else max(1000, self.lease_ttl_ms * 4),
                )
            except QLabError as exc:
                last_error = exc
                if exc.code not in {E_RECONCILE_PENDING, "E_LEASE_EXPIRED", "E_TIMEOUT"}:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        if last_error is not None:
            raise last_error
        raise QLabError(E_RECONCILE_PENDING, "dispatch retry exhausted", batch_id=batch.batch_id)

    def _merge_result(
        self,
        state: RuntimeState,
        batch: BatchSpec,
        start_layer: int,
        prefix_records: list[dict[str, Any]],
        worker_result: dict[str, Any],
        command_seq: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        records = copy.deepcopy(prefix_records)
        for record in worker_result["layers"]:
            record = copy.deepcopy(record)
            record["layer_index"] = int(record["layer_index"]) + start_layer
            records.append(record)
        output = np.asarray(worker_result["output"], dtype=np.float64)
        ledger_key = digest_json(
            {
                "batch_id": batch.batch_id,
                "revision": state.revision,
                "frac_bits": state.frac_bits,
                "quant_level": state.quant_level,
                "layer_cache_keys": [record["cache_keys"]["dequant"] for record in records],
                "output": digest_array(output),
            }
        )
        result_meta = {
            "batch_id": batch.batch_id,
            "ledger_cache_key": ledger_key,
            "output_digest": digest_array(output),
            "output": output.astype(float).tolist(),
            "layer_cache_keys": [record["cache_keys"]["dequant"] for record in records],
        }
        run_batch = {
            "batch_id": batch.batch_id,
            "sample_digest": batch.digest(),
            "revision": state.revision,
            "frac_bits": state.frac_bits,
            "quant_level": state.quant_level,
            "worker_id": worker_result["worker_id"],
            "duration_ms": int(worker_result.get("duration_ms", 0)),
            "command_seq": command_seq,
            "status": "complete",
            "output": result_meta["output"],
            "output_digest": result_meta["output_digest"],
            "ledger_cache_key": ledger_key,
            "layers": records,
        }
        receipt = copy.deepcopy(worker_result["receipt"])
        receipt["quant_params_summary"] = quant_summary(records)
        prior_refs = sorted(
            self._semantic_delivery_ref(item["receipt"])
            for item in self.ledger["deliveries"]
            if item["receipt"].get("batch_id") == batch.batch_id
        )
        receipt["superseded_refs"] = prior_refs
        receipt["stable_rank"] = 1
        return run_batch, receipt

    def _semantic_delivery_ref(self, receipt: dict[str, Any]) -> str:
        return digest_json(
            {
                "batch_id": receipt.get("batch_id"),
                "layer_name": receipt.get("layer_name"),
                "output_summary": receipt.get("output_summary"),
                "revision": receipt.get("revision"),
                "worker_id": receipt.get("worker_id"),
                "stable_rank": receipt.get("stable_rank"),
                "quant_params_summary": receipt.get("quant_params_summary"),
            }
        )

    def _execute_prepared(self, prepared: dict[str, Any], command_seq: int) -> None:
        if not prepared:
            return
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        batch_worker: dict[str, str] = {}
        for batch_id in sorted(prepared):
            spec = prepared[batch_id]
            handle = self._worker_for_batch(batch_id, self._handles)
            grouped.setdefault(handle.spec.worker_id, []).append((batch_id, spec))
            batch_worker[batch_id] = handle.spec.worker_id
        future_map: dict[str, concurrent.futures.Future[Any]] = {}
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(prepared)))
        try:
            for worker_id in sorted(grouped):
                handle = self._handles[worker_id]
                for batch_id, spec in grouped[worker_id]:
                    future = executor.submit(
                        self._submit_with_retry,
                        handle,
                        spec["state"],
                        spec["batch"],
                        spec["start_layer"],
                        spec["activation"],
                        command_seq,
                    )
                    future_map[batch_id] = future
            for batch_id in sorted(future_map):
                spec = prepared[batch_id]
                try:
                    worker_result = future_map[batch_id].result()
                    run_batch, receipt = self._merge_result(
                        spec["state"],
                        spec["batch"],
                        spec["start_layer"],
                        spec["prefix_records"],
                        worker_result,
                        command_seq,
                    )
                    self.ledger["run"]["batches"][batch_id] = run_batch
                    self._append_delivery(
                        self.ledger,
                        parse_receipt(receipt),
                        origin="worker",
                        command_seq=command_seq,
                        worker_pid=self._handles[batch_worker[batch_id]].process.pid,
                    )
                    self.ledger["pending_batches"] = [
                        item for item in self.ledger["pending_batches"] if item != batch_id
                    ]
                except QLabError as exc:
                    self.ledger["run"]["batches"][batch_id] = {
                        "batch_id": batch_id,
                        "sample_digest": spec["batch"].digest(),
                        "revision": spec["state"].revision,
                        "frac_bits": spec["state"].frac_bits,
                        "quant_level": spec["state"].quant_level,
                        "worker_id": None,
                        "command_seq": command_seq,
                        "status": "failed",
                        "error": {"code": exc.code, "message": exc.message, **exc.details},
                        "layers": spec["prefix_records"],
                    }
                    self.ledger["pending_batches"] = [
                        item for item in self.ledger["pending_batches"] if item != batch_id
                    ]
            self.ledger["pending_batches"] = sorted(set(self.ledger["pending_batches"]))
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    def _prefix_for_batch(
        self,
        batch: BatchSpec,
        state: RuntimeState,
        old_batch: dict[str, Any] | None,
        start_hint: int,
        command_seq: int,
        reason: str,
    ) -> tuple[int, list[dict[str, Any]], np.ndarray]:
        if old_batch is None or old_batch.get("status") != "complete":
            return 0, [], np.asarray(batch.samples, dtype=np.float64)
        old_layers = list(old_batch.get("layers", []))
        activation = np.asarray(batch.samples, dtype=np.float64)
        valid = 0
        limit = min(int(start_hint), len(old_layers))
        for index in range(limit):
            layer = state.layers[index]
            expected = layer_cache_keys(
                layer,
                activation=activation,
                revision=state.revision,
                frac_bits=state.frac_bits,
                quant_level=state.quant_level,
                batch_ids=[batch.batch_id],
            )
            old = old_layers[index]
            if old.get("cache_keys", {}).get("dequant") != expected["dequant"]:
                break
            valid += 1
            activation = np.asarray(old["dequantized_output"], dtype=np.float64)
            self._event(
                self.ledger,
                command_seq=command_seq,
                kind="reuse",
                entity="layer",
                batch_id=batch.batch_id,
                layer=layer.name,
                layer_index=index,
                cache_key=expected["dequant"],
                reason="prefix_cache_key_match",
            )
        for index in range(valid, len(old_layers)):
            layer = state.layers[index]
            old = old_layers[index]
            self._event(
                self.ledger,
                command_seq=command_seq,
                kind="invalidate",
                entity="layer",
                batch_id=batch.batch_id,
                layer=old.get("layer", layer.name),
                layer_index=index,
                cache_key=old.get("cache_keys", {}).get("dequant"),
                reason=reason,
            )
        if old_batch.get("ledger_cache_key"):
            self._event(
                self.ledger,
                command_seq=command_seq,
                kind="invalidate",
                entity="ledger_entry",
                batch_id=batch.batch_id,
                layer=None,
                layer_index=None,
                cache_key=old_batch.get("ledger_cache_key"),
                reason=reason,
            )
        return valid, copy.deepcopy(old_layers[:valid]), activation

    def _prepare_state_and_command(self, command: dict[str, Any], state: RuntimeState) -> dict[str, Any]:
        command_type = command["type"]
        command_seq = int(command["seq"])
        prepared: dict[str, Any] = {}
        if command_type == "append_batch":
            batch_id = command["batch_id"]
            batch = next(item for item in state.batches if item.batch_id == batch_id)
            prepared[batch_id] = {
                "batch": batch,
                "state": state,
                "start_layer": 0,
                "prefix_records": [],
                "activation": np.asarray(batch.samples, dtype=np.float64),
                "reason": "append_batch",
            }
        elif command_type in {"set_quant_level", "set_frac_bits", "set_revision", "rollback_revision"}:
            reason = command_type
            for batch in state.batches:
                old = self.ledger["run"]["batches"].get(batch.batch_id)
                valid, prefix, activation = self._prefix_for_batch(
                    batch, state, old, 0, command_seq, reason
                )
                prepared[batch.batch_id] = {
                    "batch": batch,
                    "state": state,
                    "start_layer": valid,
                    "prefix_records": prefix,
                    "activation": activation,
                    "reason": reason,
                }
        elif command_type == "replace_weight":
            layer_name = command["layer"]
            found = state.find_layer(layer_name)
            if found is None:
                raise QLabError(E_COMMAND, f"accepted command references unknown layer {layer_name}")
            layer_index, _ = found
            for batch in state.batches:
                old = self.ledger["run"]["batches"].get(batch.batch_id)
                valid, prefix, activation = self._prefix_for_batch(
                    batch, state, old, layer_index, command_seq, f"replace_weight:{layer_name}"
                )
                prepared[batch.batch_id] = {
                    "batch": batch,
                    "state": state,
                    "start_layer": valid,
                    "prefix_records": prefix,
                    "activation": activation,
                    "reason": f"replace_weight:{layer_name}",
                }
        elif command_type == "remove_batch":
            batch_id = command["batch_id"]
            old = self.ledger["run"]["batches"].get(batch_id)
            if old:
                for index, layer_record in enumerate(old.get("layers", [])):
                    self._event(
                        self.ledger,
                        command_seq=command_seq,
                        kind="invalidate",
                        entity="layer",
                        batch_id=batch_id,
                        layer=layer_record.get("layer"),
                        layer_index=index,
                        cache_key=layer_record.get("cache_keys", {}).get("dequant"),
                        reason="remove_batch",
                    )
                if old.get("ledger_cache_key"):
                    self._event(
                        self.ledger,
                        command_seq=command_seq,
                        kind="invalidate",
                        entity="ledger_entry",
                        batch_id=batch_id,
                        layer=None,
                        layer_index=None,
                        cache_key=old.get("ledger_cache_key"),
                        reason="remove_batch",
                    )
                self.ledger["run"]["batches"].pop(batch_id, None)
            prepared = {}
        elif command_type == "set_lease_limit":
            handle = self._handles.get(command["worker_id"])
            if handle is not None:
                response = handle.client.request(
                    {
                        "op": "set_lease_limit",
                        "value": command["lease_limit"],
                        "request_id": f"command-{command_seq}",
                    }
                )
                if not response.get("ok"):
                    raise QLabError(E_WORKER_START, f"lease-limit command failed: {response.get('message')}")
        elif command_type != "noop":
            raise QLabError(E_COMMAND, f"unsupported accepted command {command_type}")
        return prepared

    def _mark_pending(self, prepared: dict[str, Any], state: RuntimeState, command_seq: int) -> None:
        ledger_state = state_to_dict(state)
        self.ledger["state"] = ledger_state
        for batch_id, spec in prepared.items():
            old = self.ledger["run"]["batches"].get(batch_id, {})
            self.ledger["run"]["batches"][batch_id] = {
                "batch_id": batch_id,
                "sample_digest": spec["batch"].digest(),
                "revision": state.revision,
                "frac_bits": state.frac_bits,
                "quant_level": state.quant_level,
                "worker_id": None,
                "command_seq": command_seq,
                "status": "pending",
                "output": old.get("output") if spec["start_layer"] else None,
                "output_digest": old.get("output_digest") if spec["start_layer"] else None,
                "ledger_cache_key": old.get("ledger_cache_key") if spec["start_layer"] else None,
                "layers": copy.deepcopy(spec["prefix_records"]),
            }
        self.ledger["pending_batches"] = sorted(set(self.ledger.get("pending_batches", [])) | set(prepared))
        self.store.save(self.ledger)

    def _resume_pending(self) -> None:
        pending = sorted(self.ledger.get("pending_batches", []))
        if not pending:
            return
        state = self._replay(self.ledger["applied_commands"])
        prepared: dict[str, Any] = {}
        batches = {batch.batch_id: batch for batch in state.batches}
        for batch_id in pending:
            batch = batches.get(batch_id)
            if batch is None:
                self.ledger["pending_batches"] = [
                    item for item in self.ledger["pending_batches"] if item != batch_id
                ]
                continue
            old = self.ledger["run"]["batches"].get(batch_id, {})
            prefix = copy.deepcopy(old.get("layers", []))
            if prefix:
                activation = np.asarray(prefix[-1]["dequantized_output"], dtype=np.float64)
            else:
                activation = np.asarray(batch.samples, dtype=np.float64)
            prepared[batch_id] = {
                "batch": batch,
                "state": state,
                "start_layer": len(prefix),
                "prefix_records": prefix,
                "activation": activation,
                "reason": "resume_pending",
            }
        if prepared:
            self._execute_prepared(prepared, int(self.ledger["applied_commands"][-1]["seq"]) if self.ledger["applied_commands"] else 0)
            self.store.save(self.ledger)

    def _refresh_cache(self) -> None:
        layer_cache: dict[str, Any] = {}
        ledger_entries: dict[str, Any] = {}
        for batch_id in sorted(self.ledger["run"]["batches"]):
            batch = self.ledger["run"]["batches"][batch_id]
            if batch.get("status") != "complete":
                continue
            for record in batch.get("layers", []):
                key = record["cache_keys"]["dequant"]
                entry = layer_cache.setdefault(
                    key,
                    {
                        "cache_key": key,
                        "layer": record["layer"],
                        "layer_content_fingerprint": record["layer_content_fingerprint"],
                        "layer_config_fingerprint": record["layer_config_fingerprint"],
                        "group_fingerprints": record["group_fingerprints"],
                        "revision": record["revision"],
                        "frac_bits": record["frac_bits"],
                        "quant_level": record["quant_level"],
                        "dequantized_digest": record["dequantized_digest"],
                        "referenced_batch_ids": [],
                    },
                )
                entry["referenced_batch_ids"] = sorted(
                    set(entry["referenced_batch_ids"]) | set(record.get("referenced_batch_ids", [batch_id]))
                )
            ledger_entries[batch["ledger_cache_key"]] = {
                "cache_key": batch["ledger_cache_key"],
                "batch_id": batch_id,
                "revision": batch["revision"],
                "frac_bits": batch["frac_bits"],
                "quant_level": batch["quant_level"],
                "worker_id": batch["worker_id"],
                "output_digest": batch["output_digest"],
                "layer_cache_keys": [record["cache_keys"]["dequant"] for record in batch.get("layers", [])],
                "referenced_batch_ids": [batch_id],
            }
        self.ledger["cache"] = {
            "layers": {key: layer_cache[key] for key in sorted(layer_cache)},
            "ledger_entries": {key: ledger_entries[key] for key in sorted(ledger_entries)},
        }

    def _refresh_receipts(self) -> None:
        receipts_to_write: list[Receipt] = []
        for delivery in self.ledger["deliveries"]:
            try:
                receipts_to_write.append(parse_receipt(delivery["receipt"]))
            except QLabError:
                continue
        materialize_receipt_versions(receipts_to_write, self.receipts_dir)
        paths = [self.receipts_dir / "q1.json", self.receipts_dir / "q2.json", self.receipts_dir / "q3.json"]
        receipts, issues = load_receipts(paths, sidecar_path=self.receipts_dir / "q3_to_q1_sidecar.json")
        for item in issues:
            self.ledger["command_issues"].append(item.as_dict())
        arbitration = arbitrate_receipts(receipts)
        self.ledger["receipts"] = [item.to_internal_dict() for item in receipts]
        self.ledger["arbitration_decisions"] = arbitration["decisions"]
        self.ledger["authority"] = arbitration["authority"]
        downgrades: list[dict[str, Any]] = []
        roundtrips: list[dict[str, Any]] = []
        for delivery in self.ledger["deliveries"]:
            receipt = parse_receipt(delivery["receipt"])
            q1, sidecar = downgrade_q3_to_q1(receipt)
            check = receipt_roundtrip(receipt)
            downgrades.append(
                {
                    "delivery_key": delivery["delivery_key"],
                    "command_seq": delivery.get("command_seq"),
                    "batch_id": receipt.batch_id,
                    "layer_name": receipt.layer_name,
                    "from_format": "q3",
                    "to_format": "q1",
                    "sidecar_key": sidecar["sidecar_key"],
                    "restored_fields": [
                        "revision",
                        "duration_ms",
                        "worker_id",
                        "stable_rank",
                        "superseded_refs",
                        "quant_params_summary",
                    ],
                    "q1": q1,
                }
            )
            roundtrips.append(
                {
                    "delivery_key": delivery["delivery_key"],
                    "batch_id": receipt.batch_id,
                    "layer_name": receipt.layer_name,
                    "conclusion_unchanged": check["conclusion_unchanged"],
                    "completeness_before": check["completeness_before"],
                    "completeness_after": check["completeness_after"],
                }
            )
        self.ledger["downgrades"] = downgrades
        self.ledger["roundtrip_checks"] = roundtrips

    def _collect_lease_events(self) -> None:
        existing = copy.deepcopy(self.ledger.get("lease_events", []))
        events = list(self._lease_events)
        self._lease_events = []
        for worker_id in sorted(self._handles):
            handle = self._handles[worker_id]
            try:
                response = handle.client.request({"op": "events", "request_id": f"events-{worker_id}"})
                if response.get("ok"):
                    events.extend(response.get("events", []))
            except Exception as exc:
                events.append(
                    {
                        "code": E_RECONCILE_PENDING,
                        "worker_id": worker_id,
                        "message": f"cannot drain lease events: {exc}",
                    }
                )
        combined = existing + events
        deduplicated: dict[str, dict[str, Any]] = {}
        for event in combined:
            deduplicated[canonical_json(event)] = event
        events = list(deduplicated.values())
        events.sort(
            key=lambda item: (
                str(item.get("code")),
                str(item.get("worker_id", "")),
                str(item.get("batch_id", "")),
                int(item.get("event_seq", 0)),
                str(item.get("lease_id", "")),
            )
        )
        self.ledger["lease_events"] = events

    def _refresh_diagnostics(self) -> None:
        reuse_count = sum(1 for event in self.ledger["events"] if event.get("kind") == "reuse")
        invalid_count = sum(1 for event in self.ledger["events"] if event.get("kind") == "invalidate")
        expiry_count = sum(1 for event in self.ledger["lease_events"] if event.get("code") == "E_LEASE_EXPIRED")
        reconcile_count = sum(
            1 for event in self.ledger["lease_events"] if event.get("code") == "E_RECONCILE_PENDING"
        )
        incomplete = sorted(
            batch_id
            for batch_id, batch in self.ledger["run"]["batches"].items()
            if batch.get("status") != "complete"
        )
        self.ledger["diagnostics"] = {
            "reuse_count": reuse_count,
            "invalidation_count": invalid_count,
            "pending_queue": sorted(self.ledger.get("pending_batches", [])),
            "E_LEASE_EXPIRED": expiry_count,
            "E_RECONCILE_PENDING": reconcile_count,
            "incomplete_batches": incomplete,
            "worker_lease_events": copy.deepcopy(self.ledger["lease_events"]),
        }

    def run(self, *, rebuild: bool = False) -> dict[str, Any]:
        existing = None if rebuild else self.store.load()
        if existing is not None:
            if existing.get("base_config_hash") != self.base_state.runtime_config_hash():
                raise QLabError(
                    E_LEDGER_VERSION,
                    "base configuration hash changed; run a full rebuild",
                    stored=existing.get("base_config_hash"),
                    current=self.base_state.runtime_config_hash(),
                )
            stored_commands = list(existing.get("applied_commands", []))
            if not self.commands[: len(stored_commands)] == stored_commands:
                raise QLabError(
                    E_COMMAND,
                    "command stream does not extend the persisted ledger; use rebuild",
                    stored_count=len(stored_commands),
                )
        ledger = existing or empty_ledger(self.base_state.runtime_config_hash())
        self.ledger = ledger
        self._lease_events = []
        stored_count = len(ledger.get("applied_commands", []))
        new_commands = self.commands[stored_count:]
        state = self._replay(ledger.get("applied_commands", []))
        if existing is not None and ledger.get("state") and ledger.get("state") != state_to_dict(state):
            raise QLabError(E_LEDGER_CORRUPT, "persisted state does not match the replayed command prefix")
        ledger["state"] = state_to_dict(state)
        ledger["command_issues"] = self.issue_dicts
        if existing is None:
            ledger["pending_batches"] = []
            self._ensure_historical_deliveries(ledger)
            base_batches = list(state.batches)
            prepared = {
                batch.batch_id: {
                    "batch": batch,
                    "state": state,
                    "start_layer": 0,
                    "prefix_records": [],
                    "activation": np.asarray(batch.samples, dtype=np.float64),
                    "reason": "base_initial",
                }
                for batch in base_batches
            }
            if prepared:
                self._mark_pending(prepared, state, 0)
            else:
                self.store.save(ledger)
        else:
            self._ensure_historical_deliveries(ledger)

        with self.workers() as handles:
            self._handles = handles
            self._set_worker_limits(state, handles)
            self._resume_pending()
            if existing is None and ledger["run"]["batches"]:
                # Base batches were marked pending above and need one execution.
                prepared: dict[str, Any] = {}
                for batch_id, batch_result in sorted(ledger["run"]["batches"].items()):
                    if batch_result.get("status") != "pending":
                        continue
                    batch = next(item for item in state.batches if item.batch_id == batch_id)
                    prepared[batch_id] = {
                        "batch": batch,
                        "state": state,
                        "start_layer": 0,
                        "prefix_records": [],
                        "activation": np.asarray(batch.samples, dtype=np.float64),
                        "reason": "base_initial",
                    }
                if prepared:
                    self._execute_prepared(prepared, 0)
                    self.store.save(ledger)

            for command in new_commands:
                before = state.clone()
                state = before.clone()
                try:
                    apply_command(state, command)
                except (ValueError, KeyError) as exc:
                    raise QLabError(E_COMMAND, f"cannot apply accepted command {command.get('seq')}: {exc}") from exc
                ledger["applied_commands"].append(command)
                prepared = self._prepare_state_and_command(command, state)
                self._mark_pending(prepared, state, int(command["seq"]))
                if prepared:
                    self._execute_prepared(prepared, int(command["seq"]))
                self._refresh_cache()
                self._refresh_receipts()
                self._collect_lease_events()
                self._refresh_diagnostics()
                self.store.save(ledger)

            self._refresh_cache()
            self._refresh_receipts()
            self._collect_lease_events()
            self._refresh_diagnostics()
            self.store.save(ledger)

        return {
            "ledger_path": str(self.store.path),
            "snapshot_digest": snapshot_digest(ledger),
            "semantic_snapshot": semantic_snapshot(ledger),
            "recovery_report": recovery_report(ledger, ledger_path=self.store.path),
            "issue_count": len(self.issue_dicts),
            "command_count": len(ledger["applied_commands"]),
        }


def compare_ledgers(incremental: dict[str, Any], rebuilt: dict[str, Any]) -> dict[str, Any]:
    left = semantic_snapshot(incremental)
    right = semantic_snapshot(rebuilt)
    equal = canonical_json(left) == canonical_json(right)
    differences: list[dict[str, Any]] = []
    if not equal:
        def walk(a: Any, b: Any, path: str) -> None:
            if len(differences) >= 50:
                return
            if type(a) is not type(b):
                differences.append({"path": path, "incremental": a, "rebuilt": b})
                return
            if isinstance(a, dict):
                for key in sorted(set(a) | set(b)):
                    if key not in a or key not in b:
                        differences.append({"path": f"{path}.{key}", "incremental": a.get(key), "rebuilt": b.get(key)})
                    else:
                        walk(a[key], b[key], f"{path}.{key}")
                return
            if isinstance(a, list):
                if len(a) != len(b):
                    differences.append({"path": f"{path}.length", "incremental": len(a), "rebuilt": len(b)})
                for index, (left_item, right_item) in enumerate(zip(a, b)):
                    walk(left_item, right_item, f"{path}[{index}]")
                return
            if a != b:
                differences.append({"path": path, "incremental": a, "rebuilt": b})

        walk(left, right, "$")
    return {
        "equal": equal,
        "incremental_digest": digest_json(left),
        "rebuilt_digest": digest_json(right),
        "difference_count": len(differences),
        "differences": differences,
    }
