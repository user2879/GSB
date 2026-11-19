from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .comm import BudgetLedger, LoopbackHub
from .config import CompressionCandidate, SimConfig
from .data import ShardSet
from .decision import decide_window, plan_window_tasks
from .error_tracking import aggregate_fixed_point, aggregate_float64, track_tensor_errors
from .errors import SimError
from .injection import Injection
from .model import derive_window_gradients, initialize_parameters, sanitize_nonfinite
from .protocol import FRAME_HEADER_SIZE
from .quantization import QuantParams, compress_tensor, decode_packet, quant_params_from_dict
from .retention import FidelityStore
from .snapshot import SavedState, SnapshotManager
from .topology import Topology


PHASES = ("derived", "quantized", "communicated", "tracked")


class InjectedCrash(SimError):
    def __init__(self, phase: str, window: int) -> None:
        super().__init__(
            "E_INJECTED_CRASH",
            f"injected crash after {phase} phase of window {window}",
            phase=phase,
            window=window,
        )


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class Simulator:
    def __init__(
        self,
        *,
        config: SimConfig,
        shards: ShardSet,
        topology: Topology,
        snapshot_dir: str | None = None,
        injection: Injection | None = None,
        stop_event: threading.Event | None = None,
        max_windows: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.shards = shards
        self.topology = topology
        self.injection = injection
        self.stop_event = stop_event or threading.Event()
        self.max_windows = max_windows
        self.verbose = verbose
        self.manager = SnapshotManager(
            snapshot_dir or config.snapshot_dir,
            config_hash=config.config_hash,
            topology_hash=topology.topology_hash,
            data_hash=shards.data_hash,
            snapshot_interval_steps=config.snapshot_interval_steps,
            fidelity_generation_limit=config.fidelity_generations,
        )
        loaded = self.manager.load_latest()
        self.manager.validate_compatibility(loaded)
        if loaded is None:
            self.state = self._fresh_state()
        else:
            self._validate_loaded_state(loaded)
            self.state = loaded
        self.fidelity = FidelityStore(config.fidelity_generations, self.state.fidelity_entries)

    def _fresh_state(self) -> SavedState:
        base = initialize_parameters(self.config)
        params = {
            worker_id: {name: array.copy() for name, array in base.items()}
            for worker_id in range(self.config.workers)
        }
        return SavedState(params=params, phase="", completed_steps=0, window_index=-1)

    def _validate_loaded_state(self, state: SavedState) -> None:
        if state.phase not in {"", *PHASES}:
            raise SimError("E_SNAPSHOT_CORRUPT", f"unknown snapshot phase {state.phase!r}")
        expected_workers = set(range(self.config.workers))
        if set(state.params) != expected_workers:
            raise SimError("E_SNAPSHOT_CORRUPT", "snapshot parameter worker set does not match config")
        for worker_id in range(self.config.workers):
            if set(state.params[worker_id]) != set(self.config.tensor_names()):
                raise SimError("E_SNAPSHOT_CORRUPT", "snapshot parameter tensor set does not match config")
        if state.phase in {"derived", "quantized", "communicated"}:
            if set(state.grads) != expected_workers:
                raise SimError("E_SNAPSHOT_CORRUPT", "snapshot gradient worker set does not match config")
            expected_start, expected_end = self.config.window_bounds(state.window_index)
            if (state.window_start, state.window_end) != (expected_start, expected_end):
                raise SimError("E_SNAPSHOT_CORRUPT", "snapshot window bounds do not match config")
            if state.completed_steps != state.window_start:
                raise SimError(
                    "E_SNAPSHOT_CORRUPT",
                    "unfinished phase must keep the completed-step cursor at the window start",
                )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _commit(self, phase: str) -> None:
        self.manager.commit(self.state)
        self._log(
            f"committed phase={phase} window={self.state.window_index} "
            f"completed_steps={self.state.completed_steps} version={self.state.version}"
        )
        if self.injection and self.injection.should_crash_after_phase(phase, self.state.window_index):
            raise InjectedCrash(phase, self.state.window_index)

    def _begin_window(self, window_index: int) -> None:
        start, end = self.config.window_bounds(window_index)
        self.state.window_index = window_index
        self.state.window_start = start
        self.state.window_end = end
        self.state.phase = ""
        self.state.grads = {}
        self.state.packets = {}
        self.state.received_q = {}
        self.state.aggregate_fixed = {}
        self.state.selected = None
        self.state.quant_params = {}
        self.state.decision = None
        self.state.send_records = []
        self.state.current_errors = []
        self.state.budget_snapshot = {}

    def _phase_derive(self) -> None:
        if self.state.phase in {"derived", "quantized", "communicated"}:
            return
        start = self.state.window_start
        end = self.state.window_end
        self._log(f"window {self.state.window_index}: deriving gradients for steps [{start}, {end})")
        gradients = derive_window_gradients(self.config, self.shards, start, end)
        if self.injection:
            self.injection.apply_nonfinite(
                gradients, self.state.window_index, self.state.window_start, self.state.window_end
            )

        cleaned_grads, gradient_failures = sanitize_nonfinite(gradients)
        cleaned_params, parameter_failures = sanitize_nonfinite(self.state.params)
        for failure in gradient_failures:
            failure["window"] = self.state.window_index
            failure["phase"] = "gradient_derivation"
            failure["message"] = (
                f"non-finite gradient for worker {failure['worker_id']} tensor {failure['tensor']}"
            )
        for failure in parameter_failures:
            failure["window"] = self.state.window_index
            failure["phase"] = "parameter_scan"
            failure["message"] = (
                f"non-finite parameter for worker {failure['worker_id']} tensor {failure['tensor']}"
            )
        self.state.params = cleaned_params
        self.state.grads = cleaned_grads
        self.state.current_errors.extend(gradient_failures)
        self.state.current_errors.extend(parameter_failures)
        self.state.phase = "derived"
        self._commit("derived")

    def _tasks_from_state(self) -> list[dict[str, Any]]:
        if not self.state.decision:
            raise SimError("E_SNAPSHOT_CORRUPT", "quantized snapshot has no decision record")
        ordered = [item["tensor"] for item in self.state.decision["ordered_tensors"]]
        packet_bytes = self.state.decision["selected_packet_bytes_by_tensor"]
        bits = int(self.state.selected["bits"])
        tasks = plan_window_tasks(
            topology=self.topology,
            window=self.state.window_index,
            ordered_tensors=ordered,
            packet_bytes=packet_bytes,
            bits=bits,
        )
        for task in tasks:
            task["candidate_scores"] = self.state.decision["candidate_scores"]
            task["selection_reason"] = self.state.decision["selected_reason"]
        return tasks

    def _phase_quantize(self) -> None:
        if self.state.phase in {"quantized", "communicated"}:
            return
        overrides = self.injection.budget_overrides() if self.injection else None
        decision_result = decide_window(
            config=self.config,
            topology=self.topology,
            window=self.state.window_index,
            gradients=self.state.grads,
            budget_overrides=overrides,
        )
        selected = decision_result.selected
        packets: dict[int, dict[str, bytes]] = {}
        compression_failures: list[dict[str, Any]] = []
        for worker_id in range(self.config.workers):
            packets[worker_id] = {}
            for tensor_name in sorted(self.config.tensor_names()):
                try:
                    packet = compress_tensor(
                        self.state.grads[worker_id][tensor_name],
                        tensor_name,
                        selected.bits,
                        selected.scheme,
                        decision_result.quant_params[tensor_name],
                    )
                    expected_size = decision_result.decision["selected_packet_bytes_by_tensor"][
                        tensor_name
                    ]
                    if len(packet) != expected_size:
                        raise SimError(
                            "E_QUANT_LENGTH_MISMATCH",
                            f"packet length {len(packet)} differs from decision length {expected_size}",
                            tensor=tensor_name,
                        )
                    packets[worker_id][tensor_name] = packet
                except BaseException as exc:
                    compression_failures.append(
                        self._failure(
                            getattr(exc, "code", "E_QUANTIZATION_FAILED"),
                            self.state.window_index,
                            worker_id,
                            tensor_name,
                            type(exc).__name__,
                            str(exc),
                        )
                    )
        self.state.packets = packets
        self.state.selected = {
            "scheme": selected.scheme,
            "bits": selected.bits,
            "label": selected.label,
        }
        self.state.quant_params = {
            name: params.as_dict()
            for name, params in sorted(decision_result.quant_params.items())
        }
        self.state.decision = decision_result.decision
        self.state.current_errors.extend(compression_failures)
        self.state.phase = "quantized"
        self._commit("quantized")

    @staticmethod
    def _failure(
        code: str,
        window: int,
        worker_id: int | None,
        tensor: str | None,
        exception_type: str,
        message: str,
        **extra: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "code": code,
            "window": window,
            "worker_id": worker_id,
            "tensor": tensor,
            "exception_type": exception_type,
            "message": message,
        }
        record.update(extra)
        return record

    def _batch_tasks(self, tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not tasks:
            return []
        if self.topology.kind == "full":
            return [
                sorted(
                    tasks,
                    key=lambda task: (
                        task["origin_worker"],
                        task["tensor_rank"],
                        task["dst_worker"],
                        task["route_round"],
                    ),
                )
            ]
        batches: list[list[dict[str, Any]]] = []
        for route_round in range(max(task["route_round"] for task in tasks) + 1):
            batch = [task for task in tasks if task["route_round"] == route_round]
            batch.sort(
                key=lambda task: (
                    task["origin_worker"],
                    task["tensor_rank"],
                    task["dst_worker"],
                )
            )
            batches.append(batch)
        return batches

    def _send_one(
        self,
        task: dict[str, Any],
        packet: bytes,
        ledger: BudgetLedger,
        hub: LoopbackHub,
        sequence: int,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        window = int(task["window"])
        link = f"{task['src_holder']}->{task['dst_worker']}"
        record: dict[str, Any] = {
            "execution_sequence": sequence,
            "stable_key": list(task["stable_key"]),
            "window": window,
            "origin_worker": task["origin_worker"],
            "src_holder": task["src_holder"],
            "dst_worker": task["dst_worker"],
            "route_round": task["route_round"],
            "tensor": task["tensor"],
            "bits": task["bits"],
            "packet_bytes": len(packet),
            "wire_bytes": len(packet) + FRAME_HEADER_SIZE,
            "link": link,
            "sent": False,
            "candidate_scores": task["candidate_scores"],
            "selection_reason": task["selection_reason"],
        }
        if self.injection:
            injected = self.injection.communication_failure(task, window)
            if injected:
                failure = self._failure(
                    "E_COMM_SEND_FAILED",
                    window,
                    int(task["origin_worker"]),
                    str(task["tensor"]),
                    injected,
                    f"injected communication failure for {link}",
                    link=link,
                    dst_worker=int(task["dst_worker"]),
                )
                return record, failure
            delay = self.injection.send_delay_seconds()
            if delay > 0:
                time.sleep(delay)
        reserved, sent_before, budget = ledger.reserve(link, int(record["wire_bytes"]))
        if not reserved:
            failure = self._failure(
                "E_COMM_BUDGET_EXHAUSTED",
                window,
                int(task["origin_worker"]),
                str(task["tensor"]),
                "CommunicationBudgetExceeded",
                f"link {link} refused {record['wire_bytes']} bytes after {sent_before} bytes",
                link=link,
                sent_bytes=sent_before,
                requested_bytes=int(record["wire_bytes"]),
                budget_bytes=int(budget),
                dst_worker=int(task["dst_worker"]),
            )
            return record, failure
        try:
            wire_size = hub.send(
                origin_worker=int(task["origin_worker"]),
                src_holder=int(task["src_holder"]),
                dst_worker=int(task["dst_worker"]),
                route_round=int(task["route_round"]),
                window=window,
                packet=packet,
            )
            record["wire_bytes"] = wire_size
            record["sent"] = True
            return record, None
        except BaseException as exc:
            ledger.release(link, int(record["wire_bytes"]))
            failure = self._failure(
                getattr(exc, "code", "E_COMM_SEND_FAILED"),
                window,
                int(task["origin_worker"]),
                str(task["tensor"]),
                type(exc).__name__,
                str(exc),
                link=link,
                dst_worker=int(task["dst_worker"]),
            )
            return record, failure

    def _wait_for_expected(
        self,
        hub: LoopbackHub,
        expected_by_dst: dict[int, set[tuple[int, int, str]]],
        batch_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        record_by_key = {
            (
                int(record["dst_worker"]),
                int(record["window"]),
                int(record["origin_worker"]),
                str(record["tensor"]),
            ): record
            for record in batch_records
            if record["sent"]
        }
        for dst in sorted(expected_by_dst):
            missing = hub.receivers[dst].wait_for_keys(sorted(expected_by_dst[dst]), timeout=5.0)
            for key in missing:
                record = record_by_key.get((dst, key[0], key[1], key[2]))
                if record is None:
                    continue
                failures.append(
                    self._failure(
                        "E_COMM_RECEIVE_MISSING",
                        key[0],
                        int(record["origin_worker"]),
                        key[2],
                        "MissingReceivedFrame",
                        f"sent frame was not delivered to worker {dst}",
                        link=record["link"],
                        dst_worker=dst,
                    )
                )
        return failures

    def _execute_communication(self) -> None:
        if self.state.phase == "communicated":
            return
        tasks = self._tasks_from_state()
        if not self.state.decision:
            raise SimError("E_SNAPSHOT_CORRUPT", "missing decision for communication phase")
        link_budgets = {
            str(key): int(value) for key, value in self.state.decision["link_budgets"].items()
        }
        ledger = BudgetLedger(link_budgets, self.config.compression.max_bytes_per_window)
        all_records: list[dict[str, Any]] = []
        all_failures: list[dict[str, Any]] = []
        sequence = 0
        self._log(
            f"window {self.state.window_index}: communicating {len(tasks)} planned sends "
            f"with concurrency={self.config.concurrency}"
        )
        with LoopbackHub(self.config.workers) as hub:
            for batch in self._batch_tasks(tasks):
                prepared: list[tuple[dict[str, Any], bytes, int]] = []
                for task in batch:
                    if int(task["src_holder"]) == int(task["origin_worker"]):
                        packet = self.state.packets.get(task["origin_worker"], {}).get(task["tensor"])
                    else:
                        packet = hub.receivers[int(task["src_holder"])].get_packet(
                            self.state.window_index, int(task["origin_worker"]), str(task["tensor"])
                        )
                    if packet is None:
                        all_failures.append(
                            self._failure(
                                "E_COMM_PACKET_UNAVAILABLE",
                                self.state.window_index,
                                int(task["origin_worker"]),
                                str(task["tensor"]),
                                "MissingForwardPacket",
                                f"worker {task['src_holder']} has no packet to forward to {task['dst_worker']}",
                                link=f"{task['src_holder']}->{task['dst_worker']}",
                            )
                        )
                        all_records.append(
                            {
                                "execution_sequence": sequence,
                                "stable_key": list(task["stable_key"]),
                                "window": int(task["window"]),
                                "origin_worker": int(task["origin_worker"]),
                                "src_holder": int(task["src_holder"]),
                                "dst_worker": int(task["dst_worker"]),
                                "route_round": int(task["route_round"]),
                                "tensor": str(task["tensor"]),
                                "bits": int(task["bits"]),
                                "packet_bytes": int(task["packet_bytes"]),
                                "wire_bytes": int(task["wire_bytes"]),
                                "link": f"{task['src_holder']}->{task['dst_worker']}",
                                "sent": False,
                                "candidate_scores": task["candidate_scores"],
                                "selection_reason": task["selection_reason"],
                            }
                        )
                        sequence += 1
                        continue
                    prepared.append((task, packet, sequence))
                    sequence += 1
                if not prepared:
                    continue
                bandwidth_limit = int(
                    self.state.decision.get(
                        "bandwidth_concurrency_limit", self.config.concurrency
                    )
                )
                effective_concurrency = max(1, min(bandwidth_limit, len(prepared)))
                with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                    futures = [
                        (
                            task,
                            packet,
                            seq,
                            executor.submit(self._send_one, task, packet, ledger, hub, seq),
                        )
                        for task, packet, seq in prepared
                    ]
                    batch_records: list[dict[str, Any]] = []
                    for task, packet, seq, future in futures:
                        try:
                            record, failure = future.result()
                        except BaseException as exc:
                            record = {
                                "execution_sequence": seq,
                                "stable_key": list(task["stable_key"]),
                                "window": int(task["window"]),
                                "origin_worker": int(task["origin_worker"]),
                                "src_holder": int(task["src_holder"]),
                                "dst_worker": int(task["dst_worker"]),
                                "route_round": int(task["route_round"]),
                                "tensor": str(task["tensor"]),
                                "bits": int(task["bits"]),
                                "packet_bytes": len(packet),
                                "wire_bytes": len(packet) + FRAME_HEADER_SIZE,
                                "link": f"{task['src_holder']}->{task['dst_worker']}",
                                "sent": False,
                                "candidate_scores": task["candidate_scores"],
                                "selection_reason": task["selection_reason"],
                            }
                            failure = self._failure(
                                getattr(exc, "code", "E_COMM_SEND_FAILED"),
                                self.state.window_index,
                                int(task["origin_worker"]),
                                str(task["tensor"]),
                                type(exc).__name__,
                                str(exc),
                                link=record["link"],
                            )
                        batch_records.append(record)
                        if failure:
                            all_failures.append(failure)
                all_records.extend(batch_records)
                expected_by_dst: dict[int, set[tuple[int, int, str]]] = {}
                for record in batch_records:
                    if not record["sent"]:
                        continue
                    expected_by_dst.setdefault(int(record["dst_worker"]), set()).add(
                        (
                            int(record["window"]),
                            int(record["origin_worker"]),
                            str(record["tensor"]),
                        )
                    )
                all_failures.extend(self._wait_for_expected(hub, expected_by_dst, batch_records))
                for receiver in hub.receivers.values():
                    while receiver.receive_errors:
                        receive_error = receiver.receive_errors.pop(0)
                        all_failures.append(
                            self._failure(
                                str(receive_error.get("code", "E_COMM_RECEIVE_FAILED")),
                                self.state.window_index,
                                int(receive_error.get("worker_id", receiver.worker_id)),
                                None,
                                str(receive_error.get("exception_type", "ReceiveError")),
                                str(receive_error.get("message", "receive failure")),
                            )
                        )

            received_q, aggregate_fixed, aggregation_failures = self._aggregate_from_sockets(hub)
        all_failures.extend(aggregation_failures)
        all_records.sort(key=lambda record: tuple(record["stable_key"]))
        self.state.send_records = all_records
        self.state.current_errors.extend(
            sorted(all_failures, key=lambda record: self._error_sort_key(record))
        )
        self.state.received_q = received_q
        self.state.aggregate_fixed = aggregate_fixed
        budget_snapshot = ledger.snapshot()
        budget_snapshot["actual_execution_order"] = [
            record["stable_key"] for record in sorted(all_records, key=lambda item: item["execution_sequence"])
        ]
        budget_snapshot["concurrency_limit"] = self.config.concurrency
        self.state.budget_snapshot = budget_snapshot
        self.state.phase = "communicated"
        self._commit("communicated")

    def _aggregate_from_sockets(
        self, hub: LoopbackHub
    ) -> tuple[
        dict[tuple[int, int, str], np.ndarray],
        dict[int, dict[str, np.ndarray]],
        list[dict[str, Any]],
    ]:
        received_q: dict[tuple[int, int, str], np.ndarray] = {}
        aggregate_fixed: dict[int, dict[str, np.ndarray]] = {}
        failures: list[dict[str, Any]] = []
        selected_bits = int(self.state.selected["bits"]) if self.state.selected else -1
        selected_scheme = str(self.state.selected["scheme"]) if self.state.selected else ""
        for dst in range(self.config.workers):
            aggregate_fixed[dst] = {}
            for tensor_name in sorted(self.config.tensor_names()):
                own_packet = self.state.packets.get(dst, {}).get(tensor_name)
                if own_packet is None:
                    failures.append(
                        self._failure(
                            "E_COMM_PACKET_UNAVAILABLE",
                            self.state.window_index,
                            dst,
                            tensor_name,
                            "MissingLocalPacket",
                            "local compressed packet is unavailable",
                        )
                    )
                    continue
                try:
                    own_decoded = decode_packet(
                        own_packet,
                        expected_name=tensor_name,
                        expected_bits=selected_bits,
                        expected_scheme=selected_scheme,
                    )
                except BaseException as exc:
                    failures.append(
                        self._failure(
                            getattr(exc, "code", "E_QUANT_HEADER_INVALID"),
                            self.state.window_index,
                            dst,
                            tensor_name,
                            type(exc).__name__,
                            str(exc),
                        )
                    )
                    continue
                q_by_origin: dict[int, np.ndarray] = {dst: own_decoded.q}
                params = own_decoded.header
                for origin in range(self.config.workers):
                    if origin == dst:
                        continue
                    packet = hub.receivers[dst].get_packet(
                        self.state.window_index, origin, tensor_name
                    )
                    if packet is None:
                        failures.append(
                            self._failure(
                                "E_COMM_RECEIVE_MISSING",
                                self.state.window_index,
                                origin,
                                tensor_name,
                                "MissingReceivedFrame",
                                f"worker {dst} did not receive origin {origin}",
                                dst_worker=dst,
                            )
                        )
                        continue
                    try:
                        decoded = decode_packet(
                            packet,
                            expected_name=tensor_name,
                            expected_bits=selected_bits,
                            expected_scheme=selected_scheme,
                        )
                    except BaseException as exc:
                        failures.append(
                            self._failure(
                                getattr(exc, "code", "E_QUANT_HEADER_INVALID"),
                                self.state.window_index,
                                origin,
                                tensor_name,
                                type(exc).__name__,
                                str(exc),
                                dst_worker=dst,
                            )
                        )
                        continue
                    if (
                        decoded.header.scale != params.scale
                        or decoded.header.zero_point != params.zero_point
                    ):
                        failures.append(
                            self._failure(
                                "E_QUANT_PARAM_MISMATCH",
                                self.state.window_index,
                                origin,
                                tensor_name,
                                "QuantizationParameterMismatch",
                                "workers did not use common fixed-point parameters",
                                dst_worker=dst,
                            )
                        )
                        continue
                    q_by_origin[origin] = decoded.q
                    received_q[(dst, origin, tensor_name)] = decoded.q
                try:
                    values, _, _ = aggregate_fixed_point(
                        q_by_origin, params, self.config.aggregation.mode
                    )
                    aggregate_fixed[dst][tensor_name] = values
                except BaseException as exc:
                    failures.append(
                        self._failure(
                            getattr(exc, "code", "E_AGGREGATION_FAILED"),
                            self.state.window_index,
                            dst,
                            tensor_name,
                            type(exc).__name__,
                            str(exc),
                        )
                    )
        return received_q, aggregate_fixed, failures

    def _phase_track(self) -> None:
        if self.state.phase == "tracked":
            return
        if not self.state.selected or not self.state.quant_params:
            raise SimError("E_SNAPSHOT_CORRUPT", "communicated snapshot has no quantization decision")
        lower_candidate = self._select_lower_candidate()
        tracking_records: list[dict[str, Any]] = []
        update_failures: list[dict[str, Any]] = []
        coverage_complete = True
        for dst in range(self.config.workers):
            for tensor_name in sorted(self.config.tensor_names()):
                params = quant_params_from_dict(self.state.quant_params[tensor_name])
                originals = {
                    origin: self.state.grads[origin][tensor_name]
                    for origin in range(self.config.workers)
                }
                source_q: dict[int, np.ndarray] = {}
                for origin in range(self.config.workers):
                    packet = self.state.packets.get(origin, {}).get(tensor_name)
                    if packet is None:
                        continue
                    try:
                        source_q[origin] = decode_packet(
                            packet,
                            expected_name=tensor_name,
                            expected_bits=params.bits,
                            expected_scheme=params.scheme,
                        ).q
                    except BaseException as exc:
                        update_failures.append(
                            self._failure(
                                getattr(exc, "code", "E_QUANT_HEADER_INVALID"),
                                self.state.window_index,
                                origin,
                                tensor_name,
                                type(exc).__name__,
                                str(exc),
                            )
                        )
                if not source_q:
                    coverage_complete = False
                    update_failures.append(
                        self._failure(
                            "E_TRACKING_SOURCE_UNAVAILABLE",
                            self.state.window_index,
                            dst,
                            tensor_name,
                            "MissingQuantizedSource",
                            "no compressed source packet was available for error tracking",
                        )
                    )
                    continue
                received_q = {
                    origin: q
                    for (record_dst, origin, name), q in self.state.received_q.items()
                    if record_dst == dst and name == tensor_name
                }
                if dst in source_q:
                    received_q[dst] = source_q[dst]
                complete = len(received_q) == self.config.workers
                coverage_complete = coverage_complete and complete
                lower_params = None
                if lower_candidate is not None:
                    lower_params = self._lower_quant_params(tensor_name, lower_candidate)
                tracking_records.append(
                    track_tensor_errors(
                        window=self.state.window_index,
                        worker_id=dst,
                        tensor_name=tensor_name,
                        originals_by_origin=originals,
                        source_q_by_origin=source_q,
                        received_q_by_origin=received_q,
                        params=params,
                        mode=self.config.aggregation.mode,
                        tolerance=self.config.communication_error_tolerance,
                        lower_params=lower_params,
                        complete=complete,
                    )
                )
        tracking_records.sort(key=lambda record: (record["worker_id"], record["tensor"]))

        for worker_id in range(self.config.workers):
            for tensor_name in sorted(self.config.tensor_names()):
                aggregate = self.state.aggregate_fixed.get(worker_id, {}).get(tensor_name)
                if aggregate is None:
                    continue
                params = quant_params_from_dict(self.state.quant_params[tensor_name])
                current = self.state.params[worker_id][tensor_name].astype(np.float64)
                updated = current + np.asarray(aggregate, dtype=np.float64)
                rounded = np.rint(updated / params.scale) * params.scale
                candidate = rounded.astype(np.float32)
                if not np.all(np.isfinite(candidate)):
                    update_failures.append(
                        self._failure(
                            "E_PARAM_NONFINITE",
                            self.state.window_index,
                            worker_id,
                            tensor_name,
                            "NonFiniteParameterError",
                            "parameter update produced non-finite values; update skipped",
                        )
                    )
                    continue
                self.state.params[worker_id][tensor_name] = candidate
        update_failures.extend(self._scan_params())

        exact_gradients: dict[str, np.ndarray] = {}
        for tensor_name in sorted(self.config.tensor_names()):
            arrays = {
                worker_id: self.state.grads[worker_id][tensor_name]
                for worker_id in range(self.config.workers)
            }
            truth, _ = aggregate_float64(arrays, self.config.aggregation.mode)
            exact_gradients[tensor_name] = np.asarray(truth, dtype=np.float32)
        before_events = len(self.fidelity.rotation_events)
        retained = self.fidelity.add(
            self.state.window_index, self.state.window_end, exact_gradients
        )
        new_rotation_events = self.fidelity.rotation_events[before_events:]
        self.state.retention_events.extend(new_rotation_events)

        errors = sorted(
            self.state.current_errors + update_failures,
            key=lambda record: self._error_sort_key(record),
        )
        window_record = {
            "window": self.state.window_index,
            "window_start": self.state.window_start,
            "window_end": self.state.window_end,
            "selected": self.state.selected,
            "decision": self.state.decision,
            "send_records": self.state.send_records,
            "budget": self.state.budget_snapshot,
            "errors": errors,
            "error_tracking": tracking_records,
            "boundaries": [
                record["boundary"] for record in tracking_records if record.get("boundary") is not None
            ],
            "retention": {
                "retained_gradient_id": retained.gradient_id if retained else None,
                "retained_gradient_ids_after_commit": self.fidelity.ids(),
                "rotation_events": new_rotation_events,
                "retention_generations": self.config.fidelity_generations,
            },
            "aggregation_complete_for_all_workers": coverage_complete,
            "parameter_update_rule": "elementwise add then round to selected fixed-point quantum",
        }
        self.state.window_records.append(window_record)
        self.state.current_errors = []
        self.state.budget_snapshot = {}
        self.state.fidelity_entries = self.fidelity.entries
        self.state.phase = "tracked"
        self.state.completed_steps = self.state.window_end
        self._commit("tracked")

    def _select_lower_candidate(self) -> CompressionCandidate | None:
        if not self.state.selected:
            return None
        selected_scheme = str(self.state.selected["scheme"])
        selected_bits = int(self.state.selected["bits"])
        same_scheme = [
            candidate
            for candidate in self.config.compression.candidates
            if candidate.scheme == selected_scheme and candidate.bits < selected_bits
        ]
        if same_scheme:
            return max(same_scheme, key=lambda candidate: candidate.bits)
        any_lower = [
            candidate
            for candidate in self.config.compression.candidates
            if candidate.bits < selected_bits
        ]
        if any_lower:
            return max(any_lower, key=lambda candidate: (candidate.bits, -candidate.scheme_rank))
        return None

    def _lower_quant_params(
        self, tensor_name: str, candidate: CompressionCandidate
    ) -> QuantParams:
        arrays = [
            self.state.grads[worker_id][tensor_name]
            for worker_id in range(self.config.workers)
        ]
        from .quantization import compute_common_quant_params

        return compute_common_quant_params(tensor_name, arrays, candidate.bits, candidate.scheme)

    def _scan_params(self) -> list[dict[str, Any]]:
        self.state.params, failures = sanitize_nonfinite(self.state.params)
        for failure in failures:
            failure["window"] = self.state.window_index
            failure["phase"] = "parameter_scan"
            failure["message"] = (
                f"non-finite parameter for worker {failure['worker_id']} tensor {failure['tensor']}"
            )
        return failures

    @staticmethod
    def _error_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        def as_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        return (
            as_int(record.get("window", -1)),
            as_int(record.get("worker_id", -1)),
            str(record.get("tensor", "")),
            str(record.get("code", "")),
            str(record.get("link", "")),
        )

    def build_report(self, *, stopped_reason: str, exit_code: int) -> dict[str, Any]:
        all_errors: list[dict[str, Any]] = []
        all_tracking: list[dict[str, Any]] = []
        all_boundaries: list[dict[str, Any]] = []
        total_wire = 0
        for window_record in self.state.window_records:
            all_errors.extend(window_record.get("errors", []))
            all_tracking.extend(window_record.get("error_tracking", []))
            all_boundaries.extend(window_record.get("boundaries", []))
            total_wire += int(window_record.get("budget", {}).get("total_sent_bytes", 0))
        all_errors.sort(key=self._error_sort_key)
        all_tracking.sort(key=lambda record: (record["window"], record["worker_id"], record["tensor"]))
        param_hashes = {
            worker_id: _array_hash(self.state.params[worker_id])
            for worker_id in sorted(self.state.params)
        }
        parameter_values = [self.state.params[worker_id] for worker_id in sorted(self.state.params)]
        params_equal = True
        if parameter_values:
            reference = parameter_values[0]
            params_equal = all(
                set(candidate) == set(reference)
                and all(np.array_equal(candidate[name], reference[name]) for name in reference)
                for candidate in parameter_values[1:]
            )
        report = {
            "status": "completed" if self.state.completed_steps >= self.config.steps else "stopped",
            "stopped_reason": stopped_reason,
            "exit_code": exit_code,
            "config_hash": self.config.config_hash,
            "topology_hash": self.topology.topology_hash,
            "data_hash": self.shards.data_hash,
            "snapshot_dir": str(self.manager.root),
            "snapshot_version": self.state.version,
            "snapshot_phase": self.state.phase,
            "snapshot_interval_steps": self.config.snapshot_interval_steps,
            "completed_steps": self.state.completed_steps,
            "total_steps": self.config.steps,
            "next_window_index": self.state.completed_steps // self.config.compression.window_steps,
            "parameter_hashes": param_hashes,
            "parameters_equal_across_workers": params_equal,
            "total_wire_bytes_sent": total_wire,
            "fidelity_generations_limit": self.config.fidelity_generations,
            "retained_fidelity_ids": self.fidelity.ids(),
            "retention_events": self.state.retention_events,
            "errors": all_errors,
            "error_count": len(all_errors),
            "error_tracking": all_tracking,
            "boundary_tensors": all_boundaries,
            "windows": self.state.window_records,
            "warnings": self.state.warnings,
        }
        return report

    def run(self) -> dict[str, Any]:
        processed_windows = 0
        stopped_reason = "completed"
        completed_before = self.state.completed_steps
        try:
            while self.state.completed_steps < self.config.steps:
                if self.stop_event.is_set() and self.state.phase in {"", "tracked"}:
                    stopped_reason = "signal"
                    break
                if self.state.phase in {"", "tracked"}:
                    if self.max_windows is not None and processed_windows >= self.max_windows:
                        stopped_reason = "max_windows"
                        break
                    window_index = self.state.completed_steps // self.config.compression.window_steps
                    self._begin_window(window_index)
                elif self.state.phase not in {"derived", "quantized", "communicated"}:
                    raise SimError(
                        "E_SNAPSHOT_CORRUPT", f"cannot resume from phase {self.state.phase!r}"
                    )

                self._phase_derive()
                self._phase_quantize()
                self._execute_communication()
                self._phase_track()
                processed_windows += 1

                if self.stop_event.is_set():
                    stopped_reason = "signal"
                    break
                if self.max_windows is not None and processed_windows >= self.max_windows:
                    stopped_reason = "max_windows"
                    break
            if self.state.completed_steps >= self.config.steps:
                stopped_reason = "completed"
        finally:
            report = self.build_report(stopped_reason=stopped_reason, exit_code=0)
            report["windows_processed_this_run"] = processed_windows
            report["completed_steps_before_run"] = completed_before
            self.manager.write_report(report)
        return report
