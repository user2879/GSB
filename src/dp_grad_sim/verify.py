from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import SimConfig
from .data import ShardSet
from .quantization import decode_packet
from .snapshot import SnapshotManager
from .topology import Topology


def verify_snapshot(
    *,
    config: SimConfig,
    shards: ShardSet,
    topology: Topology,
    snapshot_dir: str,
) -> dict[str, Any]:
    manager = SnapshotManager(
        snapshot_dir,
        config_hash=config.config_hash,
        topology_hash=topology.topology_hash,
        data_hash=shards.data_hash,
        snapshot_interval_steps=config.snapshot_interval_steps,
        fidelity_generation_limit=config.fidelity_generations,
    )
    state = manager.load_latest()
    manager.validate_compatibility(state)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        entry: dict[str, Any] = {"name": name, "passed": bool(passed)}
        if detail is not None:
            entry["detail"] = detail
        checks.append(entry)

    if state is None:
        check("snapshot_exists", False, "no committed snapshot")
        return {
            "ok": False,
            "snapshot_dir": str(manager.root),
            "checks": checks,
        }

    check("snapshot_exists", True, state.manifest.get("version"))
    check("config_hash_matches", state.manifest.get("config_hash") == config.config_hash)
    check("topology_hash_matches", state.manifest.get("topology_hash") == topology.topology_hash)
    check("data_hash_matches", state.manifest.get("data_hash") == shards.data_hash)
    check("phase_known", state.phase in {"derived", "quantized", "communicated", "tracked"})
    check("completed_steps_range", 0 <= state.completed_steps <= config.steps)

    params_finite = all(
        np.all(np.isfinite(array))
        for worker in state.params.values()
        for array in worker.values()
    )
    check("parameters_are_finite", params_finite)
    expected_tensor_names = set(config.tensor_names())
    check(
        "parameter_tensor_inventory",
        all(set(worker) == expected_tensor_names for worker in state.params.values()),
    )

    window_indices = [int(record["window"]) for record in state.window_records]
    check(
        "window_records_are_unique_and_ordered",
        window_indices == sorted(set(window_indices)),
        window_indices,
    )
    expected_completed_windows = math.ceil(state.completed_steps / config.compression.window_steps)
    if state.phase in {"derived", "quantized", "communicated"}:
        expected_completed_windows = state.window_start // config.compression.window_steps
    check(
        "completed_window_count_matches_cursor",
        len(state.window_records) == expected_completed_windows,
        {
            "records": len(state.window_records),
            "expected": expected_completed_windows,
            "phase": state.phase,
        },
    )

    budget_ok = True
    decision_send_ok = True
    ordering_ok = True
    for record in state.window_records:
        budget = record.get("budget", {})
        link_budgets = {str(key): int(value) for key, value in budget.get("link_budgets", {}).items()}
        sent_by_link = {link: 0 for link in link_budgets}
        for send in record.get("send_records", []):
            if send.get("sent"):
                link = str(send["link"])
                sent_by_link[link] = sent_by_link.get(link, 0) + int(send["wire_bytes"])
        if any(
            sent_by_link.get(link, 0) > limit for link, limit in link_budgets.items()
        ):
            budget_ok = False
        if int(budget.get("total_sent_bytes", 0)) > config.compression.max_bytes_per_window:
            budget_ok = False
        successful_wire_bytes = sum(
            int(send["wire_bytes"]) for send in record.get("send_records", []) if send.get("sent")
        )
        if int(budget.get("total_sent_bytes", 0)) != successful_wire_bytes:
            budget_ok = False
        planned = int(record.get("decision", {}).get("planned_send_count", -1))
        if planned != len(record.get("send_records", [])):
            decision_send_ok = False
        keys = [tuple(send["stable_key"]) for send in record.get("send_records", [])]
        if keys != sorted(keys):
            ordering_ok = False
        tracking = record.get("error_tracking", [])
        tracking_keys = [(item["worker_id"], item["tensor"]) for item in tracking]
        if tracking_keys != sorted(tracking_keys):
            ordering_ok = False
    check("window_budgets_respected", budget_ok)
    check("decision_records_match_send_records", decision_send_ok)
    check("deterministic_report_ordering", ordering_ok)

    fidelity_ids = [entry.gradient_id for entry in state.fidelity_entries]
    check(
        "fidelity_generation_limit_respected",
        len(fidelity_ids) <= config.fidelity_generations,
        {
            "retained": len(fidelity_ids),
            "limit": config.fidelity_generations,
        },
    )
    check("fidelity_ids_unique", len(fidelity_ids) == len(set(fidelity_ids)))
    fidelity_shapes_ok = all(
        set(entry.tensors) == expected_tensor_names
        and all(
            tuple(entry.tensors[name].shape) == tuple(tensor["shape"])
            for name in entry.tensors
            for tensor in config.model.expanded_tensors()
            if tensor["name"] == name
        )
        for entry in state.fidelity_entries
    )
    check("fidelity_tensor_inventory_and_shapes", fidelity_shapes_ok)

    packets_ok = True
    if state.packets and state.selected:
        try:
            for worker_id in sorted(state.packets):
                for tensor_name, packet in sorted(state.packets[worker_id].items()):
                    decoded = decode_packet(
                        packet,
                        expected_name=tensor_name,
                        expected_bits=int(state.selected["bits"]),
                        expected_scheme=str(state.selected["scheme"]),
                    )
                    if state.decision:
                        expected_size = state.decision["selected_packet_bytes_by_tensor"][tensor_name]
                        if len(packet) != int(expected_size):
                            packets_ok = False
                    if tuple(decoded.values.shape) != tuple(
                        next(
                            tensor["shape"]
                            for tensor in config.model.expanded_tensors()
                            if tensor["name"] == tensor_name
                        )
                    ):
                        packets_ok = False
        except Exception:
            packets_ok = False
    check("compressed_packets_decode_and_match_decision", packets_ok)
    retained_references_ok = True
    historical_ids: set[str] = set()
    for event in state.retention_events:
        if event.get("inserted_gradient_id"):
            historical_ids.add(str(event["inserted_gradient_id"]))
        if event.get("replaced_gradient_id"):
            historical_ids.add(str(event["replaced_gradient_id"]))
    for record in state.window_records:
        retained_id = record.get("retention", {}).get("retained_gradient_id")
        if retained_id is not None and retained_id not in set(fidelity_ids) | historical_ids:
            retained_references_ok = False
    check("fidelity_references_are_reconstructable", retained_references_ok)
    referenced_current_ids = {
        str(record.get("retention", {}).get("retained_gradient_id"))
        for record in state.window_records
        if record.get("retention", {}).get("retained_gradient_id") in fidelity_ids
    }
    check(
        "current_fidelity_entries_are_referenced",
        set(fidelity_ids).issubset(referenced_current_ids),
        {
            "current": fidelity_ids,
            "referenced": sorted(referenced_current_ids),
        },
    )

    parameter_hashes = {
        worker_id: _hash_arrays(worker, expected_tensor_names)
        for worker_id, worker in sorted(state.params.items())
    }
    params_equal = len(set(parameter_hashes.values())) <= 1
    report = {
        "ok": all(item["passed"] for item in checks),
        "snapshot_dir": str(manager.root),
        "snapshot_version": state.version,
        "phase": state.phase,
        "completed_steps": state.completed_steps,
        "total_steps": config.steps,
        "config_hash": config.config_hash,
        "topology_hash": topology.topology_hash,
        "data_hash": shards.data_hash,
        "parameter_hashes": parameter_hashes,
        "parameters_equal_across_workers": params_equal,
        "retained_fidelity_ids": fidelity_ids,
        "checks": checks,
    }
    return report


def _hash_arrays(arrays: dict[str, np.ndarray], names: set[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(names):
        array = np.asarray(arrays[name], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
