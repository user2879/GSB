"""Session-spec execution and deterministic local sample generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import DiagnosticError, E_FORMAT, E_GEOM
from .geometry import decode_hex_block
from .store import Session, SessionSpec
from .util import atomic_write_json, read_json


def load_spec(path: str | Path, *, session_root: str | Path | None = None) -> SessionSpec:
    spec_path = Path(path)
    document = read_json(spec_path, field_name="session")
    if not isinstance(document, Mapping):
        raise DiagnosticError(E_FORMAT, "session manifest must be an object", field="session")
    root = Path(session_root) if session_root is not None else spec_path.parent
    return SessionSpec.from_mapping(root, document)


def initialize_from_spec(
    spec_path: str | Path,
    *,
    session_root: str | Path,
    reset: bool = False,
) -> tuple[Session, SessionSpec]:
    spec = load_spec(spec_path, session_root=session_root)
    session = Session.initialize(
        session_root,
        block_size=spec.block_size,
        block_count=spec.block_count,
        image_name=spec.image_name,
        manifest_format=spec.manifest_format,
        threshold=spec.threshold,
        snapshot_plan=spec.snapshot_plan,
        rollback_target=spec.rollback_target,
        writes=spec.writes,
        steps=spec.steps,
        reset=reset,
    )
    return session, spec


def _write_payload(step: Mapping[str, Any], block_size: int) -> tuple[int, bytes]:
    block_value = step.get("block", step.get("block_index"))
    if isinstance(block_value, bool) or not isinstance(block_value, int):
        raise DiagnosticError(
            E_GEOM,
            "write step block must be an integer",
            field="block_index",
            details={"actual": block_value},
        )
    if "hex" in step:
        return block_value, decode_hex_block(str(step["hex"]), block_size)
    if "fill_hex" in step:
        fill = str(step["fill_hex"]).removeprefix("0x")
        if len(fill) % 2 or not fill:
            raise DiagnosticError(
                E_GEOM,
                "fill_hex must contain complete bytes",
                field="hex_data",
                details={"actual": fill},
            )
        try:
            fill_bytes = bytes.fromhex(fill)
        except ValueError as exc:
            raise DiagnosticError(
                E_GEOM,
                "fill_hex contains invalid hexadecimal characters",
                field="hex_data",
                details={"reason": str(exc)},
            )
        if block_size % len(fill_bytes):
            raise DiagnosticError(
                E_GEOM,
                "fill_hex byte pattern must evenly divide block_size",
                field="hex_data",
                details={"block_size": block_size, "pattern_bytes": len(fill_bytes)},
            )
        return block_value, (fill_bytes * (block_size // len(fill_bytes)))[:block_size]
    raise DiagnosticError(
        E_GEOM,
        "write step requires a hex or fill_hex payload",
        field="hex_data",
        details={"block_index": block_value},
    )


def execute_steps(session: Session, steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    last_snapshot: str | None = None
    for position, step in enumerate(steps):
        action = step.get("action")
        if action == "snapshot":
            snapshot_id = step.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                raise DiagnosticError(
                    E_FORMAT,
                    "snapshot step requires snapshot_id",
                    field="snapshot_id",
                    details={"step": position},
                )
            if "parent_snapshot_id" in step:
                parent = step.get("parent_snapshot_id")
                if parent is not None and not isinstance(parent, str):
                    raise DiagnosticError(
                        E_FORMAT,
                        "parent_snapshot_id must be a string or null",
                        snapshot_id=snapshot_id,
                        field="parent_snapshot_id",
                    )
            else:
                parent = last_snapshot
            row = session.create_snapshot(snapshot_id, parent_snapshot_id=parent)
            last_snapshot = snapshot_id
            results.append(
                {
                    "step": position,
                    "action": "snapshot",
                    "snapshot_id": snapshot_id,
                    "parent_snapshot_id": parent,
                    "snapshot_version": row["snapshot_version"],
                }
            )
        elif action == "write":
            block_index, data = _write_payload(step, session.block_size)
            result = session.write_block(block_index, data)
            results.append({"step": position, "action": "write", **result})
        elif action == "audit":
            conclusion = session.audit()
            results.append(
                {
                    "step": position,
                    "action": "audit",
                    "audit_version": conclusion.get("audit_version"),
                    "status": conclusion.get("status"),
                }
            )
        else:
            raise DiagnosticError(
                E_FORMAT,
                "unknown session step action",
                field="action",
                details={"step": position, "actual": action},
            )
    return results


def run_declared_session(session: Session) -> list[dict[str, Any]]:
    spec_path = session.root / "session.json"
    document = read_json(spec_path, field_name="session")
    spec = SessionSpec.from_mapping(session.root, document)
    if spec.steps:
        return execute_steps(session, spec.steps)
    results: list[dict[str, Any]] = []
    parent: str | None = None
    for snapshot_id in spec.snapshot_plan:
        session.create_snapshot(snapshot_id, parent_snapshot_id=parent)
        results.append(
            {
                "action": "snapshot",
                "snapshot_id": snapshot_id,
                "parent_snapshot_id": parent,
            }
        )
        parent = snapshot_id
    for step in spec.writes:
        block_index, data = _write_payload(step, session.block_size)
        results.append({"action": "write", **session.write_block(block_index, data)})
    return results


def _fill(token: int) -> str:
    return f"{token & 0xFF:02x}"


def sample_specs() -> dict[str, dict[str, Any]]:
    small_steps: list[dict[str, Any]] = [
        {"action": "snapshot", "snapshot_id": "baseline", "parent_snapshot_id": None},
        {"action": "write", "block": 0, "fill_hex": _fill(0x11)},
        {"action": "write", "block": 3, "fill_hex": _fill(0x22)},
        {"action": "write", "block": 7, "fill_hex": _fill(0x33)},
        {"action": "snapshot", "snapshot_id": "checkpoint-a", "parent_snapshot_id": "baseline"},
        {"action": "write", "block": 1, "fill_hex": _fill(0x44)},
        {"action": "write", "block": 3, "fill_hex": _fill(0x55)},
        {"action": "write", "block": 9, "fill_hex": _fill(0x66)},
        {"action": "write", "block": 10, "fill_hex": _fill(0x77)},
        {"action": "write", "block": 15, "fill_hex": _fill(0x88)},
        {"action": "snapshot", "snapshot_id": "checkpoint-b", "parent_snapshot_id": "checkpoint-a"},
        {"action": "write", "block": 0, "fill_hex": _fill(0x99)},
        {"action": "write", "block": 8, "fill_hex": _fill(0xAA)},
    ]
    large_writes_one = [0, 1, 2, 4, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 700, 777, 900, 1023]
    large_writes_two = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39]
    large_steps: list[dict[str, Any]] = [
        {"action": "snapshot", "snapshot_id": "large-base", "parent_snapshot_id": None},
    ]
    large_steps.extend(
        {"action": "write", "block": block, "fill_hex": _fill(0x40 + (block % 120))}
        for block in large_writes_one
    )
    large_steps.extend(
        [
            {"action": "snapshot", "snapshot_id": "large-mid", "parent_snapshot_id": "large-base"},
        ]
    )
    large_steps.extend(
        {"action": "write", "block": block, "fill_hex": _fill(0x80 + (block % 100))}
        for block in large_writes_two
    )
    large_steps.append(
        {"action": "snapshot", "snapshot_id": "large-tail", "parent_snapshot_id": "large-mid"}
    )
    high_blocks_one = list(range(0, 128, 3))
    high_blocks_two = list(range(1, 128, 3)) + list(range(2, 64, 5))
    high_steps: list[dict[str, Any]] = [
        {"action": "snapshot", "snapshot_id": "churn-base", "parent_snapshot_id": None},
    ]
    high_steps.extend(
        {"action": "write", "block": block, "fill_hex": _fill(0xC0 + (block % 40))}
        for block in high_blocks_one
    )
    high_steps.append(
        {"action": "snapshot", "snapshot_id": "churn-mid", "parent_snapshot_id": "churn-base"}
    )
    high_steps.extend(
        {"action": "write", "block": block, "fill_hex": _fill(0x20 + (block % 80))}
        for block in high_blocks_two
    )
    high_steps.append(
        {"action": "snapshot", "snapshot_id": "churn-tail", "parent_snapshot_id": "churn-mid"}
    )
    return {
        "small": {
            "schema_version": 1,
            "description": "小卷：512B x 16 blocks，三快照与链式写时复制",
            "volume": {"image": "volume.bin", "block_size": 512, "block_count": 16},
            "manifest_format": "m3",
            "rollback_threshold": 0.5,
            "snapshot_plan": ["baseline", "checkpoint-a", "checkpoint-b"],
            "rollback_target": "checkpoint-a",
            "steps": small_steps,
        },
        "large": {
            "schema_version": 1,
            "description": "大卷：4KiB x 1024 blocks，低变更比例与长链",
            "volume": {"image": "volume.bin", "block_size": 4096, "block_count": 1024},
            "manifest_format": "m3",
            "rollback_threshold": 0.25,
            "snapshot_plan": ["large-base", "large-mid", "large-tail"],
            "rollback_target": "large-mid",
            "steps": large_steps,
        },
        "high_churn": {
            "schema_version": 1,
            "description": "高变更块比例：1KiB x 128 blocks，超过 70% 变更",
            "volume": {"image": "volume.bin", "block_size": 1024, "block_count": 128},
            "manifest_format": "m3",
            "rollback_threshold": 0.25,
            "snapshot_plan": ["churn-base", "churn-mid", "churn-tail"],
            "rollback_target": "churn-base",
            "steps": high_steps,
        },
    }


def prepare_samples(workspace: str | Path, *, reset: bool = True) -> dict[str, Any]:
    root = Path(workspace).resolve()
    specs_dir = root / "samples"
    generated_dir = root / "samples" / "generated"
    specs_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name, document in sample_specs().items():
        spec_path = specs_dir / f"{name}.session.json"
        atomic_write_json(spec_path, document)
        session_root = generated_dir / name
        session, spec = initialize_from_spec(
            spec_path,
            session_root=session_root,
            reset=reset,
        )
        steps = execute_steps(session, spec.steps)
        model = session.build_manifest_model()
        results[name] = {
            "session_root": str(session_root),
            "spec_path": str(spec_path),
            "description": document["description"],
            "block_size": session.block_size,
            "block_count": session.block_count,
            "snapshot_count": len(session.snapshot_rows()),
            "state_version": session.state_version,
            "manifest_digest": f"sha256:{model.manifest_digest}",
            "steps_executed": len(steps),
            "volume_size": session.geometry().image_size,
            "rollback_target": spec.rollback_target,
        }
    atomic_write_json(root / "samples" / "prepared-samples.json", {"samples": results})
    return results
