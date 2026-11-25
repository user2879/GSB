from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from .config import RenderConfig
from .engine import RenderEngine
from .errors import InputInvalid
from .inputs import load_conditions, load_png_condition, load_vocabulary, validate_prompt
from .lease import LeaseClient, LeaseServer, now_ms


VALID_OPS = {
    "set_prompt",
    "set_image",
    "delete_image",
    "set_motion",
    "tune_motion",
    "set_weights",
    "set_denoise_steps",
    "set_tile_height",
    "set_quant",
}


class CommandSession:
    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        config: RenderConfig,
        lease_ttl_ms: int | None = None,
        start_lease_server: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.workspace_path = Path(workspace).resolve()
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.engine = RenderEngine(self.workspace_path)
        self.config = config
        self.lease_server: LeaseServer | None = None
        self.lease_client: LeaseClient | None = None
        self.lease_info: dict[str, Any] | None = None
        if start_lease_server:
            self.lease_server = LeaseServer(
                host=config.executor_host, port=0, max_slots=config.max_slots
            ).start()
            self.lease_client = LeaseClient(config.executor_host, self.lease_server.port)
            ttl = int(lease_ttl_ms or config.lease_ttl_ms)
            record = self.lease_client.acquire(
                lease_key=f"workspace:{self.workspace_path.name}",
                holder="fixed-render-workbench",
                scopes=["stage:encode", "stage:fuse", "stage:denoise", "stage:decode", "stage:output"],
                ttl_ms=ttl,
            )
            self.lease_info = _lease_info(record)
            self.lease_info["port"] = self.lease_server.port
        self._prepare_writable_motion()

    def close(self) -> None:
        try:
            if self.lease_client and self.lease_info:
                self.lease_client.release(self.lease_info["lease_id"], self.lease_info["generation"])
        except Exception:
            pass
        if self.lease_server:
            self.lease_server.close()

    def __enter__(self) -> "CommandSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _prepare_writable_motion(self) -> None:
        source = self.config.resolve(self.config.conditions.motion_path)
        target = self.workspace_path / "inputs" / "motion_current.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source and source.exists() and source != target:
            target.write_bytes(source.read_bytes())
        self.config = self.config.with_changes({"motion_path": str(target)})

    def renew_lease(self, ttl_ms: int | None = None) -> dict[str, Any] | None:
        if not self.lease_client or not self.lease_info:
            return None
        record = self.lease_client.renew(
            self.lease_info["lease_id"],
            int(self.lease_info["generation"]),
            ttl_ms or self.config.lease_ttl_ms,
        )
        self.lease_info = _lease_info(record)
        if self.lease_server is not None:
            self.lease_info["port"] = self.lease_server.port
        return self.lease_info

    def validate_command_stream(self, commands: list[dict[str, Any]]) -> None:
        draft = self.config
        simulated_motion: tuple[int, int, int, int] | None = None
        errors: dict[str, str] = {}
        for index, command in enumerate(commands):
            field_prefix = f"commands[{index}]"
            op = command.get("op")
            if op not in VALID_OPS:
                errors[f"{field_prefix}.op"] = f"unsupported op {op!r}"
                continue
            try:
                if op == "set_prompt":
                    tokens = _parse_command_tokens(command)
                    draft = draft.with_changes({"prompt_tokens": tokens})
                    vocab = load_vocabulary(
                        draft.resolve(draft.conditions.vocab_path), draft.text_dim
                    )
                    validate_prompt(tokens, vocab)
                elif op == "set_image":
                    image_path = self._resolve_input_path(str(command.get("path", "")))
                    if not image_path.exists():
                        raise InputInvalid("image file does not exist", field_name="path")
                    load_png_condition(image_path, draft.image_width, draft.image_height)
                    draft = draft.with_changes({"image_path": str(image_path)})
                elif op == "delete_image":
                    draft = draft.with_changes({"image_path": None})
                elif op == "set_motion":
                    motion_path = self._resolve_input_path(str(command.get("path", "")))
                    candidate = draft.with_changes({"motion_path": str(motion_path)})
                    load_conditions(candidate)
                    draft = candidate
                elif op == "tune_motion":
                    br = _require_int(command, "block_row", 0, draft.frame_height // draft.motion_block_h - 1)
                    bc = _require_int(command, "block_col", 0, draft.frame_width // draft.motion_block_w - 1)
                    if "dx_q8" in command or "dy_q8" in command:
                        _optional_int(command, "dx_q8", -8192, 8192)
                        _optional_int(command, "dy_q8", -8192, 8192)
                    else:
                        _optional_int(command, "dx_delta_q8", -8192, 8192)
                        _optional_int(command, "dy_delta_q8", -8192, 8192)
                    simulated_motion = (br, bc, 0, 0)
                elif op == "set_weights":
                    weights = command.get("weights")
                    if not isinstance(weights, dict):
                        raise InputInvalid("weights must be a mapping", field_name="weights")
                    draft = draft.with_changes({"fusion_weights": weights})
                elif op == "set_denoise_steps":
                    value = _require_int(command, "value", 1, 64)
                    draft = draft.with_changes({"denoise_steps": value})
                elif op == "set_tile_height":
                    value = _require_int(command, "value", 1, draft.frame_height)
                    draft = draft.with_changes({"tile_height": value})
                elif op == "set_quant":
                    draft = draft.with_changes(
                        {"fixed_frac_bits": command.get("frac_bits"), "quant_mode": command.get("mode")}
                    )
            except InputInvalid as exc:
                for key, message in (exc.fields or {}).items():
                    errors[f"{field_prefix}.{key}"] = message
                if not exc.fields:
                    errors[f"{field_prefix}.{exc.field_name or 'command'}"] = exc.message
        if errors:
            raise InputInvalid("command stream rejected as a whole file", fields=errors)

    def apply_and_render(
        self,
        command: dict[str, Any],
        injection: dict[str, Any] | None = None,
        ledger_name: str = "latest",
    ) -> dict[str, Any]:
        old_manifest = self.engine.snapshots.current()
        old_version = int((old_manifest or {}).get("version", 0))
        started = time.perf_counter()
        self.apply(command)
        self.renew_lease()
        bundle = load_conditions(self.config)
        result = self.engine.render(
            self.config,
            bundle,
            lease_client=self.lease_client,
            lease_info=self.lease_info,
            injection=injection,
            ledger_name=ledger_name,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        preview = self.preview(result["manifest"], old_manifest, old_version, elapsed_ms)
        result["preview"] = preview
        return result

    def apply(self, command: dict[str, Any]) -> RenderConfig:
        op = command["op"]
        if op == "set_prompt":
            tokens = tuple(command["tokens"]) if isinstance(command["tokens"], list) else tuple(str(command["tokens"]).split())
            self.config = self.config.with_changes({"prompt_tokens": tokens})
        elif op == "set_image":
            path = self._resolve_input_path(str(command["path"]))
            self.config = self.config.with_changes({"image_path": str(path)})
        elif op == "delete_image":
            self.config = self.config.with_changes({"image_path": None})
        elif op == "set_motion":
            path = self._resolve_input_path(str(command["path"]))
            self.config = self.config.with_changes({"motion_path": str(path)})
        elif op == "tune_motion":
            self._tune_motion(command)
        elif op == "set_weights":
            self.config = self.config.with_changes({"fusion_weights": command["weights"]})
        elif op == "set_denoise_steps":
            self.config = self.config.with_changes({"denoise_steps": int(command["value"])})
        elif op == "set_tile_height":
            self.config = self.config.with_changes({"tile_height": int(command["value"])})
        elif op == "set_quant":
            self.config = self.config.with_changes(
                {"fixed_frac_bits": int(command["frac_bits"]), "quant_mode": str(command["mode"])}
            )
        else:
            raise InputInvalid(f"unsupported command op {op}", field_name="op")
        return self.config

    def render_current(
        self, injection: dict[str, Any] | None = None, ledger_name: str = "latest"
    ) -> dict[str, Any]:
        self.renew_lease()
        bundle = load_conditions(self.config)
        return self.engine.render(
            self.config,
            bundle,
            lease_client=self.lease_client,
            lease_info=self.lease_info,
            injection=injection,
            ledger_name=ledger_name,
        )

    def preview(
        self,
        manifest: dict[str, Any],
        old_manifest: dict[str, Any] | None,
        old_version: int,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        budget = int(self.config.preview_budget_ms)
        base = manifest
        stale = elapsed_ms > budget
        code = "E_OK"
        based_on_version = int(manifest["version"])
        if stale and old_manifest is not None:
            base = old_manifest
            code = "E_PREVIEW_STALE"
            based_on_version = old_version
        elif stale:
            code = "E_PREVIEW_STALE"
            based_on_version = 0
        lease = self.lease_info or manifest.get("lease", {})
        return {
            "code": code,
            "fresh": not stale or old_manifest is None,
            "based_on_version": based_on_version,
            "latest_version": int(manifest["version"]),
            "budget_ms": budget,
            "elapsed_ms": elapsed_ms,
            "completed_frames": base.get("completed_frames", 0),
            "affected_frame_range": base.get("affected_frame_range", []),
            "tile_completion": base.get("tile_completion", {}),
            "lease_expires_ms": lease.get("expires_ms"),
            "lease_generation": lease.get("generation"),
            "progress": {
                "completed": base.get("cursor", {}).get("completed_count", 0),
                "total": base.get("cursor", {}).get("total_nodes", 0),
            },
            "frame_fingerprints": base.get("frame_fingerprints", {}),
            "generated_at_ms": now_ms(),
        }

    def _resolve_input_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        root_candidate = (self.root / path).resolve()
        if root_candidate.exists():
            return root_candidate
        config_candidate = self.config.base_dir / path
        return config_candidate.resolve()

    def _tune_motion(self, command: dict[str, Any]) -> None:
        motion_path = Path(self.config.conditions.motion_path)
        rows: list[dict[str, str]] = []
        with motion_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or ["block_row", "block_col", "dx_q8", "dy_q8"]
            rows = list(reader)
        br = int(command["block_row"])
        bc = int(command["block_col"])
        for row in rows:
            if int(row["block_row"]) == br and int(row["block_col"]) == bc:
                if "dx_q8" in command:
                    row["dx_q8"] = str(int(command["dx_q8"]))
                else:
                    row["dx_q8"] = str(int(row["dx_q8"]) + int(command.get("dx_delta_q8", 0)))
                if "dy_q8" in command:
                    row["dy_q8"] = str(int(command["dy_q8"]))
                else:
                    row["dy_q8"] = str(int(row["dy_q8"]) + int(command.get("dy_delta_q8", 0)))
                break
        else:
            raise InputInvalid("motion block not found", field_name="block_row")
        with motion_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)


def _lease_info(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "lease_id": record["lease_id"],
        "generation": int(record["generation"]),
        "issued_ms": int(record["issued_ms"]),
        "expires_ms": int(record["expires_ms"]),
        "port": None,
    }


def _parse_command_tokens(command: dict[str, Any]) -> tuple[str, ...]:
    tokens = command.get("tokens")
    if isinstance(tokens, str):
        return tuple(tokens.split())
    if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
        return tuple(tokens)
    raise InputInvalid("set_prompt requires tokens list/string", field_name="commands.op.tokens")


def _require_int(command: dict[str, Any], name: str, minimum: int, maximum: int) -> int:
    value = command.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InputInvalid(
            f"{name} must be integer in [{minimum},{maximum}]", field_name=name
        )
    return int(value)


def _optional_int(command: dict[str, Any], name: str, minimum: int, maximum: int) -> None:
    if name in command:
        _require_int(command, name, minimum, maximum)
