from __future__ import annotations

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .codec import (
    encode_image,
    encode_motion,
    encode_text,
    fuse_conditions,
    validate_condition_vector,
)
from .config import Config
from .errors import (
    DiagnosticError,
    E_BLOCK_ROW_OUT_OF_RANGE,
    E_BUDGET_PENDING,
    E_CONFIG_INVALID,
    E_NONFINITE_LATENT,
    E_PARTIAL_FAILURE,
    E_STALE_SNAPSHOT_VERSION,
    E_STALE_SUMMARY,
    fail,
)
from .pipeline import (
    compose_frame,
    decode_block_row,
    denoise_step,
    initialize_latent,
    latent_element_count,
    latent_shape,
    projection_parameters,
    validate_latent_values,
)
from .resources import (
    EditCommand,
    ImageCondition,
    Injection,
    MotionTable,
    PromptSet,
    TextPrompt,
    load_image,
    load_injections,
    load_motion_table,
    load_prompt_set,
)
from .store import ArtifactStore, SnapshotStore
from .util import (
    atomic_write_bytes,
    cache_key,
    decimal_text,
    decimal_to_fixed,
    fingerprint,
    is_int,
    to_decimal,
    to_int,
)


_DEFAULT = object()


@dataclass
class RuntimeState:
    prompt_name: str
    image: ImageCondition | None
    motion: MotionTable
    weights: dict[str, Decimal]
    denoise_steps: int
    seed: int
    guidance: Decimal
    quantization_level: int
    block_rows: int
    batch_point_budget: int
    max_concurrency: int


@dataclass
class TaskDescriptor:
    stage: str
    frame: int
    step: int | None = None
    row: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        stage_order = {
            "denoise": 60,
            "decode_row": 70,
        }[self.stage]
        return (
            stage_order,
            -1 if self.frame is None else int(self.frame),
            -1 if self.step is None else int(self.step),
            -1 if self.row is None else int(self.row),
        )

    @property
    def stable_key(self) -> str:
        pieces = [self.stage, f"f{self.frame:04d}"]
        if self.step is not None:
            pieces.append(f"s{self.step:04d}")
        if self.row is not None:
            pieces.append(f"r{self.row:04d}")
        return ":".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage": self.stage,
            "frame": self.frame,
            "stable_key": self.stable_key,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.row is not None:
            result["row"] = self.row
        return result


class RenderEngine:
    """Deterministic fixed-point renderer with exact incremental reuse."""

    def __init__(
        self,
        config: Config,
        workspace_root: str | os.PathLike[str],
        *,
        prompt_name: str = "p0",
        image_path: str | os.PathLike[str] | None | object = _DEFAULT,
        motion_path: str | os.PathLike[str] | None = None,
        store_root: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        injections: Sequence[Injection] | None = None,
        injections_path: str | os.PathLike[str] | None = None,
        restore: bool = True,
    ) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.prompt_set: PromptSet = load_prompt_set(config.resources.prompts, config)
        if prompt_name not in self.prompt_set.prompts:
            fail(
                E_CONFIG_INVALID,
                "prompt.initial",
                "initial prompt does not exist",
                detail={"prompt": prompt_name, "available": sorted(self.prompt_set.prompts)},
            )
        chosen_image_path: str | None
        if image_path is _DEFAULT:
            chosen_image_path = config.resources.image
        else:
            chosen_image_path = None if image_path is None else str(image_path)
        image = (
            None
            if chosen_image_path is None
            else load_image(self._resolve(chosen_image_path), config)
        )
        selected_motion_path = config.resources.motion if motion_path is None else str(motion_path)
        motion = load_motion_table(self._resolve(selected_motion_path), config)

        weights = config.weights_dict()
        self.state = RuntimeState(
            prompt_name=prompt_name,
            image=image,
            motion=motion,
            weights={name: weights[name] for name in ("text", "image", "motion")},
            denoise_steps=config.denoise_steps,
            seed=config.seed,
            guidance=config.guidance,
            quantization_level=config.quantization_level,
            block_rows=config.block_rows,
            batch_point_budget=config.batch_point_budget,
            max_concurrency=config.max_concurrency,
        )
        self.store_root = (
            Path(store_root).resolve()
            if store_root is not None
            else (self.workspace_root / "store").resolve()
        )
        self.artifact_store = ArtifactStore(self.store_root / "cache")
        self.snapshot_store = SnapshotStore(
            self.store_root / "snapshot-store",
            artifact_version=config.snapshot_version,
        )
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else (self.workspace_root / "output").resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        injection_list = list(injections or ())
        if injections_path is not None:
            injection_list.extend(load_injections(injections_path))
        self.injections = tuple(injection_list)
        self._injection_hits: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []

        self.cursor = 0
        self.latest_complete: dict[str, Any] | None = None
        self.pending_denoise: list[dict[str, Any]] = []
        self.consumed_points_total = 0
        self._changed_fields: list[str] = []
        self._affected_frames: list[int] = []
        self._ledger: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._text_product: dict[str, Any] | None = None
        self._image_product: dict[str, Any] | None = None
        self._motion_products: dict[int, dict[str, Any]] = {}
        self._fusion_products: dict[int, dict[str, Any]] = {}
        self._init_products: dict[int, dict[str, Any]] = {}
        self._denoise_products: dict[tuple[int, int], dict[str, Any]] = {}
        self._row_products: dict[tuple[int, int], dict[str, Any]] = {}
        self._frame_products: dict[int, dict[str, Any]] = {}
        self._blocked_frames: set[int] = set()
        self._projection_cache: dict[int, dict[str, Any]] = {}
        self._last_snapshot_version = 0
        self._last_snapshot_new = False
        self._current_denoise_scope = self._denoise_scope_fingerprint()
        self._command_history: list[dict[str, Any]] = []
        self._batch_computed_tasks: list[dict[str, Any]] = []
        self._batch_cache_hits = 0
        self._batch_consumed_points = 0
        self._time_budget_exhausted = False

        self._restored = False
        if restore:
            self._restore_snapshot()

    # ------------------------------------------------------------------
    # Public state and command API
    # ------------------------------------------------------------------

    def _resolve(self, path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        return str(candidate.resolve())

    def _resolve_command_path(self, command: EditCommand, path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate.resolve())
        if command.source and command.source != "<memory>":
            return str((Path(command.source).parent / candidate).resolve())
        return self._resolve(candidate)

    @property
    def prompt(self) -> TextPrompt:
        return self.prompt_set.prompts[self.state.prompt_name]

    def _runtime_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.state.prompt_name,
            "image_path": None if self.state.image is None else self.state.image.path,
            "motion_path": self.state.motion.path,
            "weights": {name: decimal_text(self.state.weights[name]) for name in sorted(self.state.weights)},
            "denoise_steps": self.state.denoise_steps,
            "seed": self.state.seed,
            "guidance": decimal_text(self.state.guidance),
            "quantization_level": self.state.quantization_level,
            "block_rows": self.state.block_rows,
            "batch_point_budget": self.state.batch_point_budget,
            "max_concurrency": self.state.max_concurrency,
        }

    def _mutable_fingerprint_material(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.fingerprint,
            "image": None if self.state.image is None else self.state.image.content_fingerprint,
            "motion": self.state.motion.content_fingerprint,
            "weights": {name: decimal_text(self.state.weights[name]) for name in sorted(self.state.weights)},
            "denoise_steps": self.state.denoise_steps,
            "seed": self.state.seed,
            "guidance": decimal_text(self.state.guidance),
            "quantization_level": self.state.quantization_level,
            "block_rows": self.state.block_rows,
        }

    def state_fingerprint(self) -> str:
        return fingerprint(
            {
                "config": self.config.config_fingerprint,
                "mutable": self._mutable_fingerprint_material(),
            },
            "render-state-v1",
        )

    def _denoise_scope_fingerprint(self) -> str:
        return fingerprint(
            {
                "prompt": self.prompt.fingerprint,
                "image": None if self.state.image is None else self.state.image.content_fingerprint,
                "motion": self.state.motion.content_fingerprint,
                "weights": {name: decimal_text(self.state.weights[name]) for name in sorted(self.state.weights)},
                "denoise_steps": self.state.denoise_steps,
                "seed": self.state.seed,
                "guidance": decimal_text(self.state.guidance),
                "quantization": self.state.quantization_level,
            },
            "denoise-scope-v1",
        )

    def projection(self) -> dict[str, Any]:
        if self.state.seed not in self._projection_cache:
            self._projection_cache[self.state.seed] = projection_parameters(
                self.config, seed=self.state.seed
            )
        return self._projection_cache[self.state.seed]

    def submit(self, command: EditCommand | dict[str, Any]) -> dict[str, Any]:
        if isinstance(command, dict):
            op = command.get("op")
            if not isinstance(op, str) or not op:
                fail(E_CONFIG_INVALID, f"commands[{len(self._command_history)}].op", "command op is missing")
            payload = {key: value for key, value in command.items() if key != "op"}
            command = EditCommand(
                index=len(self._command_history),
                op=op,
                payload=payload,
                source="<memory>",
                raw=command,
            )
        old_material = self._mutable_fingerprint_material()
        old_denoise_scope = self._current_denoise_scope
        op = command.op
        entry = f"commands[{command.index}].{op}"
        if op == "set_prompt":
            prompt_name = command.payload.get("prompt")
            if not isinstance(prompt_name, str) or prompt_name not in self.prompt_set.prompts:
                fail(E_CONFIG_INVALID, f"{entry}.prompt", "prompt does not exist", detail={"prompt": prompt_name})
            self.state.prompt_name = prompt_name
        elif op == "set_image":
            path = command.payload.get("path")
            if not isinstance(path, str) or not path:
                fail(E_CONFIG_INVALID, f"{entry}.path", "image path must be a non-empty string")
            self.state.image = load_image(self._resolve_command_path(command, path), self.config)
        elif op == "delete_image":
            self.state.image = None
        elif op == "set_motion":
            path = command.payload.get("path")
            if not isinstance(path, str) or not path:
                fail(E_CONFIG_INVALID, f"{entry}.path", "motion path must be a non-empty string")
            self.state.motion = load_motion_table(self._resolve_command_path(command, path), self.config)
        elif op == "set_weights":
            raw_weights = command.payload.get("weights")
            if not isinstance(raw_weights, dict):
                fail(E_CONFIG_INVALID, f"{entry}.weights", "weights must be an object")
            for name, value in raw_weights.items():
                if name not in {"text", "image", "motion"}:
                    fail(E_CONFIG_INVALID, f"{entry}.weights.{name}", "unknown fusion weight")
                self.state.weights[name] = to_decimal(
                    value, f"{entry}.weights.{name}", nonnegative=True
                )
        elif op == "set_denoise_steps":
            self.state.denoise_steps = to_int(
                command.payload.get("steps"), f"{entry}.steps", minimum=1
            )
        elif op == "set_block_rows":
            rows = to_int(command.payload.get("rows"), f"{entry}.rows", minimum=1)
            if self.config.height % rows != 0:
                fail(
                    E_CONFIG_INVALID,
                    f"{entry}.rows",
                    "block row count must divide frame height",
                    detail={"rows": rows, "height": self.config.height},
                )
            self.state.block_rows = rows
        elif op == "set_quant_level":
            level = to_int(command.payload.get("level"), f"{entry}.level", minimum=1)
            if level not in self.config.allowed_quantization_levels:
                fail(
                    E_CONFIG_INVALID,
                    f"{entry}.level",
                    "quantization level is not in the allowed enumeration",
                    detail={"level": level, "allowed": list(self.config.allowed_quantization_levels)},
                )
            self.state.quantization_level = level
        elif op == "set_guidance":
            self.state.guidance = to_decimal(
                command.payload.get("guidance"), f"{entry}.guidance", nonnegative=True
            )
        elif op == "set_seed":
            self.state.seed = to_int(command.payload.get("seed"), f"{entry}.seed", minimum=0)
        elif op == "set_batch_budget":
            self.state.batch_point_budget = to_int(
                command.payload.get("points"), f"{entry}.points", minimum=1
            )
        else:
            fail(E_CONFIG_INVALID, f"{entry}", "unknown command op", detail={"op": op})

        new_material = self._mutable_fingerprint_material()
        changed: list[str] = []
        for field_name in sorted(set(old_material) | set(new_material)):
            if old_material.get(field_name) != new_material.get(field_name):
                changed.append(field_name)
        new_denoise_scope = self._denoise_scope_fingerprint()
        if old_denoise_scope != new_denoise_scope:
            self.pending_denoise = []
        self._current_denoise_scope = new_denoise_scope
        self._changed_fields = changed
        self._affected_frames = list(range(self.config.frame_count)) if changed else []
        self.cursor += 1
        self._ledger = {}
        self.failures = []
        self._text_product = None
        self._image_product = None
        self._motion_products = {}
        self._fusion_products = {}
        self._init_products = {}
        self._denoise_products = {}
        self._row_products = {}
        self._frame_products = {}
        self._blocked_frames = set()
        self._command_history.append(
            {
                "cursor": self.cursor,
                "index": command.index,
                "op": command.op,
                "payload": command.payload,
                "changed_fields": changed,
                "render_fingerprint": self.state_fingerprint(),
            }
        )
        return {
            "cursor": self.cursor,
            "op": command.op,
            "changed_fields": changed,
            "affected_frames": self._affected_frames,
            "render_fingerprint": self.state_fingerprint(),
        }

    # ------------------------------------------------------------------
    # Injection and diagnostics helpers
    # ------------------------------------------------------------------

    def _find_injection(
        self,
        stage: str,
        kind: str,
        target: dict[str, Any] | None = None,
    ) -> Injection | None:
        for injection in self.injections:
            if injection.matches(stage, kind, target):
                return injection
        return None

    def _record_injection_hit(
        self,
        injection: Injection,
        *,
        stage: str,
        target: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        hit = self._injection_hits.setdefault(
            injection.injection_id,
            {
                "id": injection.injection_id,
                "stage": injection.stage,
                "kind": injection.kind,
                "count": 0,
                "tasks": [],
                "code": code,
            },
        )
        hit["count"] += 1
        task = {"stage": stage}
        if target:
            task.update(target)
        if task not in hit["tasks"]:
            hit["tasks"].append(task)
        if code is not None:
            hit["code"] = code

    def _record_failure(
        self,
        error: DiagnosticError,
        *,
        stage: str,
        frame: int | None = None,
        step: int | None = None,
        row: int | None = None,
    ) -> None:
        item = error.to_dict()
        item.update({"stage": stage, "frame": frame, "step": step, "row": row})
        key = (item.get("code"), item.get("entry"), item.get("injection_id"), stage, frame, step, row)
        existing = {
            (
                entry.get("code"),
                entry.get("entry"),
                entry.get("injection_id"),
                entry.get("stage"),
                entry.get("frame"),
                entry.get("step"),
                entry.get("row"),
            )
            for entry in self.failures
        }
        if key not in existing:
            self.failures.append(item)

    def _record_ledger(
        self,
        *,
        stage: str,
        frame: int | None,
        step: int | None,
        row: int | None,
        status: str,
        key: str,
        upstream: str,
        parameters: str,
        quantization_level: int,
        product_fingerprint: str,
        invalidated_by: Sequence[str] = (),
    ) -> None:
        logical = (stage, -1 if frame is None else frame, -1 if step is None else step, -1 if row is None else row)
        if logical in self._ledger:
            return
        self._ledger[logical] = {
            "stage": stage,
            "frame": frame,
            "step": step,
            "row": row,
            "status": status,
            "cache_key": key,
            "upstream_fingerprint": upstream,
            "parameter_fingerprint": parameters,
            "quantization_level": quantization_level,
            "snapshot_version": self.config.snapshot_version,
            "product_fingerprint": product_fingerprint,
            "invalidated_by": list(invalidated_by),
        }

    def _sorted_ledger(self) -> list[dict[str, Any]]:
        stage_order = {
            "encode_text": 10,
            "encode_image": 20,
            "encode_motion": 30,
            "fusion": 40,
            "latent_init": 50,
            "denoise": 60,
            "decode_row": 70,
            "decode_frame": 80,
        }
        return sorted(
            self._ledger.values(),
            key=lambda item: (
                stage_order[str(item["stage"])],
                -1 if item["frame"] is None else int(item["frame"]),
                -1 if item["step"] is None else int(item["step"]),
                -1 if item["row"] is None else int(item["row"]),
            ),
        )

    def _sorted_failures(self) -> list[dict[str, Any]]:
        return sorted(
            self.failures,
            key=lambda item: (
                str(item.get("stage")),
                -1 if item.get("frame") is None else int(item["frame"]),
                -1 if item.get("step") is None else int(item["step"]),
                -1 if item.get("row") is None else int(item["row"]),
                str(item.get("code")),
                str(item.get("entry")),
            ),
        )

    def injection_report(self) -> list[dict[str, Any]]:
        return [self._injection_hits[key] for key in sorted(self._injection_hits)]

    # ------------------------------------------------------------------
    # Content-addressed artifact access
    # ------------------------------------------------------------------

    def _artifact_lookup(
        self,
        *,
        stage: str,
        upstream: str,
        parameters: str,
        quantization_level: int,
        frame: int | None,
        step: int | None,
        row: int | None,
        invalidated_by: Sequence[str],
    ) -> tuple[dict[str, Any] | None, str]:
        key = cache_key(
            stage,
            upstream,
            parameters,
            quantization_level,
            self.config.snapshot_version,
        )
        artifact = self.artifact_store.get(key)
        if artifact is None:
            return None, key
        payload = artifact["payload"]
        self._batch_cache_hits += 1
        self._record_ledger(
            stage=stage,
            frame=frame,
            step=step,
            row=row,
            status="reused",
            key=key,
            upstream=upstream,
            parameters=parameters,
            quantization_level=quantization_level,
            product_fingerprint=str(payload.get("fingerprint") or payload.get("content_fingerprint") or ""),
            invalidated_by=[],
        )
        return payload, key

    def _artifact_store_new(
        self,
        *,
        stage: str,
        key: str,
        upstream: str,
        parameters: str,
        quantization_level: int,
        frame: int | None,
        step: int | None,
        row: int | None,
        payload: dict[str, Any],
        invalidated_by: Sequence[str],
    ) -> dict[str, Any]:
        artifact, _ = self.artifact_store.put(
            stage=stage,
            key=key,
            upstream_fingerprint=upstream,
            parameter_fingerprint=parameters,
            quantization_level=quantization_level,
            snapshot_version=self.config.snapshot_version,
            payload=payload,
        )
        stored_payload = artifact["payload"]
        self._record_ledger(
            stage=stage,
            frame=frame,
            step=step,
            row=row,
            status="recomputed",
            key=key,
            upstream=upstream,
            parameters=parameters,
            quantization_level=quantization_level,
            product_fingerprint=str(
                stored_payload.get("fingerprint") or stored_payload.get("content_fingerprint") or ""
            ),
            invalidated_by=invalidated_by,
        )
        return stored_payload

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def advance(self) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self.config.interaction_latency_budget_ms / 1000.0
        self._batch_computed_tasks = []
        self._batch_cache_hits = 0
        self._batch_consumed_points = 0
        self._time_budget_exhausted = False
        self._prior_pending_denoise = list(self.pending_denoise)
        self.pending_denoise = []

        quant_injection = self._find_injection("quantization", "quant_level_oob")
        if quant_injection is not None:
            bad_level = quant_injection.value if is_int(quant_injection.value) else 9
            self._record_injection_hit(
                quant_injection,
                stage="quantization",
                code=E_CONFIG_INVALID,
            )
            try:
                if bad_level not in self.config.allowed_quantization_levels:
                    fail(
                        E_CONFIG_INVALID,
                        "condition.quantization_level",
                        "quantization level is outside the allowed enumeration",
                        detail={"value": bad_level, "allowed": list(self.config.allowed_quantization_levels)},
                        injection_id=quant_injection.injection_id,
                    )
            except DiagnosticError as error:
                self._record_failure(error, stage="quantization")
                self._blocked_frames = set(range(self.config.frame_count))

        self._ensure_condition_encodings()
        self._ensure_fusions_and_initial_latents()
        self._run_denoise_batch(deadline)
        self._decode_available_frames()
        response = self._finalize_response(started)
        self._commit_snapshot()
        # A commit can add version metadata to the response without changing content keys.
        response["snapshot"]["commit_version"] = self._last_snapshot_version
        response["snapshot"]["new_version_created"] = self._last_snapshot_new
        return response

    def _ensure_condition_encodings(self) -> None:
        if self._text_product is None:
            upstream = fingerprint(
                {
                    "prompt": self.prompt.fingerprint,
                    "vocabulary": self.config.vocabulary_fingerprint,
                },
                "text-encode-upstream-v1",
            )
            parameters = fingerprint(
                {
                    "dimension": self.config.condition_dim,
                    "frac_bits": self.config.frac_bits,
                },
                "text-encode-params-v1",
            )
            payload, key = self._artifact_lookup(
                stage="encode_text",
                upstream=upstream,
                parameters=parameters,
                quantization_level=0,
                frame=None,
                step=None,
                row=None,
                invalidated_by=self._changed_fields,
            )
            if payload is None:
                try:
                    computed = encode_text(self.prompt, self.config)
                    injection = self._find_injection("text_encode", "dimension_mismatch")
                    if injection is not None:
                        faulty = list(computed["vector"]) + [0]
                        self._record_injection_hit(
                            injection,
                            stage="text_encode",
                            code=E_CONFIG_INVALID,
                        )
                        validate_condition_vector(
                            faulty,
                            self.config.condition_dim,
                            "condition.text.vector",
                            injection_id=injection.injection_id,
                        )
                    validate_condition_vector(
                        computed["vector"],
                        self.config.condition_dim,
                        "condition.text.vector",
                    )
                    payload = self._artifact_store_new(
                        stage="encode_text",
                        key=key,
                        upstream=upstream,
                        parameters=parameters,
                        quantization_level=0,
                        frame=None,
                        step=None,
                        row=None,
                        payload=computed,
                        invalidated_by=self._changed_fields,
                    )
                except DiagnosticError as error:
                    self._record_failure(error, stage="encode_text")
                    self._blocked_frames = set(range(self.config.frame_count))
                    return
            self._text_product = payload
        if self._text_product is None:
            return

        if self.state.image is not None and self._image_product is None:
            image = self.state.image
            upstream = image.content_fingerprint
            parameters = fingerprint(
                {
                    "block_size": self.config.image_block_size,
                    "dimension": self.config.condition_dim,
                    "frac_bits": self.config.frac_bits,
                },
                "image-encode-params-v1",
            )
            payload, key = self._artifact_lookup(
                stage="encode_image",
                upstream=upstream,
                parameters=parameters,
                quantization_level=0,
                frame=None,
                step=None,
                row=None,
                invalidated_by=self._changed_fields,
            )
            if payload is None:
                try:
                    computed = encode_image(image, self.config)
                    injection = self._find_injection("image_encode", "dimension_mismatch")
                    if injection is not None:
                        faulty = list(computed["vector"]) + [0]
                        self._record_injection_hit(
                            injection,
                            stage="image_encode",
                            code=E_CONFIG_INVALID,
                        )
                        validate_condition_vector(
                            faulty,
                            self.config.condition_dim,
                            "condition.image.vector",
                            injection_id=injection.injection_id,
                        )
                    validate_condition_vector(
                        computed["vector"],
                        self.config.condition_dim,
                        "condition.image.vector",
                    )
                    payload = self._artifact_store_new(
                        stage="encode_image",
                        key=key,
                        upstream=upstream,
                        parameters=parameters,
                        quantization_level=0,
                        frame=None,
                        step=None,
                        row=None,
                        payload=computed,
                        invalidated_by=self._changed_fields,
                    )
                except DiagnosticError as error:
                    self._record_failure(error, stage="encode_image")
                    self._blocked_frames = set(range(self.config.frame_count))
                    return
            self._image_product = payload

        for frame in range(self.config.frame_count):
            if frame in self._motion_products:
                continue
            upstream = fingerprint(
                {"motion": self.state.motion.content_fingerprint, "frame": frame},
                "motion-encode-upstream-v1",
            )
            parameters = fingerprint(
                {
                    "dimension": self.config.condition_dim,
                    "frac_bits": self.config.frac_bits,
                    "block_width": self.config.motion_block_width,
                    "block_height": self.config.motion_block_height,
                },
                "motion-encode-params-v1",
            )
            payload, key = self._artifact_lookup(
                stage="encode_motion",
                upstream=upstream,
                parameters=parameters,
                quantization_level=0,
                frame=frame,
                step=None,
                row=None,
                invalidated_by=self._changed_fields,
            )
            if payload is None:
                try:
                    computed = encode_motion(self.state.motion, frame, self.config)
                    injection = self._find_injection(
                        "motion_encode", "dimension_mismatch", {"frame": frame}
                    )
                    if injection is not None:
                        faulty = list(computed["vector"]) + [0]
                        self._record_injection_hit(
                            injection,
                            stage="motion_encode",
                            target={"frame": frame},
                            code=E_CONFIG_INVALID,
                        )
                        validate_condition_vector(
                            faulty,
                            self.config.condition_dim,
                            "condition.motion.vector",
                            injection_id=injection.injection_id,
                        )
                    validate_condition_vector(
                        computed["vector"],
                        self.config.condition_dim,
                        "condition.motion.vector",
                    )
                    payload = self._artifact_store_new(
                        stage="encode_motion",
                        key=key,
                        upstream=upstream,
                        parameters=parameters,
                        quantization_level=0,
                        frame=frame,
                        step=None,
                        row=None,
                        payload=computed,
                        invalidated_by=self._changed_fields,
                    )
                except DiagnosticError as error:
                    self._record_failure(error, stage="encode_motion", frame=frame)
                    self._blocked_frames.add(frame)
                    continue
            self._motion_products[frame] = payload

    def _ensure_fusions_and_initial_latents(self) -> None:
        if self._text_product is None:
            return
        for frame in range(self.config.frame_count):
            if frame in self._blocked_frames:
                continue
            motion = self._motion_products.get(frame)
            if motion is None:
                self._blocked_frames.add(frame)
                continue
            if frame not in self._fusion_products:
                upstream = fingerprint(
                    {
                        "text": self._text_product["fingerprint"],
                        "image": None if self._image_product is None else self._image_product["fingerprint"],
                        "motion": motion["fingerprint"],
                    },
                    "fusion-upstream-v1",
                )
                parameters = fingerprint(
                    {
                        "frame": frame,
                        "weights": {
                            name: decimal_text(self.state.weights[name])
                            for name in ("text", "image", "motion")
                        },
                        "dimension": self.config.condition_dim,
                        "frac_bits": self.config.frac_bits,
                    },
                    "fusion-params-v1",
                )
                payload, key = self._artifact_lookup(
                    stage="fusion",
                    upstream=upstream,
                    parameters=parameters,
                    quantization_level=self.state.quantization_level,
                    frame=frame,
                    step=None,
                    row=None,
                    invalidated_by=self._changed_fields,
                )
                if payload is None:
                    try:
                        computed = fuse_conditions(
                            frame,
                            self._text_product,
                            self._image_product,
                            motion,
                            self.state.weights,
                            self.config,
                            quantization_level=self.state.quantization_level,
                        )
                        injection = self._find_injection("fusion", "dimension_mismatch", {"frame": frame})
                        if injection is not None:
                            faulty = list(computed["fusion_vector"]) + [0]
                            self._record_injection_hit(
                                injection,
                                stage="fusion",
                                target={"frame": frame},
                                code=E_CONFIG_INVALID,
                            )
                            validate_condition_vector(
                                faulty,
                                self.config.condition_dim,
                                "condition.fusion.vector",
                                injection_id=injection.injection_id,
                            )
                        validate_condition_vector(
                            computed["fusion_vector"],
                            self.config.condition_dim,
                            "condition.fusion.vector",
                        )
                        payload = self._artifact_store_new(
                            stage="fusion",
                            key=key,
                            upstream=upstream,
                            parameters=parameters,
                            quantization_level=self.state.quantization_level,
                            frame=frame,
                            step=None,
                            row=None,
                            payload=computed,
                            invalidated_by=self._changed_fields,
                        )
                    except DiagnosticError as error:
                        self._record_failure(error, stage="fusion", frame=frame)
                        self._blocked_frames.add(frame)
                        continue
                self._fusion_products[frame] = payload

            if frame not in self._init_products:
                fusion = self._fusion_products[frame]
                step_config_fp = fingerprint(
                    {
                        "fusion": fusion["fingerprint"],
                        "seed": self.state.seed,
                        "frame": frame,
                        "shape": list(latent_shape(self.config)),
                        "quantization": self.state.quantization_level,
                        "frac_bits": self.config.frac_bits,
                    },
                    "latent-init-step-config-v1",
                )
                upstream = fusion["fingerprint"]
                parameters = fingerprint(
                    {
                        "frame": frame,
                        "seed": self.state.seed,
                        "shape": list(latent_shape(self.config)),
                        "frac_bits": self.config.frac_bits,
                    },
                    "latent-init-params-v1",
                )
                payload, key = self._artifact_lookup(
                    stage="latent_init",
                    upstream=upstream,
                    parameters=parameters,
                    quantization_level=self.state.quantization_level,
                    frame=frame,
                    step=None,
                    row=None,
                    invalidated_by=self._changed_fields,
                )
                if payload is None:
                    try:
                        computed = initialize_latent(
                            fusion,
                            frame,
                            self.config,
                            step_fingerprint=step_config_fp,
                            init_config_fingerprint=fingerprint(
                                {
                                    "latent_shape": list(latent_shape(self.config)),
                                    "frac_bits": self.config.frac_bits,
                                },
                                "latent-init-config-v1",
                            ),
                            seed=self.state.seed,
                            quantization_level=self.state.quantization_level,
                        )
                        injection = self._find_injection("init", "nonfinite_latent", {"frame": frame})
                        if injection is not None:
                            faulty = list(computed["values"])
                            faulty[len(faulty) // 2] = float("nan")
                            self._record_injection_hit(
                                injection,
                                stage="init",
                                target={"frame": frame},
                                code=E_NONFINITE_LATENT,
                            )
                            validate_latent_values(
                                faulty,
                                latent_element_count(self.config),
                                "latent.init",
                                injection_id=injection.injection_id,
                            )
                        validate_latent_values(
                            computed["values"],
                            latent_element_count(self.config),
                            "latent.init",
                        )
                        payload = self._artifact_store_new(
                            stage="latent_init",
                            key=key,
                            upstream=upstream,
                            parameters=parameters,
                            quantization_level=self.state.quantization_level,
                            frame=frame,
                            step=None,
                            row=None,
                            payload=computed,
                            invalidated_by=self._changed_fields,
                        )
                    except DiagnosticError as error:
                        self._record_failure(error, stage="init", frame=frame)
                        self._blocked_frames.add(frame)
                        continue
                self._init_products[frame] = payload

    def _run_task_batch(
        self,
        tasks: Sequence[TaskDescriptor],
        worker: Callable[[TaskDescriptor], tuple[str, Any]],
    ) -> list[tuple[TaskDescriptor, str, Any]]:
        if not tasks:
            return []
        ordered = sorted(tasks, key=lambda item: item.sort_key)
        workers = max(1, min(self.state.max_concurrency, len(ordered)))
        results: list[tuple[TaskDescriptor, str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="renderdiag") as executor:
            futures = [executor.submit(worker, task) for task in ordered]
            for task, future in zip(ordered, futures):
                try:
                    status, value = future.result()
                except Exception as exc:  # Defensive isolation for a single task.
                    status, value = "error", exc
                results.append((task, status, value))
        return results

    def _denoise_worker(self, task: TaskDescriptor) -> tuple[str, Any]:
        try:
            config = self.config
            previous = task.payload["previous"]
            fusion = task.payload["fusion"]
            guidance_fixed = task.payload["guidance_fixed"]
            step_config_fp = task.payload["step_config_fingerprint"]
            payload = denoise_step(
                previous["values"],
                fusion["fusion_vector"],
                task.frame,
                int(task.step),
                guidance_fixed,
                config,
                step_config_fingerprint=step_config_fp,
                seed=self.state.seed,
                total_steps=self.state.denoise_steps,
                quantization_level=self.state.quantization_level,
                noise_seed=self.config.noise_seed,
            )
            injection = self._find_injection(
                "denoise",
                "nonfinite_latent",
                {"frame": task.frame, "step": int(task.step)},
            )
            if injection is not None:
                faulty = list(payload["values"])
                faulty[len(faulty) // 2] = float("nan")
                self._record_injection_hit(
                    injection,
                    stage="denoise",
                    target={"frame": task.frame, "step": int(task.step)},
                    code=E_NONFINITE_LATENT,
                )
                validate_latent_values(
                    faulty,
                    latent_element_count(config),
                    "latent.denoise",
                    injection_id=injection.injection_id,
                )
            validate_latent_values(
                payload["values"],
                latent_element_count(config),
                "latent.denoise",
            )
            return "ok", payload
        except DiagnosticError as error:
            return "error", error
        except Exception as exc:
            return "error", exc

    def _run_denoise_batch(self, deadline: float) -> None:
        if self._text_product is None:
            return
        available = {
            frame: product
            for frame, product in self._init_products.items()
            if frame not in self._blocked_frames
        }
        if not available:
            self.pending_denoise = []
            return
        done: dict[tuple[int, int], dict[str, Any]] = {}
        final_products: dict[int, dict[str, Any]] = {}
        budget = self.state.batch_point_budget
        initial_budget = budget
        prior_pending = {
            (int(item["frame"]), int(item["step"]))
            for item in getattr(self, "_prior_pending_denoise", [])
            if item.get("stage") == "denoise"
        }
        stop_for_budget = False
        for step in range(self.state.denoise_steps):
            if stop_for_budget or self._time_budget_exhausted:
                break
            candidates: list[TaskDescriptor] = []
            for frame in sorted(available):
                if frame in self._blocked_frames:
                    continue
                previous = available[frame] if step == 0 else done.get((frame, step - 1))
                if previous is None:
                    continue
                guidance_fixed = decimal_to_fixed(
                    self.state.guidance,
                    self.config.frac_bits,
                    "denoise.guidance",
                )
                step_config_fp = fingerprint(
                    {
                        "guidance_fixed": guidance_fixed,
                        "step": step,
                        "total_steps": self.state.denoise_steps,
                        "seed": self.state.seed,
                        "noise_seed": self.config.noise_seed,
                        "frame": frame,
                        "shape": list(latent_shape(self.config)),
                        "quantization": self.state.quantization_level,
                        "frac_bits": self.config.frac_bits,
                    },
                    "denoise-step-config-v1",
                )
                upstream = previous["fingerprint"]
                parameters = fingerprint(
                    {
                        "frame": frame,
                        "step": step,
                        "total_steps": self.state.denoise_steps,
                        "guidance_fixed": guidance_fixed,
                        "seed": self.state.seed,
                        "noise_seed": self.config.noise_seed,
                        "shape": list(latent_shape(self.config)),
                        "frac_bits": self.config.frac_bits,
                    },
                    "denoise-params-v1",
                )
                payload, key = self._artifact_lookup(
                    stage="denoise",
                    upstream=upstream,
                    parameters=parameters,
                    quantization_level=self.state.quantization_level,
                    frame=frame,
                    step=step,
                    row=None,
                    invalidated_by=self._changed_fields,
                )
                if payload is not None:
                    done[(frame, step)] = payload
                    available[frame] = payload
                    continue
                if budget <= 0:
                    stop_for_budget = True
                    continue
                candidates.append(
                    TaskDescriptor(
                        stage="denoise",
                        frame=frame,
                        step=step,
                        payload={
                            "previous": previous,
                            "fusion": self._fusion_products[frame],
                            "guidance_fixed": guidance_fixed,
                            "step_config_fingerprint": step_config_fp,
                            "upstream": upstream,
                            "parameters": parameters,
                            "cache_key": key,
                        },
                    )
                )

            if candidates:
                candidates = sorted(
                    candidates,
                    key=lambda task: (
                        0 if (task.frame, int(task.step)) in prior_pending else 1,
                        task.sort_key,
                    ),
                )
                selected = candidates[:budget]
                results = self._run_task_batch(selected, self._denoise_worker)
                for task, status, value in results:
                    self._batch_computed_tasks.append(task.to_dict())
                    budget -= 1
                    self._batch_consumed_points += 1
                    if status == "ok":
                        stored = self._artifact_store_new(
                            stage="denoise",
                            key=task.payload["cache_key"],
                            upstream=task.payload["upstream"],
                            parameters=task.payload["parameters"],
                            quantization_level=self.state.quantization_level,
                            frame=task.frame,
                            step=int(task.step),
                            row=None,
                            payload=value,
                            invalidated_by=self._changed_fields,
                        )
                        done[(task.frame, int(task.step))] = stored
                        available[task.frame] = stored
                    else:
                        error = value if isinstance(value, DiagnosticError) else DiagnosticError(
                            "E_INTERNAL", "denoise", str(value)
                        )
                        self._record_failure(
                            error,
                            stage="denoise",
                            frame=task.frame,
                            step=int(task.step),
                        )
                        self._blocked_frames.add(task.frame)
                if budget <= 0 and len(candidates) > len(selected):
                    stop_for_budget = True
            if time.monotonic() >= deadline:
                self._time_budget_exhausted = True
                break

        self._denoise_products = done
        for frame in sorted(available):
            product = done.get((frame, self.state.denoise_steps - 1))
            if product is not None and frame not in self._blocked_frames:
                final_products[frame] = product
        self._final_latent_products = final_products
        self.consumed_points_total += initial_budget - budget
        pending: list[dict[str, Any]] = []
        if self._text_product is not None:
            for frame in sorted(available):
                if frame in self._blocked_frames:
                    continue
                for step in range(self.state.denoise_steps):
                    if (frame, step) in done:
                        continue
                    task = TaskDescriptor(stage="denoise", frame=frame, step=step)
                    pending.append(
                        {
                            **task.to_dict(),
                            "code": E_BUDGET_PENDING,
                        }
                    )
        self.pending_denoise = sorted(
            pending,
            key=lambda item: (int(item["frame"]), int(item["step"])),
        )

    def _decode_row_worker(self, task: TaskDescriptor) -> tuple[str, Any]:
        try:
            final_latent = task.payload["final_latent"]
            projection = self.projection()
            override: int | None = None
            injection = self._find_injection(
                "decode_row",
                "block_row_oob",
                {"frame": task.frame, "row": int(task.row)},
            )
            if injection is not None:
                self._record_injection_hit(
                    injection,
                    stage="decode_row",
                    target={"frame": task.frame, "row": int(task.row)},
                    code=E_BLOCK_ROW_OUT_OF_RANGE,
                )
                override = self.state.block_rows
            payload = decode_block_row(
                final_latent["values"],
                final_latent["fingerprint"],
                int(task.row),
                projection,
                self.config,
                block_rows=self.state.block_rows,
                quantization_level=self.state.quantization_level,
                requested_row_override=override,
            )
            return "ok", payload
        except DiagnosticError as error:
            return "error", error
        except Exception as exc:
            return "error", exc

    def _decode_available_frames(self) -> None:
        final_products = getattr(self, "_final_latent_products", {})
        projection = self.projection()
        tasks: list[TaskDescriptor] = []
        for frame in sorted(final_products):
            if frame in self._blocked_frames:
                continue
            final_latent = final_products[frame]
            for row in range(self.state.block_rows):
                if (frame, row) in self._row_products:
                    continue
                from .pipeline import latent_band_for_row

                band_start, band_end = latent_band_for_row(
                    row, self.config, self.state.block_rows
                )
                upstream = fingerprint(
                    {
                        "frame": frame,
                        "latent": final_latent["fingerprint"],
                        "region": [band_start, band_end],
                        "projection": projection["metadata"]["fingerprint"],
                    },
                    "decode-row-upstream-v1",
                )
                parameters = fingerprint(
                    {
                        "frame": frame,
                        "row": row,
                        "block_rows": self.state.block_rows,
                        "frame_height": self.config.height,
                        "latent_band": [band_start, band_end],
                        "pixel_y": [
                            row * self.config.height // self.state.block_rows,
                            (row + 1) * self.config.height // self.state.block_rows,
                        ],
                    },
                    "decode-row-params-v1",
                )
                payload, key = self._artifact_lookup(
                    stage="decode_row",
                    upstream=upstream,
                    parameters=parameters,
                    quantization_level=self.state.quantization_level,
                    frame=frame,
                    step=None,
                    row=row,
                    invalidated_by=self._changed_fields,
                )
                if payload is not None:
                    self._row_products[(frame, row)] = payload
                    continue
                tasks.append(
                    TaskDescriptor(
                        stage="decode_row",
                        frame=frame,
                        row=row,
                        payload={
                            "final_latent": final_latent,
                            "upstream": upstream,
                            "parameters": parameters,
                            "cache_key": key,
                        },
                    )
                )
        if tasks:
            results = self._run_task_batch(tasks, self._decode_row_worker)
            for task, status, value in results:
                self._batch_computed_tasks.append(task.to_dict())
                if status == "ok":
                    stored = self._artifact_store_new(
                        stage="decode_row",
                        key=task.payload["cache_key"],
                        upstream=task.payload["upstream"],
                        parameters=task.payload["parameters"],
                        quantization_level=self.state.quantization_level,
                        frame=task.frame,
                        step=None,
                        row=int(task.row),
                        payload=value,
                        invalidated_by=self._changed_fields,
                    )
                    self._row_products[(task.frame, int(task.row))] = stored
                    self._write_png(
                        self.output_dir / "rows" / f"frame_{task.frame:03d}_row_{int(task.row):03d}.png",
                        stored["png_b64"],
                    )
                else:
                    error = value if isinstance(value, DiagnosticError) else DiagnosticError(
                        "E_INTERNAL", "decode_row", str(value)
                    )
                    self._record_failure(
                        error,
                        stage="decode_row",
                        frame=task.frame,
                        row=int(task.row),
                    )

        for frame in sorted(final_products):
            if frame in self._blocked_frames:
                continue
            if frame in self._frame_products:
                continue
            rows = [self._row_products.get((frame, row)) for row in range(self.state.block_rows)]
            if any(row is None for row in rows):
                continue
            final_latent = final_products[frame]
            row_fingerprints = [row["fingerprint"] for row in rows if row is not None]
            denoise_step_fingerprints = [
                self._denoise_products[(frame, step)]["fingerprint"]
                for step in range(self.state.denoise_steps)
                if (frame, step) in self._denoise_products
            ]
            upstream = fingerprint(
                {
                    "frame": frame,
                    "latent": final_latent["fingerprint"],
                    "projection": projection["metadata"]["fingerprint"],
                    "denoise_steps": denoise_step_fingerprints,
                    "rows": row_fingerprints,
                },
                "decode-frame-upstream-v1",
            )
            parameters = fingerprint(
                {"frame": frame, "block_rows": self.state.block_rows},
                "decode-frame-params-v1",
            )
            payload, key = self._artifact_lookup(
                stage="decode_frame",
                upstream=upstream,
                parameters=parameters,
                quantization_level=self.state.quantization_level,
                frame=frame,
                step=None,
                row=None,
                invalidated_by=self._changed_fields,
            )
            if payload is None:
                try:
                    payload = compose_frame(
                        frame,
                        final_latent["fingerprint"],
                        [row for row in rows if row is not None],
                        projection,
                        self.config,
                        block_rows=self.state.block_rows,
                        quantization_level=self.state.quantization_level,
                        denoise_step_fingerprints=denoise_step_fingerprints,
                    )
                    payload = self._artifact_store_new(
                        stage="decode_frame",
                        key=key,
                        upstream=upstream,
                        parameters=parameters,
                        quantization_level=self.state.quantization_level,
                        frame=frame,
                        step=None,
                        row=None,
                        payload=payload,
                        invalidated_by=self._changed_fields,
                    )
                except DiagnosticError as error:
                    self._record_failure(error, stage="decode_frame", frame=frame)
                    self._blocked_frames.add(frame)
                    continue
            self._frame_products[frame] = payload
            self._write_png(self.output_dir / f"frame_{frame:03d}.png", payload["png_b64"])

    def _write_png(self, path: Path, png_b64: str) -> None:
        atomic_write_bytes(path, base64.b64decode(png_b64.encode("ascii")))

    def _current_summary(self) -> dict[str, Any]:
        completed_frames = sorted(self._frame_products)
        block_rows_completed: dict[str, list[int]] = {}
        row_count = 0
        for frame in range(self.config.frame_count):
            rows = sorted(
                row for (row_frame, row) in self._row_products if row_frame == frame
            )
            block_rows_completed[str(frame)] = rows
            row_count += len(rows)
        denoise_done = len(self._denoise_products)
        total_denoise = self.config.frame_count * self.state.denoise_steps
        total_rows = self.config.frame_count * self.state.block_rows
        total = total_denoise + total_rows
        progress = 0.0 if total == 0 else (denoise_done + row_count) / total
        return {
            "completed_frames": completed_frames,
            "completed_frame_count": len(completed_frames),
            "affected_frame_range": (
                [] if not self._affected_frames else [min(self._affected_frames), max(self._affected_frames)]
            ),
            "block_rows_completed": block_rows_completed,
            "denoise_steps_completed": denoise_done,
            "total_denoise_steps": total_denoise,
            "row_blocks_completed": row_count,
            "total_row_blocks": total_rows,
            "progress": round(progress, 6),
            "pending_count": len(self.pending_denoise),
        }

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "completed_frames": [],
            "completed_frame_count": 0,
            "affected_frame_range": [],
            "block_rows_completed": {str(frame): [] for frame in range(self.config.frame_count)},
            "denoise_steps_completed": 0,
            "total_denoise_steps": self.config.frame_count * self.state.denoise_steps,
            "row_blocks_completed": 0,
            "total_row_blocks": self.config.frame_count * self.state.block_rows,
            "progress": 0.0,
            "pending_count": len(self.pending_denoise),
        }

    def _finalize_response(self, started: float) -> dict[str, Any]:
        current_summary = self._current_summary()
        complete = (
            len(self._frame_products) == self.config.frame_count
            and all(frame in self._frame_products for frame in range(self.config.frame_count))
            and not self.failures
        )
        if complete:
            self.latest_complete = {
                "cursor": self.cursor,
                "config_fp": self.state_fingerprint(),
                "summary": current_summary,
            }
            code = "OK"
            stale = False
            baseline_cursor = self.cursor
            baseline_config_fp = self.state_fingerprint()
            response_summary = current_summary
        else:
            stale = True
            baseline_cursor = 0 if self.latest_complete is None else int(self.latest_complete["cursor"])
            baseline_config_fp = (
                None if self.latest_complete is None else self.latest_complete["config_fp"]
            )
            response_summary = (
                self._empty_summary() if self.latest_complete is None else self.latest_complete["summary"]
            )
            code = E_PARTIAL_FAILURE if self.failures else E_STALE_SUMMARY
        elapsed_ms = (time.monotonic() - started) * 1000.0
        response = {
            "code": code,
            "stale": stale,
            "is_latest": bool(complete),
            "requested_cursor": self.cursor,
            "requested_config_fp": self.state_fingerprint(),
            "based_on_cursor": baseline_cursor,
            "based_on_config_fp": baseline_config_fp,
            "summary": response_summary,
            "current_state": current_summary,
            "affected_frame_range": current_summary["affected_frame_range"],
            "changed_fields": list(self._changed_fields),
            "batch": {
                "computed_tasks": list(self._batch_computed_tasks),
                "cache_hits": self._batch_cache_hits,
                "consumed_points": self._batch_consumed_points,
                "pending_count": len(self.pending_denoise),
                "time_budget_exhausted": self._time_budget_exhausted,
            },
            "pending": list(self.pending_denoise),
            "failures": self._sorted_failures(),
            "ledger": self._sorted_ledger(),
            "injection_hits": self.injection_report(),
            "snapshot": {
                "commit_version": self._last_snapshot_version,
                "new_version_created": self._last_snapshot_new,
            },
            "elapsed_ms": round(elapsed_ms, 3),
            "interaction_latency_budget_ms": self.config.interaction_latency_budget_ms,
            "consumed_points_total": self.consumed_points_total,
        }
        return response

    # ------------------------------------------------------------------
    # Snapshot commit / restore
    # ------------------------------------------------------------------

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "runtime": self._runtime_dict(),
            "cursor": self.cursor,
            "render_fingerprint": self.state_fingerprint(),
            "denoise_scope_fingerprint": self._current_denoise_scope,
            "affected_frames": list(self._affected_frames),
            "changed_fields": list(self._changed_fields),
            "pending_denoise": list(self.pending_denoise),
            "ledger": self._sorted_ledger(),
            "failures": self._sorted_failures(),
            "injection_hits": self.injection_report(),
            "latest_complete": self.latest_complete,
            "consumed_points_total": self.consumed_points_total,
            "denoise_done": [
                [frame, step]
                for (frame, step) in sorted(self._denoise_products)
            ],
            "row_done": [
                [frame, row]
                for (frame, row) in sorted(self._row_products)
            ],
            "frames_done": sorted(self._frame_products),
            "complete": len(self._frame_products) == self.config.frame_count and not self.failures,
            "command_history": list(self._command_history),
        }

    def _commit_snapshot(self) -> None:
        state = self._snapshot_state()
        version, created, _ = self.snapshot_store.commit(state)
        self._last_snapshot_version = version
        self._last_snapshot_new = created

    def _restore_snapshot(self) -> None:
        try:
            manifest = self.snapshot_store.load_current()
        except DiagnosticError as error:
            self._record_failure(error, stage="snapshot")
            return
        if manifest is None:
            return
        snapshot_injection = self._find_injection("snapshot", "stale_snapshot_version")
        if snapshot_injection is not None:
            expected = (
                int(snapshot_injection.value)
                if is_int(snapshot_injection.value)
                else -1
            )
            actual = int(manifest.get("artifact_version", -1))
            if actual != expected:
                self._record_injection_hit(
                    snapshot_injection,
                    stage="snapshot",
                    code=E_STALE_SNAPSHOT_VERSION,
                )
                error = DiagnosticError(
                    E_STALE_SNAPSHOT_VERSION,
                    "snapshot.artifact_version",
                    "snapshot artifact version does not match the requested restoration version",
                    detail={"actual": actual, "expected": expected},
                    injection_id=snapshot_injection.injection_id,
                )
                self._record_failure(error, stage="snapshot")
                return
        if int(manifest.get("artifact_version", -1)) != self.config.snapshot_version:
            error = DiagnosticError(
                E_STALE_SNAPSHOT_VERSION,
                "snapshot.artifact_version",
                "snapshot artifact version does not match configuration",
                detail={
                    "actual": manifest.get("artifact_version"),
                    "expected": self.config.snapshot_version,
                },
            )
            self._record_failure(error, stage="snapshot")
            return
        state = manifest.get("state")
        if not isinstance(state, dict):
            self._record_failure(
                DiagnosticError("E_SNAPSHOT_CORRUPT", "snapshot.state", "snapshot state is missing"),
                stage="snapshot",
            )
            return
        try:
            runtime = state["runtime"]
            prompt_name = runtime["prompt"]
            if prompt_name not in self.prompt_set.prompts:
                raise KeyError(prompt_name)
            image_path = runtime.get("image_path")
            image = None if image_path is None else load_image(image_path, self.config)
            motion = load_motion_table(runtime["motion_path"], self.config)
            weights_raw = runtime.get("weights", {})
            weights = {
                name: to_decimal(weights_raw[name], f"snapshot.runtime.weights.{name}", nonnegative=True)
                for name in ("text", "image", "motion")
            }
            restored = RuntimeState(
                prompt_name=prompt_name,
                image=image,
                motion=motion,
                weights=weights,
                denoise_steps=to_int(runtime["denoise_steps"], "snapshot.runtime.denoise_steps", minimum=1),
                seed=to_int(runtime["seed"], "snapshot.runtime.seed", minimum=0),
                guidance=to_decimal(runtime["guidance"], "snapshot.runtime.guidance", nonnegative=True),
                quantization_level=to_int(
                    runtime["quantization_level"],
                    "snapshot.runtime.quantization_level",
                    minimum=1,
                ),
                block_rows=to_int(runtime["block_rows"], "snapshot.runtime.block_rows", minimum=1),
                batch_point_budget=to_int(
                    runtime["batch_point_budget"],
                    "snapshot.runtime.batch_point_budget",
                    minimum=1,
                ),
                max_concurrency=to_int(
                    runtime.get("max_concurrency", self.config.max_concurrency),
                    "snapshot.runtime.max_concurrency",
                    minimum=1,
                ),
            )
            if self.config.height % restored.block_rows != 0:
                raise ValueError("restored block_rows does not divide frame height")
            if restored.quantization_level not in self.config.allowed_quantization_levels:
                raise ValueError("restored quantization level is not allowed")
            if restored.max_concurrency > self.config.max_concurrency:
                raise ValueError("restored max_concurrency exceeds configured maximum")
        except (KeyError, TypeError, ValueError, DiagnosticError) as exc:
            error = exc if isinstance(exc, DiagnosticError) else DiagnosticError(
                "E_SNAPSHOT_CORRUPT",
                "snapshot.runtime",
                "snapshot runtime state cannot be restored",
                detail={"error": repr(exc)},
            )
            self._record_failure(error, stage="snapshot")
            return

        self.state = restored
        self.cursor = int(state.get("cursor", 0))
        self._current_denoise_scope = str(state.get("denoise_scope_fingerprint", self._denoise_scope_fingerprint()))
        self._affected_frames = [int(value) for value in state.get("affected_frames", [])]
        self._changed_fields = [str(value) for value in state.get("changed_fields", [])]
        self.pending_denoise = list(state.get("pending_denoise", []))
        self.consumed_points_total = int(state.get("consumed_points_total", 0))
        self.latest_complete = state.get("latest_complete")
        self._command_history = list(state.get("command_history", []))
        ledger_entries = state.get("ledger", [])
        if isinstance(ledger_entries, list):
            for item in ledger_entries:
                if not isinstance(item, dict):
                    continue
                logical = (
                    item.get("stage"),
                    -1 if item.get("frame") is None else int(item["frame"]),
                    -1 if item.get("step") is None else int(item["step"]),
                    -1 if item.get("row") is None else int(item["row"]),
                )
                self._ledger[logical] = item
        failures_loaded = state.get("failures", [])
        if isinstance(failures_loaded, list):
            self.failures = [dict(item) for item in failures_loaded if isinstance(item, dict)]
        hits_loaded = state.get("injection_hits", [])
        if isinstance(hits_loaded, list):
            for item in hits_loaded:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    self._injection_hits[item["id"]] = dict(item)
        self._last_snapshot_version = int(manifest.get("commit_version", 0))
        self._last_snapshot_new = False
        self._restored = True

    # ------------------------------------------------------------------
    # Execution conveniences and content views
    # ------------------------------------------------------------------

    def run_command(self, command: EditCommand | dict[str, Any], *, complete: bool = True) -> dict[str, Any]:
        self.submit(command)
        if not complete:
            return self.advance()
        return self.drain_current()

    def drain_current(self, *, max_batches: int = 10000) -> dict[str, Any]:
        response = self.advance()
        batches = 1
        while (
            not response["is_latest"]
            and response["pending"]
            and batches < max_batches
        ):
            response = self.advance()
            batches += 1
        return response

    def results(self) -> dict[str, Any]:
        frames = []
        for frame in sorted(self._frame_products):
            product = self._frame_products[frame]
            frames.append(
                {
                    "frame": frame,
                    "content_fingerprint": product["content_fingerprint"],
                    "pixel_sha256": product["pixel_sha256"],
                    "png_sha256": product["png_sha256"],
                    "latent_fingerprint": product["latent_fingerprint"],
                    "denoise_step_fingerprints": product.get("denoise_step_fingerprints", []),
                }
            )
        steps = []
        for (frame, step), product in sorted(self._denoise_products.items()):
            steps.append(
                {
                    "frame": frame,
                    "step": step,
                    "summary": product["summary"],
                    "rounding_direction": product["rounding_direction"],
                    "config_fingerprint": product["step_config_fingerprint"],
                    "fingerprint": product["fingerprint"],
                }
            )
        fusions = [
            {
                "frame": frame,
                "fingerprint": product["fingerprint"],
                "vector": product["fusion_vector"],
                "residual_total": product["residual_total"],
                "residual_abs_max": product["residual_abs_max"],
                "ledger": product["ledger"],
            }
            for frame, product in sorted(self._fusion_products.items())
        ]
        rows = []
        for (frame, row), product in sorted(self._row_products.items()):
            rows.append(
                {
                    "frame": frame,
                    "row": row,
                    "fingerprint": product["fingerprint"],
                    "pixel_sha256": product["pixel_sha256"],
                    "png_sha256": product["png_sha256"],
                    "latent_region": product["latent_region"],
                }
            )
        return {
            "state_fingerprint": self.state_fingerprint(),
            "cursor": self.cursor,
            "frames": frames,
            "denoise_steps": steps,
            "fusion": fusions,
            "rows": rows,
            "text_fingerprint": None if self._text_product is None else self._text_product["fingerprint"],
            "image_fingerprint": None if self._image_product is None else self._image_product["fingerprint"],
            "motion_fingerprints": {
                str(frame): product["fingerprint"]
                for frame, product in sorted(self._motion_products.items())
            },
            "ledger": self._sorted_ledger(),
            "failures": self._sorted_failures(),
            "injection_hits": self.injection_report(),
        }

    def frame_pixels(self, frame: int) -> Any:
        import numpy as np

        product = self._frame_products[frame]
        return np.asarray(product["pixels"], dtype=np.uint8).reshape(
            self.config.height, self.config.width, self.config.channels
        )

    def row_pixels(self, frame: int, row: int) -> Any:
        import numpy as np

        product = self._row_products[(frame, row)]
        y0, y1 = product["pixel_y_range"]
        return np.asarray(product["pixels"], dtype=np.uint8).reshape(
            y1 - y0, self.config.width, self.config.channels
        )

    def export_frames(self, directory: str | os.PathLike[str] | None = None) -> list[str]:
        target = self.output_dir if directory is None else Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for frame in sorted(self._frame_products):
            path = target / f"frame_{frame:03d}.png"
            self._write_png(path, self._frame_products[frame]["png_b64"])
            paths.append(str(path))
        return paths

    def diagnostic_report(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "state_fingerprint": self.state_fingerprint(),
            "injection_hits": self.injection_report(),
            "failures": self._sorted_failures(),
            "ledger": self._sorted_ledger(),
            "snapshot_versions": self.snapshot_store.versions(),
            "artifact_count": self.artifact_store.count(),
            "results": self.results(),
        }
