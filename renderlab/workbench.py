"""Stateful incremental workbench and edit-sequence conveniences."""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    LoadResult,
    ResolutionTier,
    WorkbenchConfig,
    load_workbench,
)
from .engine import RenderEngine, RenderReport
from .errors import E_CONFIG_INVALID, WorkbenchError
from .preview import PreviewBroker, PreviewResponse
from .util import sha256_hex


@dataclass
class RunHandle:
    thread: threading.Thread
    broker: PreviewBroker

    def result(self, timeout: float | None = None) -> RenderReport:
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("render is still running")
        return getattr(self.thread, "report")

    @property
    def done(self) -> bool:
        return not self.thread.is_alive()


class Workbench:
    """Owns mutable config/sample state and a persistent local cache."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        temp_dir: str | Path | None = None,
        batch_size: int | None = None,
        preview_broker: PreviewBroker | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.result: LoadResult = load_workbench(
            self.config_path,
            temp_dir_override=str(temp_dir) if temp_dir is not None else None,
        )
        if self.result.config is None or self.result.samples is None:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "workbench input could not be loaded",
                component="workbench",
                details={"failures": [failure.to_dict() for failure in self.result.failures]},
            )
        if not self.result.can_render:
            first = self.result.failures[0] if self.result.failures else None
            raise WorkbenchError(
                first.code if first else E_CONFIG_INVALID,
                first.message if first else "workbench input is not renderable",
                component=first.component if first else "workbench",
                item=first.item if first else "",
                details={"failures": [failure.to_dict() for failure in self.result.failures]},
            )
        self.config = self.result.config
        self.samples = self.result.samples
        self.temp_dir = Path(temp_dir or self.config.runtime.temp_dir).resolve()
        self.patches: list[dict[str, Any]] = []
        self.requested_batch_size = (
            int(batch_size)
            if batch_size is not None
            else self.config.render.batch_sizes[len(self.config.render.batch_sizes) // 2]
        )
        self._validate_batch_size(self.requested_batch_size)
        self.preview_broker = preview_broker or PreviewBroker(self.config.runtime.preview_budget_ms)
        self.last_report: RenderReport | None = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, *, batch_size: int | None = None) -> RenderReport:
        selected_batch = self.requested_batch_size if batch_size is None else int(batch_size)
        self._validate_batch_size(selected_batch)
        engine = RenderEngine(
            self.result,
            batch_size=selected_batch,
            temp_dir=str(self.temp_dir),
            patches=copy.deepcopy(self.patches),
            preview_broker=self.preview_broker,
        )
        report = engine.run()
        self.last_report = report
        return report

    def run_async(self, *, batch_size: int | None = None) -> RunHandle:
        selected_batch = self.requested_batch_size if batch_size is None else int(batch_size)
        self._validate_batch_size(selected_batch)
        scheduled_signature = sha256_hex(
            "scheduled-preview-v1",
            self.samples.semantic_dict(),
            self.config.semantic_dict(),
            self.patches,
            selected_batch,
        )[:32]
        self.preview_broker.begin_run(
            scheduled_signature,
            {"phase": "scheduled", "frames_done": 0, "latest_parameter_signature": scheduled_signature},
        )

        def target() -> None:
            try:
                report = self.run(batch_size=selected_batch)
            except Exception as exc:  # Preserve a failed report for the handle.
                report = RenderReport(
                    {
                        "status": "failed",
                        "business_code": E_CONFIG_INVALID,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "frames": [],
                    }
                )
                self.preview_broker.complete_run(
                    scheduled_signature,
                    {"phase": "failed", "business_code": E_CONFIG_INVALID},
                )
            current = threading.current_thread()
            setattr(current, "report", report)

        thread = threading.Thread(target=target, name="renderlab-workbench", daemon=True)
        thread.start()
        return RunHandle(thread=thread, broker=self.preview_broker)

    def preview(self, budget_ms: int | None = None) -> PreviewResponse:
        return self.preview_broker.request_preview(budget_ms)

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------
    def apply_edit(self, edit: dict[str, Any]) -> None:
        if not isinstance(edit, dict):
            raise WorkbenchError(E_CONFIG_INVALID, "edit must be a mapping", component="edit")
        operation = edit.get("op")
        if operation == "set_batch_size":
            value = edit.get("value", edit.get("batch_size"))
            if not isinstance(value, int) or isinstance(value, bool):
                raise WorkbenchError(E_CONFIG_INVALID, "batch size edit requires integer value", component="edit", item="set_batch_size")
            self._validate_batch_size(value)
            self.requested_batch_size = value
        elif operation == "set_guidance":
            value = edit.get("value", edit.get("guidance_scale"))
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise WorkbenchError(E_CONFIG_INVALID, "guidance edit requires a finite non-negative number", component="edit", item="set_guidance")
            self.config.render.guidance_scale = float(value)
        elif operation == "set_denoise_steps":
            value = edit.get("value", edit.get("denoise_steps"))
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise WorkbenchError(E_CONFIG_INVALID, "denoise step edit requires a positive integer", component="edit", item="set_denoise_steps")
            if value != len(self.result.noise_table) or value != len(self.result.guidance_table):
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    "denoise step edit must match generated noise/guidance table length",
                    component="edit",
                    item="set_denoise_steps",
                )
            self.config.render.denoise_steps = value
        elif operation == "set_seeds":
            seeds = edit.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise WorkbenchError(E_CONFIG_INVALID, "set_seeds requires a non-empty list", component="edit", item="set_seeds")
            validated: list[int] = []
            for index, seed in enumerate(seeds):
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > self.config.render.seed_max:
                    raise WorkbenchError(
                        E_CONFIG_INVALID,
                        "seed is outside the configured range",
                        component="edit",
                        item=f"set_seeds[{index}]",
                        details={"seed": seed, "seed_max": self.config.render.seed_max},
                    )
                validated.append(int(seed))
            self.samples.seeds = validated
            self.result.valid_seeds = list(validated)
        elif operation == "replace_prompt":
            prompt_id = edit.get("prompt_id")
            tokens = edit.get("tokens")
            if not isinstance(prompt_id, str) or not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) and token for token in tokens):
                raise WorkbenchError(E_CONFIG_INVALID, "replace_prompt requires prompt_id and non-empty tokens", component="edit", item="replace_prompt")
            unknown = sorted(set(tokens) - set(self.samples.vocab))
            if unknown:
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    "replacement prompt uses tokens outside the restricted vocabulary",
                    component="edit",
                    item="replace_prompt",
                    details={"unknown_tokens": unknown},
                )
            matches = [prompt for prompt in self.samples.prompts if prompt.prompt_id == prompt_id]
            if not matches:
                raise WorkbenchError(E_CONFIG_INVALID, "prompt id not found", component="edit", item="replace_prompt", details={"prompt_id": prompt_id})
            matches[0].tokens = tuple(str(token) for token in tokens)
        elif operation == "set_resolution_tiers":
            tiers_raw = edit.get("tiers")
            if not isinstance(tiers_raw, list) or not tiers_raw:
                raise WorkbenchError(E_CONFIG_INVALID, "set_resolution_tiers requires a non-empty list", component="edit", item="set_resolution_tiers")
            tiers: list[ResolutionTier] = []
            names: set[str] = set()
            for index, raw in enumerate(tiers_raw):
                if not isinstance(raw, dict):
                    raise WorkbenchError(E_CONFIG_INVALID, "tier must be a mapping", component="edit", item=f"set_resolution_tiers[{index}]")
                name, width, height = raw.get("name"), raw.get("width"), raw.get("height")
                if not isinstance(name, str) or not name or name in names or not isinstance(width, int) or isinstance(width, bool) or width <= 0 or not isinstance(height, int) or isinstance(height, bool) or height <= 0:
                    raise WorkbenchError(E_CONFIG_INVALID, "invalid tier fields", component="edit", item=f"set_resolution_tiers[{index}]")
                names.add(name)
                tiers.append(ResolutionTier(name=name, width=int(width), height=int(height)))
            for index in range(1, len(tiers)):
                if tiers[index].area <= tiers[index - 1].area:
                    raise WorkbenchError(E_CONFIG_INVALID, "resolution tier areas must be strictly increasing", component="edit", item="set_resolution_tiers")
            self.config.render.resolution_tiers = tiers
        elif operation == "patch_latent":
            patch = {key: value for key, value in edit.items() if key != "op"}
            patch["op"] = "patch_latent"
            # RenderEngine performs the complete shape/type validation when it
            # constructs the next run.
            self.patches.append(patch)
        elif operation == "clear_patches":
            self.patches.clear()
        else:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "unknown edit operation",
                component="edit",
                item=str(operation),
            )

    def apply_edits(self, edits: list[dict[str, Any]]) -> None:
        for edit in edits:
            self.apply_edit(edit)

    def run_edit_sequence(self, edits: list[dict[str, Any]]) -> list[RenderReport]:
        reports: list[RenderReport] = []
        for edit in edits:
            self.apply_edit(edit)
            reports.append(self.run())
        return reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_batch_size(self, value: int) -> None:
        if value not in self.config.render.batch_sizes:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "batch size is not one of the configured scale tiers",
                component="runtime",
                item="batch_size",
                details={"batch_size": value, "options": list(self.config.render.batch_sizes)},
            )

    def clear_cache(self, *, keep_config: bool = True) -> None:
        import shutil

        for name in ("artifacts", "state", "snapshots", "runs"):
            path = self.temp_dir / name
            if path.exists():
                shutil.rmtree(path)
        if not keep_config:
            raise ValueError("clear_cache only removes generated local state")


def load_edit_sequence(path: str | Path) -> list[dict[str, Any]]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "edit sequence must contain an operations list",
            component="edit-sequence",
        )
    operations = payload["operations"]
    if not all(isinstance(operation, dict) for operation in operations):
        raise WorkbenchError(E_CONFIG_INVALID, "each edit operation must be a mapping", component="edit-sequence")
    return list(operations)
