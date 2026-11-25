"""Command session, preview semantics, background batches and snapshot commits."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import compile_weight_q, load_material, load_render_config
from .errors import BizError, issue
from .fixed import ALLOWED_FRAC_BITS, ALLOWED_QUANT_STEPS, BLEND_MODES, EQUAL_WEIGHT_RULES, is_int, is_positive_finite
from .hashing import content_key, layer_key
from .models import ChannelSpec, LayerSpec, LayerWeight, MaterialConfig, RenderConfig, RenderState, TextureData
from .renderer import RenderBatchOutcome, render_batch
from .snapshot import LoadedSnapshot, SnapshotStore
from .texture import load_texture


@dataclass
class RenderJob:
    sequence: int
    done: threading.Event = field(default_factory=threading.Event)
    outcome: RenderBatchOutcome | None = None
    error: BaseException | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


def _deep(value: Any) -> Any:
    return copy.deepcopy(value)


class Session:
    def __init__(
        self,
        *,
        config: RenderConfig,
        material: MaterialConfig,
        texture: TextureData | None,
        state: RenderState,
        store: SnapshotStore,
        source_config_hash: str,
        source_material_hash: str,
        injections: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        version: int = 0,
        rebuilt: bool = False,
        resume_source: str = "fresh",
    ) -> None:
        self.config = config
        self.material = material
        self.texture = texture
        self.state = state
        self.store = store
        self.source_config_hash = source_config_hash
        self.source_material_hash = source_material_hash
        self.injections = injections
        self.issues = issues
        self.version = version
        self.rebuilt = rebuilt
        self.resume_source = resume_source
        self.sequence = max(state.last_sequence, 0)
        self._job: RenderJob | None = None
        self._commit_due = False
        self._last_commit: dict[str, Any] | None = None
        self._lock = threading.Lock()
        if not state.effective_config_hash:
            state.effective_config_hash = content_key("effective-config", config.to_dict())

    @classmethod
    def open(
        cls,
        config_path: str | Path,
        material_path: str | Path,
        state_dir: str | Path,
        *,
        resume: bool = True,
        injections: list[dict[str, Any]] | None = None,
        texture_override: str | Path | None = None,
    ) -> "Session":
        issues: list[dict[str, Any]] = []
        config, config_issues = load_render_config(config_path)
        issues.extend(config_issues)
        if config is None:
            raise BizError("E_CONFIG", "render config could not be loaded", path=str(config_path), stage="config")
        material, material_issues = load_material(material_path, len(config.layers))
        issues.extend(material_issues)
        if material is None:
            raise BizError("E_MATERIAL", "material could not be loaded", path=str(material_path), stage="material")

        state = RenderState()
        store = SnapshotStore(state_dir)
        texture_path = Path(texture_override) if texture_override is not None else Path(config.texture_path)
        texture, texture_issues = load_texture(texture_path, expected_resolution=config.resolution)
        issues.extend(texture_issues)
        version = 0
        rebuilt = False
        resume_source = "fresh"
        loaded: LoadedSnapshot | None = None

        if resume:
            loaded, load_issues = store.load()
            if load_issues:
                issues.extend(load_issues)
                rebuilt = True
                resume_source = "rebuild_corrupt_snapshot"
            if loaded is not None:
                source_config_match = loaded.manifest.get("source_config_hash") == config.source_hash
                source_material_match = loaded.manifest.get("source_material_hash") == material.source_hash
                if not source_config_match or not source_material_match:
                    issues.append(
                        issue(
                            "E_CONFIG_CHANGED_REBUILD",
                            "source config/material hash changed; rebuilding from files",
                            stage="snapshot",
                            changed_config=not source_config_match,
                            changed_material=not source_material_match,
                        )
                    )
                    rebuilt = True
                    resume_source = "rebuild_config_hash_mismatch"
                else:
                    config = loaded.config
                    material = loaded.material
                    state = loaded.state
                    version = loaded.version
                    resume_source = f"snapshot:v{version:08d}"
                    path_from_effective = Path(config.texture_path)
                    restored_texture, restored_issues = load_texture(
                        path_from_effective, expected_resolution=config.resolution
                    )
                    if restored_texture is not None:
                        if restored_texture.content_key != loaded.texture.content_key:
                            issues.append(
                                issue(
                                    "E_TEXTURE_EXTERNAL_CHANGE",
                                    "texture file changed outside the session; downstream keys will invalidate",
                                    path=str(path_from_effective),
                                    stage="texture",
                                )
                            )
                            state.clear_render_products()
                        texture = restored_texture
                    elif texture is None:
                        texture = loaded.texture
                        issues.extend(restored_issues)
                    else:
                        issues.extend(restored_issues)
            elif not load_issues:
                resume_source = "fresh_no_snapshot"
        else:
            rebuilt = True
            resume_source = "forced_rebuild"

        if not (state.effective_config_hash and resume_source.startswith("snapshot:")):
            state.effective_config_hash = content_key("effective-config", config.to_dict())

        return cls(
            config=config,
            material=material,
            texture=texture,
            state=state,
            store=store,
            source_config_hash=config.source_hash,
            source_material_hash=material.source_hash,
            injections=list(injections or []),
            issues=issues,
            version=version,
            rebuilt=rebuilt,
            resume_source=resume_source,
        )

    def _start_job(self) -> RenderJob:
        if self.texture is None:
            raise BizError("E_TEXTURE_UNAVAILABLE", "no valid base texture is loaded", stage="texture")
        job = RenderJob(sequence=self.sequence)
        self._job = job

        def run() -> None:
            try:
                job.outcome = render_batch(
                    self.state,
                    self.config,
                    self.material,
                    self.texture,
                    sequence=self.sequence,
                    injections=self.injections,
                )
            except BaseException as exc:  # keep the session alive and report the stage
                job.error = exc
                failure = (
                    exc.to_dict()
                    if isinstance(exc, BizError)
                    else {
                        "code": "E_RENDER",
                        "message": f"{type(exc).__name__}: {exc}",
                        "stage": "render",
                        "injected": False,
                    }
                )
                self.state.pending = []
                self.state.failures = [failure]
                self.state.last_preview = {
                    "sequence": self.sequence,
                    "stale": False,
                    "complete": False,
                    "completed_layers": sorted(self.state.layers),
                    "completed_layer_count": len(self.state.layers),
                    "total_layers": len(self.config.layers),
                    "progress_ppm": 0,
                    "progress_permille": 0,
                    "affected_layers": [layer.index for layer in self.config.layers],
                    "channels": [],
                    "pending": [],
                    "failures": [failure],
                    "injection_hits": list(self.state.injection_hits),
                    "codes": [str(failure.get("code", "E_RENDER"))],
                }
            finally:
                job.finished_at = time.monotonic()
                job.done.set()

        thread = threading.Thread(target=run, name=f"fixedtex-render-{self.sequence}", daemon=False)
        thread.start()
        return job

    def _join_job(self) -> None:
        job = self._job
        if job is not None:
            job.done.wait()
            self._job = None

    def _settle_previous(self) -> None:
        self._join_job()
        if self._commit_due:
            self.commit()

    def _estimate_affected_layers(self) -> list[int]:
        if self.texture is None:
            return [layer.index for layer in self.config.layers]
        affected: list[int] = []
        for layer in self.config.layers:
            current = self.state.layers.get(layer.index)
            expected = layer_key(layer, self.config, self.texture)
            if current is None or current.key != expected:
                affected.append(layer.index)
        return sorted(affected)

    def _fresh_preview(self) -> dict[str, Any]:
        preview = _deep(self.state.last_preview)
        preview["stale"] = False
        preview["sequence"] = self.sequence
        preview["codes"] = sorted(set(preview.get("codes", [])))
        return preview

    def _stale_preview(self, prior: dict[str, Any]) -> dict[str, Any]:
        preview = _deep(prior)
        if not preview:
            preview = {
                "sequence": None,
                "complete": False,
                "completed_layers": sorted(self.state.layers),
                "completed_layer_count": len(self.state.layers),
                "total_layers": len(self.config.layers),
                "progress_ppm": 0,
                "progress_permille": 0,
                "channels": [],
                "pending": [],
                "failures": [],
            }
        preview["stale"] = True
        preview["stale_reason"] = "E_PREVIEW_STALE"
        preview["code"] = "E_PREVIEW_STALE"
        preview["codes"] = sorted(set(list(preview.get("codes", [])) + ["E_PREVIEW_STALE"]))
        preview["requested_sequence"] = self.sequence
        preview["available_sequence"] = preview.get("sequence")
        preview["affected_layers"] = self._estimate_affected_layers()
        preview["background_render"] = True
        return preview

    def _apply_mutation(self, command: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
        name = command.get("command")
        errors: list[dict[str, Any]] = []
        if name in {"set_channel_weight", "set_weight"}:
            channel_name = command.get("channel")
            layer_value = command.get("layer")
            if channel_name not in self.material.channels:
                return False, [issue("E_WEIGHT", "unknown material channel", path="command.channel", stage="command", channel=channel_name)]
            if not is_int(layer_value) or not 0 <= int(layer_value) < len(self.config.layers):
                return False, [issue("E_WEIGHT", "layer index is outside configured range", path="command.layer", stage="command")]
            weight_q, weight_issues = compile_weight_q(command.get("weight"))
            errors.extend(weight_issues)
            if weight_issues:
                return False, errors
            channel = self.material.channels[channel_name]
            weights = {item.index: item for item in channel.layer_weights}
            raw_weight = float(command.get("weight", 0.0))
            weights[int(layer_value)] = LayerWeight(
                index=int(layer_value), weight_q=weight_q, raw_weight=raw_weight
            )
            new_channel = channel.clone(
                layer_weights=tuple(weights[index] for index in sorted(weights)),
                valid=True,
            )
            channels = dict(self.material.channels)
            channels[channel_name] = new_channel
            self.material = self.material.clone(channels=channels)
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name == "set_channel_base_weight":
            channel_name = command.get("channel")
            if channel_name not in self.material.channels:
                return False, [issue("E_WEIGHT", "unknown material channel", path="command.channel", stage="command", channel=channel_name)]
            weight_q, weight_issues = compile_weight_q(command.get("weight"))
            if weight_issues:
                return False, weight_issues
            channel = self.material.channels[channel_name]
            if weight_q + sum(item.weight_q for item in channel.layer_weights) <= 0:
                return False, [issue("E_WEIGHT", "channel total weight must be positive", path="command.weight", stage="command", channel=channel_name)]
            channels = dict(self.material.channels)
            channels[channel_name] = channel.clone(base_weight_q=weight_q)
            self.material = self.material.clone(channels=channels)
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name in {"set_layer_param", "set_layer"}:
            layer_value = command.get("layer")
            if not is_int(layer_value) or not 0 <= int(layer_value) < len(self.config.layers):
                return False, [issue("E_CONFIG", "layer index is outside configured range", path="command.layer", stage="command")]
            layers = list(self.config.layers)
            layer_index = int(layer_value)
            current = layers[layer_index]
            frequency_milli = current.frequency_milli
            amplitude_milli = current.amplitude_milli
            if "frequency" in command:
                value = command["frequency"]
                if not is_positive_finite(value):
                    return False, [issue("E_FREQ", "frequency must be a positive finite number", path="command.frequency", stage="command", layer=layer_index)]
                frequency_milli = max(1, int(round(float(value) * 1000.0)))
            if "amplitude" in command:
                value = command["amplitude"]
                if not is_positive_finite(value):
                    return False, [issue("E_AMPLITUDE", "amplitude must be a positive finite number", path="command.amplitude", stage="command", layer=layer_index)]
                amplitude_milli = max(1, int(round(float(value) * 1000.0)))
            layers[layer_index] = LayerSpec(
                index=current.index,
                frequency_milli=frequency_milli,
                amplitude_milli=amplitude_milli,
                seed=current.seed,
                valid=True,
            )
            self.config = self.config.clone(layers=tuple(layers))
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name == "set_blend_mode":
            mode = command.get("mode")
            if mode not in BLEND_MODES:
                return False, [issue("E_BLEND_MODE", f"mode must be one of {list(BLEND_MODES)}", path="command.mode", stage="command")]
            self.config = self.config.clone(blend_mode=mode)
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name == "set_quant_step":
            value = command.get("step")
            if not is_int(value) or int(value) not in ALLOWED_QUANT_STEPS:
                return False, [issue("E_QUANT_STEP", f"step must be one of {list(ALLOWED_QUANT_STEPS)}", path="command.step", stage="command")]
            self.config = self.config.clone(fixed=self.config.fixed.__class__(**{**self.config.fixed.to_dict(), "quant_step": int(value)}))
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name == "set_frac_bits":
            value = command.get("bits")
            if not is_int(value) or int(value) not in ALLOWED_FRAC_BITS:
                return False, [issue("E_FRAC_BITS", f"bits must be one of {list(ALLOWED_FRAC_BITS)}", path="command.bits", stage="command")]
            self.config = self.config.clone(fixed=self.config.fixed.__class__(**{**self.config.fixed.to_dict(), "fraction_bits": int(value)}))
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, []

        if name == "set_equal_weight_rule":
            rule = command.get("rule")
            if rule not in EQUAL_WEIGHT_RULES:
                return False, [issue("E_WEIGHT", f"rule must be one of {list(EQUAL_WEIGHT_RULES)}", path="command.rule", stage="command")]
            self.material = self.material.clone(equal_weight_rule=rule)
            return True, []

        if name == "set_thread_pool_size":
            value = command.get("size")
            if not is_int(value) or not 1 <= int(value) <= 64:
                return False, [issue("E_THREAD_POOL", "thread pool size must be in [1, 64]", path="command.size", stage="command")]
            self.config = self.config.clone(thread_pool_size=int(value))
            return True, []

        if name == "replace_texture":
            texture_path = command.get("path")
            if not isinstance(texture_path, str) or not texture_path:
                return False, [issue("E_COMMAND", "replace_texture requires a path", path="command.path", stage="command")]
            candidate = Path(texture_path)
            if not candidate.is_absolute():
                cwd_candidate = (Path.cwd() / candidate).resolve()
                config_candidate = (Path(self.config.source_path).parent / candidate).resolve()
                candidate = cwd_candidate if cwd_candidate.exists() else config_candidate
            replacement, replacement_issues = load_texture(candidate, expected_resolution=self.config.resolution)
            if replacement is None:
                return False, replacement_issues
            self.texture = replacement
            self.config = self.config.clone(texture_path=str(candidate))
            self.state.effective_config_hash = content_key("effective-config", self.config.to_dict())
            return True, replacement_issues

        if name in {"render", "advance", "noop"}:
            return True, []
        if name == "commit":
            return True, []
        return False, [issue("E_COMMAND", f"unknown command: {name}", path="command", stage="command")]

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict) or not isinstance(command.get("command"), str):
            return {
                "ok": False,
                "command": command,
                "errors": [issue("E_COMMAND", "command must be an object with a string command field", path="command", stage="command")],
                "preview": _deep(self.state.last_preview),
            }
        self._settle_previous()
        if command.get("command") == "commit":
            manifest = self.commit()
            return {
                "ok": True,
                "command": command,
                "errors": [],
                "committed": {
                    "version": manifest["version"],
                    "ledger_count": manifest["ledger_count"],
                    "complete": self.state.composite is not None,
                },
                "preview": self._fresh_preview(),
            }

        prior_preview = _deep(self.state.last_preview)
        valid, errors = self._apply_mutation(command)
        if not valid:
            return {
                "ok": False,
                "command": command,
                "errors": errors
                or [issue("E_COMMAND", "command validation failed", path="command", stage="command")],
                "preview": prior_preview,
            }

        self.sequence += 1
        self.state.last_sequence = self.sequence
        try:
            job = self._start_job()
        except BizError as exc:
            return {
                "ok": False,
                "command": command,
                "errors": [exc.to_dict()],
                "preview": prior_preview,
            }
        preview_budget = max(0, self.config.preview_budget_ms)
        completed_in_budget = job.done.wait(preview_budget / 1000.0)
        if completed_in_budget:
            preview = self._fresh_preview()
        else:
            preview = self._stale_preview(prior_preview)
        preview["elapsed_ms"] = int((time.monotonic() - job.started_at) * 1000)
        preview["preview_budget_ms"] = preview_budget
        preview["command_valid"] = True
        preview["command_errors"] = errors
        result: dict[str, Any] = {
            "ok": True,
            "command": command,
            "errors": errors,
            "preview": preview,
        }

        if self.sequence % self.config.commit_interval_commands == 0:
            if job.done.is_set():
                self._join_job()
                manifest = self.commit()
                result["committed"] = {
                    "version": manifest["version"],
                    "ledger_count": manifest["ledger_count"],
                    "complete": self.state.composite is not None,
                }
            else:
                # Preserve the preview deadline. The next command/finalize
                # settles and commits this command before doing new work.
                self._commit_due = True
                result["commit_pending"] = True
        return result

    def advance(self, *, wait: bool = True) -> dict[str, Any]:
        self._settle_previous()
        try:
            job = self._start_job()
        except BizError as exc:
            failure = exc.to_dict()
            self.state.failures = [failure]
            self.state.pending = []
            self.state.last_preview = {
                "sequence": self.sequence,
                "stale": False,
                "complete": False,
                "completed_layers": sorted(self.state.layers),
                "completed_layer_count": len(self.state.layers),
                "total_layers": len(self.config.layers),
                "progress_ppm": 0,
                "progress_permille": 0,
                "affected_layers": [layer.index for layer in self.config.layers],
                "channels": [],
                "pending": [],
                "failures": [failure],
                "injection_hits": list(self.state.injection_hits),
                "codes": [exc.code],
            }
            return _deep(self.state.last_preview)
        if wait:
            job.done.wait()
            self._job = None
        return _deep(self.state.last_preview)

    def complete(self, *, max_batches: int | None = None) -> dict[str, Any]:
        self._settle_previous()
        limit = max_batches if max_batches is not None else max(16, len(self.config.layers) * 4 + 16)
        batches = 0
        while batches < limit:
            if self.state.composite is not None and not self.state.pending:
                break
            if self.state.failures:
                break
            self.advance(wait=True)
            batches += 1
            if self.state.failures:
                break
        summary = _deep(self.state.last_preview)
        summary["batches_used"] = batches
        return summary

    def commit(self) -> dict[str, Any]:
        self._join_job()
        if self.texture is None:
            raise BizError("E_TEXTURE_UNAVAILABLE", "cannot commit without a valid texture", stage="snapshot")
        manifest = self.store.save(
            self.state,
            self.config,
            self.material,
            self.texture,
            source_config_hash=self.source_config_hash,
            source_material_hash=self.source_material_hash,
        )
        self.version = int(manifest["version"])
        self.state.version = self.version
        self._last_commit = manifest
        self._commit_due = False
        return manifest

    def finalize(self, *, drain: bool = True, force_commit: bool = True) -> dict[str, Any]:
        if drain:
            summary = self.complete()
        else:
            self._join_job()
            summary = _deep(self.state.last_preview)
        committed = None
        if force_commit:
            manifest = self.commit()
            committed = {
                "version": manifest["version"],
                "ledger_count": manifest["ledger_count"],
                "complete": self.state.composite is not None,
            }
        self._join_job()
        return {"summary": summary, "committed": committed, "version": self.version}

    def status(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sequence": self.sequence,
            "resume_source": self.resume_source,
            "rebuilt": self.rebuilt,
            "issues": self.issues,
            "config_hash": content_key("effective-config", self.config.to_dict()),
            "source_config_hash": self.source_config_hash,
            "source_material_hash": self.source_material_hash,
            "texture": self.texture.summary() if self.texture is not None else None,
            "preview": _deep(self.state.last_preview),
            "committed": self._last_commit is not None,
        }
