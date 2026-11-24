"""Persistent scene state and deterministic edit application."""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import codes
from .config import SceneConfig
from .models import Camera, LoadedProject, StoredView, ViewRecord
from .util import array_fingerprint, atomic_write_bytes, atomic_write_json, file_fingerprint, read_json

STATE_SCHEMA_VERSION = 1


class EditError(Exception):
    def __init__(self, code: str, message: str, *, view_index: int | None = None, level: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.view_index = view_index
        self.level = level


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vh = np.linalg.svd(matrix)
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result.astype(np.float64)


class SceneState:
    def __init__(
        self,
        config: SceneConfig,
        views: dict[int, StoredView],
        sh_mods: list[dict[str, Any]] | None = None,
        applied_edits: list[dict[str, Any]] | None = None,
        source_sample: str | None = None,
    ) -> None:
        self.config = config
        self.views = views
        self.sh_mods = list(sh_mods or [])
        self.applied_edits = list(applied_edits or [])
        self.source_sample = source_sample or ""

    @property
    def applied_edit_seqs(self) -> list[int]:
        return sorted(int(item["seq"]) for item in self.applied_edits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied_edits": self.applied_edits,
            "config": self.config.as_dict(),
            "schema_version": STATE_SCHEMA_VERSION,
            "sh_mods": self.sh_mods,
            "source_sample": self.source_sample,
            "views": {str(key): self.views[key].as_dict() for key in sorted(self.views)},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneState":
        if int(value.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported state schema version")
        config = SceneConfig.from_dict(value["config"])
        views = {int(key): StoredView.from_dict(item) for key, item in value["views"].items()}
        return cls(
            config=config,
            views=views,
            sh_mods=list(value.get("sh_mods", [])),
            applied_edits=list(value.get("applied_edits", [])),
            source_sample=value.get("source_sample"),
        )

    def copy(self) -> "SceneState":
        return SceneState.from_dict(copy.deepcopy(self.as_dict()))

    def view_record(self, work_dir: Path, view_index: int) -> ViewRecord:
        stored = self.views[view_index]
        image_path = work_dir / "state" / stored.image_path
        depth_path = work_dir / "state" / stored.depth_path
        return ViewRecord(
            view_index=view_index,
            image_path=image_path,
            depth_path=depth_path,
            camera=stored.camera_model(),
            image_fingerprint=stored.image_fingerprint,
            depth_fingerprint=stored.depth_fingerprint,
            image_size=(int(stored.image_size[0]), int(stored.image_size[1])),
            injection_id=stored.injection_id,
        )

    def records(self, work_dir: Path) -> dict[int, ViewRecord]:
        return {view_index: self.view_record(work_dir, view_index) for view_index in sorted(self.views)}

    def sh_mods_for(self, view_index: int, level: int, kernel_index: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for mod in self.sh_mods:
            if int(mod.get("view_index", -1)) != view_index:
                continue
            if level not in [int(item) for item in mod.get("levels", [])]:
                continue
            target = int(mod.get("kernel_index", -1))
            if target not in (-1, kernel_index):
                continue
            result.append(mod)
        return sorted(result, key=lambda item: int(item.get("seq", 0)))

    def write_final_sample(self, out_dir: Path) -> Path:
        """Bake the current state into a plan-free sample for a fresh full rebuild."""

        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "images").mkdir(parents=True)
        (out_dir / "depth").mkdir(parents=True)
        atomic_write_json(out_dir / "config.json", self.config.as_dict())
        cameras = [self.views[key].camera for key in sorted(self.views)]
        atomic_write_json(out_dir / "cameras.json", cameras)
        for view_index in sorted(self.views):
            stored = self.views[view_index]
            source_image = Path(self._work_dir) / "state" / stored.image_path
            source_depth = Path(self._work_dir) / "state" / stored.depth_path
            image_target = out_dir / "images" / f"view_{view_index:04d}.png"
            depth_target = out_dir / "depth" / f"view_{view_index:04d}.npy"
            atomic_write_bytes(image_target, source_image.read_bytes())
            atomic_write_bytes(depth_target, source_depth.read_bytes())
        atomic_write_json(out_dir / "sh_mods.json", self.sh_mods)
        (out_dir / "edits.jsonl").write_text("", encoding="utf-8")
        atomic_write_json(out_dir / "injections.json", [])
        return out_dir

    def attach_work_dir(self, work_dir: Path) -> None:
        self._work_dir = str(Path(work_dir).resolve())


class StateManager:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.state_dir = self.work_dir / "state"
        self.state_path = self.state_dir / "state.json"
        self.views_dir = self.state_dir / "views"

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> SceneState:
        state = SceneState.from_dict(read_json(self.state_path))
        state.attach_work_dir(self.work_dir)
        return state

    def save(self, state: SceneState) -> None:
        state.attach_work_dir(self.work_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, state.as_dict())

    def _store_view_asset(
        self,
        view_index: int,
        image_source: Path,
        depth_source: Path,
        camera: Camera,
        injection_id: str | None = None,
    ) -> StoredView:
        view_dir = self.views_dir / f"view_{view_index:04d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        image_bytes = image_source.read_bytes()
        depth_bytes = depth_source.read_bytes()
        image_target = view_dir / "image.png"
        depth_target = view_dir / "depth.npy"
        atomic_write_bytes(image_target, image_bytes)
        atomic_write_bytes(depth_target, depth_bytes)
        width, height = camera_image_size(image_target)
        return StoredView(
            view_index=view_index,
            image_path=f"views/view_{view_index:04d}/image.png",
            depth_path=f"views/view_{view_index:04d}/depth.npy",
            camera=camera.as_dict(),
            image_fingerprint=file_fingerprint(image_target),
            depth_fingerprint=file_fingerprint(depth_target),
            image_size=[width, height],
            injection_id=injection_id,
        )

    def initialize(self, project: LoadedProject) -> SceneState:
        if project.config is None:
            raise ValueError("cannot initialize state without a valid config")
        self.views_dir.mkdir(parents=True, exist_ok=True)
        views: dict[int, StoredView] = {}
        for view_index in sorted(project.views):
            view = project.views[view_index]
            views[view_index] = self._store_view_asset(
                view_index,
                view.image_path,
                view.depth_path,
                view.camera,
                view.injection_id,
            )
        sh_mods: list[dict[str, Any]] = []
        sh_path = project.sample_dir / "sh_mods.json"
        if sh_path.exists():
            try:
                loaded = read_json(sh_path)
                if isinstance(loaded, list):
                    sh_mods = [dict(item) for item in loaded if isinstance(item, dict)]
            except (OSError, json.JSONDecodeError):
                sh_mods = []
        state = SceneState(
            config=project.config.copy(),
            views=views,
            sh_mods=sh_mods,
            source_sample=str(project.sample_dir),
        )
        state.attach_work_dir(self.work_dir)
        self.save(state)
        return state

    def reset(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)

    def apply_edit(self, state: SceneState, edit: dict[str, Any]) -> SceneState:
        seq = int(edit["seq"])
        if seq in state.applied_edit_seqs:
            return state
        edit_type = str(edit["type"])
        params = dict(edit.get("params", {}))
        if edit_type == "view_replace":
            view_index = int(params["view_index"])
            if view_index not in state.views:
                raise EditError(codes.E_EDIT_VIEW_MISSING, f"view {view_index} does not exist", view_index=view_index)
            stored = self._read_asset(state, params, view_index)
            state.views[view_index] = stored
        elif edit_type == "view_add":
            view_index = int(params["view_index"])
            if view_index in state.views:
                raise EditError(codes.E_EDIT_VIEW_DUPLICATE, f"view {view_index} already exists", view_index=view_index)
            stored = self._read_asset(state, params, view_index)
            state.views[view_index] = stored
        elif edit_type == "view_remove":
            view_index = int(params["view_index"])
            if view_index not in state.views:
                raise EditError(codes.E_EDIT_VIEW_MISSING, f"view {view_index} does not exist", view_index=view_index)
            state.views.pop(view_index)
            shutil.rmtree(self.views_dir / f"view_{view_index:04d}", ignore_errors=True)
        elif edit_type == "camera_tweak":
            view_index = int(params["view_index"])
            if view_index not in state.views:
                raise EditError(codes.E_EDIT_VIEW_MISSING, f"view {view_index} does not exist", view_index=view_index)
            stored = state.views[view_index]
            camera = stored.camera_model()
            rotation = camera.rotation
            if "rotation_delta" in params:
                delta = np.asarray(params["rotation_delta"], dtype=np.float64)
                if delta.shape != (3, 3) or not np.all(np.isfinite(delta)):
                    raise EditError(codes.E_EDIT_JSON, "rotation_delta must be a finite 3x3 matrix", view_index=view_index)
                if not np.all(delta == 0.0):
                    rotation = _orthonormalize(rotation + delta)
            translation = camera.translation
            if "translation_delta" in params:
                delta_t = np.asarray(params["translation_delta"], dtype=np.float64)
                if delta_t.shape != (3,) or not np.all(np.isfinite(delta_t)):
                    raise EditError(codes.E_EDIT_JSON, "translation_delta must be a finite vector of length 3", view_index=view_index)
                if not np.all(delta_t == 0.0):
                    translation = translation + delta_t
            new_camera = Camera(view_index, camera.intrinsic.copy(), rotation, translation)
            stored.camera = new_camera.as_dict()
        elif edit_type == "level_scale":
            level = int(params["level"])
            value = float(params["value"])
            if not 0 <= level < state.config.levels or value <= 0:
                raise EditError(codes.E_EDIT_LEVEL, f"invalid level_scale level={level} value={value}", level=level)
            state.config.level_scale_overrides[level] = value
        elif edit_type == "level_density":
            level = int(params["level"])
            value = float(params["value"])
            if not 0 <= level < state.config.levels or value <= 0:
                raise EditError(codes.E_EDIT_DENSITY, f"invalid level_density level={level} value={value}", level=level)
            state.config.level_density[level] = value
        elif edit_type == "sh_modify":
            view_index = int(params["view_index"])
            if view_index not in state.views:
                raise EditError(codes.E_EDIT_VIEW_MISSING, f"view {view_index} does not exist", view_index=view_index)
            levels = [int(item) for item in params.get("levels", [])]
            if any(level < 0 or level >= state.config.levels for level in levels):
                raise EditError(codes.E_EDIT_LEVEL, "sh_modify references an invalid level", view_index=view_index)
            coefficient = int(params["coefficient"])
            if not 0 <= coefficient < (state.config.sh_degree + 1) ** 2:
                raise EditError(codes.E_EDIT_SH_COEFF, "sh coefficient outside basis", view_index=view_index)
            channel = int(params["channel"])
            if not 0 <= channel < 3:
                raise EditError(codes.E_EDIT_SH_COEFF, "sh channel outside 0..2", view_index=view_index)
            mod = {
                "channel": channel,
                "coefficient": coefficient,
                "delta": float(params["delta"]),
                "kernel_index": int(params["kernel_index"]),
                "levels": levels,
                "seq": seq,
                "view_index": view_index,
            }
            state.sh_mods.append(mod)
        elif edit_type == "quant_switch":
            frac_bits = int(params.get("frac_bits", state.config.frac_bits))
            gear = str(params.get("quant_gear", state.config.quant_gear))
            if frac_bits not in codes.ALLOWED_FRAC_BITS:
                raise EditError(codes.E_EDIT_QUANT, f"unsupported frac_bits {frac_bits}")
            if gear not in codes.QUANT_GEARS:
                raise EditError(codes.E_EDIT_QUANT, f"unsupported quant_gear {gear}")
            state.config.frac_bits = frac_bits
            state.config.quant_gear = gear
        else:
            raise EditError(codes.E_EDIT_TYPE, f"unsupported edit type {edit_type}")
        state.applied_edits.append(dict(edit))
        return state

    def _read_asset(self, state: SceneState, params: dict[str, Any], view_index: int) -> StoredView:
        source_sample = Path(state.source_sample)
        asset_dir = (source_sample / str(params.get("asset_dir", ""))).resolve()
        if not asset_dir.is_dir():
            raise EditError(codes.E_ASSET_MISSING, f"asset directory not found: {asset_dir}", view_index=view_index)
        image_candidates = sorted(asset_dir.glob("view_*.png")) + sorted(asset_dir.glob("image.png"))
        depth_candidates = sorted(asset_dir.glob("view_*.npy")) + sorted(asset_dir.glob("depth.npy"))
        if not image_candidates:
            raise EditError(codes.E_ASSET_MISSING, f"asset image missing in {asset_dir}", view_index=view_index)
        if not depth_candidates:
            raise EditError(codes.E_ASSET_MISSING, f"asset depth missing in {asset_dir}", view_index=view_index)
        camera_path = asset_dir / "camera.json"
        if not camera_path.exists():
            raise EditError(codes.E_CAMERA_MISSING, f"asset camera missing in {asset_dir}", view_index=view_index)
        try:
            camera_value = read_json(camera_path)
            camera = Camera.from_dict(camera_value)
        except Exception as exc:
            raise EditError(codes.E_CAMERA_JSON, f"invalid asset camera: {exc}", view_index=view_index) from exc
        camera.view_index = view_index
        if camera.intrinsic.shape != (3, 3) or abs(float(np.linalg.det(camera.intrinsic))) <= 1e-9:
            raise EditError(codes.E_CAMERA_INTRINSIC, "asset camera intrinsic is invalid", view_index=view_index)
        if not np.allclose(camera.rotation.T @ camera.rotation, np.eye(3), atol=1e-6):
            raise EditError(codes.E_CAMERA_ROTATION, "asset camera rotation is not orthogonal", view_index=view_index)
        try:
            with Image.open(image_candidates[0]) as image:
                image.load()
                if image.size != state.config.image_size or image.mode != state.config.pixel_mode:
                    raise EditError(
                        codes.E_IMAGE_RESOLUTION,
                        f"asset image expected {state.config.image_size} {state.config.pixel_mode}, got {image.size} {image.mode}",
                        view_index=view_index,
                    )
        except EditError:
            raise
        except Exception as exc:
            raise EditError(codes.E_IMAGE_DECODE, f"asset image cannot be decoded: {exc}", view_index=view_index) from exc
        try:
            depth = np.load(depth_candidates[0], allow_pickle=False)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.shape != state.config.image_size or not np.all(np.isfinite(depth)):
                raise EditError(codes.E_DEPTH_SHAPE, "asset depth shape/value is invalid", view_index=view_index)
        except EditError:
            raise
        except Exception as exc:
            raise EditError(codes.E_DEPTH_SHAPE, f"asset depth cannot be loaded: {exc}", view_index=view_index) from exc
        return self._store_view_asset(view_index, image_candidates[0], depth_candidates[0], camera)


def camera_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    return int(width), int(height)
