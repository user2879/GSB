"""Versioned geometry state derived from one model and one evaluation grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import SCHEMA_VERSION, content_hash
from .errors import DomainError, ErrorCode
from .geometry import (
    InvalidationClosure,
    PatchGeometry,
    aggregate_geometry_hashes,
)
from .model import NURBSModel
from .nurbs import NURBSPatch
from .render import DISPLAY_MODES, RenderBuffer, SoftwareRenderer, display_hashes


STATE_SCHEMA_VERSION = "cad-surface.geometry-state.v1"
GRID_CONVENTION = "inclusive_endpoints.u_major_then_v.v1"


def default_parameter_values(sample_count: int) -> tuple[float, ...]:
    if sample_count < 2:
        raise DomainError(ErrorCode.GEOM_STALE, "sample grid must have at least two samples per axis")
    return tuple(float(item) for item in np.linspace(0.0, 1.0, sample_count, dtype="<f8"))


@dataclass(frozen=True)
class StateHashes:
    geometry_hash: str
    normals_hash: str
    iso_hash: str
    tessellation_hash: str
    display_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_hash": self.geometry_hash,
            "normals_hash": self.normals_hash,
            "iso_hash": self.iso_hash,
            "tessellation_hash": self.tessellation_hash,
            "display_hashes": dict(sorted(self.display_hashes.items())),
        }


class GeometryState:
    def __init__(
        self,
        *,
        version: int,
        parent_version: int | None,
        model: NURBSModel,
        geometries: Mapping[str, PatchGeometry],
        hashes: StateHashes,
        grid_resolution: int,
        grid_convention: str = GRID_CONVENTION,
        last_closure: InvalidationClosure | None = None,
    ) -> None:
        self.version = int(version)
        self.parent_version = None if parent_version is None else int(parent_version)
        self.model = model
        self.geometries = dict(geometries)
        self.hashes = hashes
        self.grid_resolution = int(grid_resolution)
        self.grid_convention = str(grid_convention)
        self.last_closure = last_closure
        self._validate()

    def _validate(self) -> None:
        patch_keys = {patch.key for patch in self.model.patches}
        if set(self.geometries) != patch_keys:
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "state patch cache keys do not match model patches",
                model_keys=sorted(patch_keys),
                cache_keys=sorted(self.geometries),
            )
        for key, geometry in self.geometries.items():
            if geometry.patch_key != key:
                raise DomainError(ErrorCode.GEOM_STALE, "patch cache key mismatch", key=key)
            if geometry.version != self.version:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "patch cache version does not match state version",
                    patch_key=key,
                    geometry_version=geometry.version,
                    state_version=self.version,
                )
            expected_control_hash = self.model.patch(key).control_content_hash()
            if geometry.control_hash != expected_control_hash:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "patch cache was derived from different control points",
                    patch_key=key,
                    state_version=self.version,
                )
        aggregate = aggregate_geometry_hashes(self.geometries)
        for field in ("geometry_hash", "normals_hash", "iso_hash", "tessellation_hash"):
            if getattr(self.hashes, field) != aggregate[field]:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "state aggregate hash does not match patch caches",
                    field=field,
                    state_version=self.version,
                )
        missing_modes = set(DISPLAY_MODES) - set(self.hashes.display_hashes)
        if missing_modes:
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "state is missing display buffer hashes",
                missing=sorted(missing_modes),
            )

    def _hash_body(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "version": self.version,
            "parent_version": self.parent_version,
            "model_hash": self.model.content_hash(),
            "grid_resolution": self.grid_resolution,
            "grid_convention": self.grid_convention,
            **self.hashes.to_dict(),
        }

    @property
    def state_hash(self) -> str:
        return content_hash(self._hash_body())

    @property
    def geometry_hash(self) -> str:
        return self.hashes.geometry_hash

    @property
    def normals_hash(self) -> str:
        return self.hashes.normals_hash

    @property
    def display_hashes(self) -> dict[str, str]:
        return dict(self.hashes.display_hashes)

    def render_all(self) -> dict[str, RenderBuffer]:
        return SoftwareRenderer(self.model, self.geometries).render_all()

    def verify_display(self) -> dict[str, str]:
        actual = display_hashes(self.render_all())
        if actual != self.display_hashes:
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "re-rendered display buffers differ from the state hash",
                version=self.version,
                expected=self.display_hashes,
                actual=actual,
            )
        return actual

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "version": self.version,
            "parent_version": self.parent_version,
            "model": self.model.to_dict(),
            "hashes": self.hashes.to_dict(),
            "state_hash": self.state_hash,
            "grid_resolution": self.grid_resolution,
            "grid_convention": self.grid_convention,
            "cache_index": {key: geometry.geometry_hash for key, geometry in sorted(self.geometries.items())},
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        geometries: Mapping[str, PatchGeometry],
    ) -> "GeometryState":
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported geometry state schema version",
                expected=STATE_SCHEMA_VERSION,
                actual=payload.get("schema_version"),
            )
        try:
            hashes_payload = dict(payload["hashes"])
            hashes = StateHashes(
                geometry_hash=str(hashes_payload["geometry_hash"]),
                normals_hash=str(hashes_payload["normals_hash"]),
                iso_hash=str(hashes_payload["iso_hash"]),
                tessellation_hash=str(hashes_payload["tessellation_hash"]),
                display_hashes={str(key): str(value) for key, value in hashes_payload["display_hashes"].items()},
            )
            state = cls(
                version=int(payload["version"]),
                parent_version=None if payload.get("parent_version") is None else int(payload["parent_version"]),
                model=NURBSModel.from_dict(payload["model"]),
                geometries=geometries,
                hashes=hashes,
                grid_resolution=int(payload["grid_resolution"]),
                grid_convention=str(payload["grid_convention"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError(ErrorCode.SCHEMA_VERSION, "invalid geometry state payload", error=str(exc)) from exc
        expected_state_hash = payload.get("state_hash")
        if expected_state_hash != state.state_hash:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "geometry state hash does not match the snapshot",
                expected=expected_state_hash,
                actual=state.state_hash,
            )
        cache_index = {str(key): str(value) for key, value in payload.get("cache_index", {}).items()}
        actual_index = {key: geometry.geometry_hash for key, geometry in sorted(geometries.items())}
        if cache_index != actual_index:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "state cache index does not match patch caches",
                expected=cache_index,
                actual=actual_index,
            )
        return state


def _state_hashes(geometries: Mapping[str, PatchGeometry], displays: Mapping[str, str]) -> StateHashes:
    aggregate = aggregate_geometry_hashes(geometries)
    return StateHashes(
        geometry_hash=aggregate["geometry_hash"],
        normals_hash=aggregate["normals_hash"],
        iso_hash=aggregate["iso_hash"],
        tessellation_hash=aggregate["tessellation_hash"],
        display_hashes=dict(sorted(displays.items())),
    )


def build_initial_state(
    model: NURBSModel,
    *,
    version: int = 0,
    parent_version: int | None = None,
    grid_resolution: int = 16,
) -> GeometryState:
    values = default_parameter_values(grid_resolution)
    geometries = {
        patch.key: PatchGeometry.build(patch, version=version, u_values=values, v_values=values)
        for patch in model.patches
    }
    displays = display_hashes(SoftwareRenderer(model, geometries).render_all())
    return GeometryState(
        version=version,
        parent_version=parent_version,
        model=model,
        geometries=geometries,
        hashes=_state_hashes(geometries, displays),
        grid_resolution=grid_resolution,
        last_closure=None,
    )


def derive_edited_state(
    previous: GeometryState,
    *,
    new_version: int,
    operation_version: int,
    patch_key: str,
    index_u: int,
    index_v: int,
    new_value: Sequence[float],
) -> tuple[GeometryState, NURBSPatch, InvalidationClosure]:
    patch = previous.model.patch(patch_key)
    patch.validate_control_index(index_u, index_v)
    new_patch = patch.with_control_point(index_u, index_v, new_value)
    old_geometry = previous.geometries[patch_key]
    closure = old_geometry.closure_for_control_point(index_u, index_v)
    new_geometry = old_geometry.apply_control_edit(
        new_patch,
        new_version=new_version,
        index_u=index_u,
        index_v=index_v,
        closure=closure,
        operation_version=operation_version,
    )
    geometries = {
        key: geometry.rebind_version(new_version)
        for key, geometry in previous.geometries.items()
        if key != patch_key
    }
    geometries[patch_key] = new_geometry
    model = previous.model.with_patch(new_patch)
    displays = display_hashes(SoftwareRenderer(model, geometries).render_all())
    state = GeometryState(
        version=new_version,
        parent_version=previous.version,
        model=model,
        geometries=geometries,
        hashes=_state_hashes(geometries, displays),
        grid_resolution=previous.grid_resolution,
        grid_convention=previous.grid_convention,
        last_closure=closure,
    )
    return state, new_patch, closure
