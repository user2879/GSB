"""Pipeline manifest loading with all-or-nothing strong validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .digest import digest_file
from .errors import SpecError

SUPPORTED_MANIFEST_VERSIONS = (1, 2, 3)
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    digest: str
    version: int

    def as_tuple(self):
        return (self.key, self.digest, self.version)

    def to_dict(self):
        return {"key": self.key, "digest": self.digest, "version": self.version}


@dataclass(frozen=True)
class ArtifactDecl:
    key: str
    stage_id: str
    version: int
    rel_path: str
    abs_path: Path
    refs: tuple[ArtifactRef, ...] = field(default_factory=tuple)

    def ref_tuples(self):
        return tuple(r.as_tuple() for r in self.refs)


@dataclass(frozen=True)
class Stage:
    stage_id: str
    order: int
    artifacts: tuple[ArtifactDecl, ...]


@dataclass(frozen=True)
class Manifest:
    format_version: int
    pipeline_version: int
    stages: tuple[Stage, ...]
    artifacts: dict[str, ArtifactDecl]
    source_path: Path
    workspace_root: Path

    def ordered_artifact_keys(self):
        keys = []
        for stage in self.stages:
            for art in stage.artifacts:
                keys.append(art.key)
        return keys

    def to_dict(self):
        return {
            "format_version": self.format_version,
            "pipeline_version": self.pipeline_version,
            "stages": [
                {
                    "id": st.stage_id,
                    "artifacts": [
                        {
                            "key": a.key,
                            "version": a.version,
                            "file": a.rel_path,
                            "refs": [r.to_dict() for r in a.refs],
                        }
                        for a in st.artifacts
                    ],
                }
                for st in self.stages
            ],
        }


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_manifest(manifest_path, workspace_root) -> Manifest:
    manifest_path = Path(manifest_path)
    workspace_root = Path(workspace_root).resolve()
    errors: list[str] = []

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError("manifest file not found", str(manifest_path))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SpecError("manifest is not valid UTF-8 JSON", "$")

    if not isinstance(raw, dict):
        raise SpecError("manifest root must be an object", "$")

    format_version = raw.get("format_version")
    if not _is_int(format_version) or format_version not in SUPPORTED_MANIFEST_VERSIONS:
        errors.append("format_version")
    pipeline_version = raw.get("pipeline_version", 1)
    if not _is_int(pipeline_version) or pipeline_version < 1:
        errors.append("pipeline_version")

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        errors.append("stages")
        raise SpecError("manifest validation failed", errors)

    stages: list[Stage] = []
    artifacts: dict[str, ArtifactDecl] = {}
    seen_stage_ids: set[str] = set()
    seen_keys: set[str] = set()

    for si, stage_raw in enumerate(stages_raw):
        sp = f"stages[{si}]"
        if not isinstance(stage_raw, dict):
            errors.append(sp)
            continue
        stage_id = stage_raw.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            errors.append(f"{sp}.id")
            continue
        if stage_id in seen_stage_ids:
            errors.append(f"{sp}.id")
        seen_stage_ids.add(stage_id)

        arts_raw = stage_raw.get("artifacts")
        if not isinstance(arts_raw, list) or not arts_raw:
            errors.append(f"{sp}.artifacts")
            continue

        decls: list[ArtifactDecl] = []
        for ai, art_raw in enumerate(arts_raw):
            ap = f"{sp}.artifacts[{ai}]"
            if not isinstance(art_raw, dict):
                errors.append(ap)
                continue
            key = art_raw.get("key")
            if not isinstance(key, str) or not KEY_RE.match(key or ""):
                errors.append(f"{ap}.key")
            elif key in seen_keys:
                errors.append(f"{ap}.key")
            else:
                seen_keys.add(key)

            version = art_raw.get("version")
            if not _is_int(version) or version < 0:
                errors.append(f"{ap}.version")

            rel = art_raw.get("file")
            abs_path = None
            if not isinstance(rel, str) or not rel:
                errors.append(f"{ap}.file")
            else:
                candidate = (workspace_root / rel).resolve()
                try:
                    inside = candidate == workspace_root or candidate.is_relative_to(
                        workspace_root
                    )
                except AttributeError:  # pragma: no cover - Python < 3.9
                    inside = str(candidate).startswith(str(workspace_root))
                if not inside:
                    errors.append(f"{ap}.file")
                elif not candidate.is_file():
                    errors.append(f"{ap}.file")
                else:
                    abs_path = candidate

            refs: list[ArtifactRef] = []
            refs_raw = art_raw.get("refs", [])
            if not isinstance(refs_raw, list):
                errors.append(f"{ap}.refs")
            else:
                for ri, ref_raw in enumerate(refs_raw):
                    rp = f"{ap}.refs[{ri}]"
                    if not isinstance(ref_raw, dict):
                        errors.append(rp)
                        continue
                    rkey = ref_raw.get("key")
                    rdigest = ref_raw.get("digest")
                    rversion = ref_raw.get("version")
                    if not isinstance(rkey, str) or not KEY_RE.match(rkey or ""):
                        errors.append(f"{rp}.key")
                    if not isinstance(rdigest, str) or not rdigest.startswith("sha256:"):
                        errors.append(f"{rp}.digest")
                    if not _is_int(rversion) or rversion < 0:
                        errors.append(f"{rp}.version")
                    refs.append(
                        ArtifactRef(
                            key=rkey if isinstance(rkey, str) else "",
                            digest=rdigest if isinstance(rdigest, str) else "",
                            version=rversion if _is_int(rversion) else -1,
                        )
                    )

            if isinstance(key, str) and KEY_RE.match(key) and abs_path is not None:
                decls.append(
                    ArtifactDecl(
                        key=key,
                        stage_id=stage_id,
                        version=version if _is_int(version) else -1,
                        rel_path=rel,
                        abs_path=abs_path,
                        refs=tuple(refs),
                    )
                )
        stages.append(Stage(stage_id=stage_id, order=si, artifacts=tuple(decls)))

    if errors:
        raise SpecError("manifest validation failed", errors)

    # Cross-entity checks now that every key is known.
    for stage in stages:
        for decl in stage.artifacts:
            ap_key = decl.key
            for ri, ref in enumerate(decl.refs):
                target = artifacts.get(ref.key)
                if target is None:
                    errors.append(_ref_path(stages, decl.key, ri, "key"))
                    continue
                if ref.version > target.version:
                    errors.append(_ref_path(stages, decl.key, ri, "version"))
                actual = digest_file(target.abs_path)
                if ref.digest != actual:
                    errors.append(_ref_path(stages, decl.key, ri, "digest"))
            artifacts[decl.key] = decl

    if errors:
        raise SpecError("manifest validation failed", errors)

    return Manifest(
        format_version=format_version,
        pipeline_version=pipeline_version,
        stages=tuple(stages),
        artifacts=artifacts,
        source_path=manifest_path.resolve(),
        workspace_root=workspace_root,
    )


def _ref_path(stages, owner_key, ri, field):
    for si, stage in enumerate(stages):
        for ai, decl in enumerate(stage.artifacts):
            if decl.key == owner_key:
                return f"stages[{si}].artifacts[{ai}].refs[{ri}].{field}"
    return f"refs.{owner_key}[{ri}].{field}"
