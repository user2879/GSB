"""Content-addressed stage artifacts and their persisted cache bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import E_SNAPSHOT_STALE, RenderError
from .util import (
    array_sha256,
    hash_object,
    read_json,
    sha256_bytes,
    write_bytes_fast,
    write_json_fast,
)


@dataclass(frozen=True)
class PromptArtifact:
    key: str
    content_fp: str
    normalized_prompt: str
    embedding_fp: tuple[int, ...]
    hits: tuple[dict[str, Any], ...]
    unknown_terms: tuple[str, ...]
    params: dict[str, Any]
    deps: tuple[str, ...]

    @property
    def embedding(self) -> list[float]:
        return [value / 1_000_000.0 for value in self.embedding_fp]

    def metadata(self) -> dict[str, Any]:
        return {
            "normalized_prompt": self.normalized_prompt,
            "embedding_fp": list(self.embedding_fp),
            "embedding": self.embedding,
            "hits": list(self.hits),
            "unknown_terms": list(self.unknown_terms),
            "params": self.params,
            "deps": list(self.deps),
        }


@dataclass(frozen=True)
class LatentArtifact:
    key: str
    stage: str
    content_fp: str
    array: np.ndarray
    params: dict[str, Any]
    deps: tuple[str, ...]
    metadata: dict[str, Any]

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "content_fp": self.content_fp,
            "params": self.params,
            "deps": list(self.deps),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DecodedFrameArtifact:
    key: str
    content_fp: str
    pixels: np.ndarray
    png: bytes
    blocks: tuple[dict[str, Any], ...]
    params: dict[str, Any]
    deps: tuple[str, ...]

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "content_fp": self.content_fp,
            "pixels_sha256": array_sha256(self.pixels),
            "png_sha256": sha256_bytes(self.png),
            "blocks": list(self.blocks),
            "params": self.params,
            "deps": list(self.deps),
        }


@dataclass(frozen=True)
class CompositionArtifact:
    key: str
    content_fp: str
    frame_fps: tuple[str, ...]
    frame_keys: tuple[str, ...]
    preview_png: bytes
    params: dict[str, Any]
    deps: tuple[str, ...]

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "content_fp": self.content_fp,
            "frame_fps": list(self.frame_fps),
            "frame_keys": list(self.frame_keys),
            "preview_png_sha256": sha256_bytes(self.preview_png),
            "params": self.params,
            "deps": list(self.deps),
        }


def prompt_content_fingerprint(
    normalized_prompt: str,
    embedding_fp: tuple[int, ...],
    hits: tuple[dict[str, Any], ...],
    unknown_terms: tuple[str, ...],
    params: dict[str, Any],
) -> str:
    return hash_object(
        {
            "normalized_prompt": normalized_prompt,
            "embedding_fp": list(embedding_fp),
            "hits": list(hits),
            "unknown_terms": list(unknown_terms),
            "params": params,
        },
        domain="fixedvideo-prompt-artifact",
    )


def make_prompt_artifact(
    *,
    key: str,
    normalized_prompt: str,
    embedding_fp: tuple[int, ...],
    hits: tuple[dict[str, Any], ...],
    unknown_terms: tuple[str, ...],
    params: dict[str, Any],
    deps: tuple[str, ...],
) -> PromptArtifact:
    content_fp = prompt_content_fingerprint(
        normalized_prompt, embedding_fp, hits, unknown_terms, params
    )
    return PromptArtifact(
        key=key,
        content_fp=content_fp,
        normalized_prompt=normalized_prompt,
        embedding_fp=embedding_fp,
        hits=hits,
        unknown_terms=unknown_terms,
        params=params,
        deps=deps,
    )


def latent_content_fingerprint(stage: str, array: np.ndarray, params: dict[str, Any]) -> str:
    return hash_object(
        {"stage": stage, "array_sha256": array_sha256(array), "params": params},
        domain="fixedvideo-latent-artifact",
    )


def make_latent_artifact(
    *,
    key: str,
    stage: str,
    array: np.ndarray,
    params: dict[str, Any],
    deps: tuple[str, ...],
    metadata: dict[str, Any],
) -> LatentArtifact:
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return LatentArtifact(
        key=key,
        stage=stage,
        content_fp=latent_content_fingerprint(stage, contiguous, params),
        array=contiguous,
        params=params,
        deps=deps,
        metadata=metadata,
    )


def decoded_content_fingerprint(pixels: np.ndarray, params: dict[str, Any]) -> str:
    return hash_object(
        {"pixels_sha256": array_sha256(pixels), "params": params},
        domain="fixedvideo-decoded-artifact",
    )


def make_decoded_artifact(
    *,
    key: str,
    pixels: np.ndarray,
    png: bytes,
    blocks: tuple[dict[str, Any], ...],
    params: dict[str, Any],
    deps: tuple[str, ...],
) -> DecodedFrameArtifact:
    contiguous = np.ascontiguousarray(pixels, dtype=np.uint8)
    return DecodedFrameArtifact(
        key=key,
        content_fp=decoded_content_fingerprint(contiguous, params),
        pixels=contiguous,
        png=png,
        blocks=blocks,
        params=params,
        deps=deps,
    )


def composition_content_fingerprint(
    frame_fps: tuple[str, ...], preview_png: bytes, params: dict[str, Any]
) -> str:
    return hash_object(
        {
            "frame_fps": list(frame_fps),
            "preview_png_sha256": sha256_bytes(preview_png),
            "params": params,
        },
        domain="fixedvideo-composition-artifact",
    )


def make_composition_artifact(
    *,
    key: str,
    frame_fps: tuple[str, ...],
    frame_keys: tuple[str, ...],
    preview_png: bytes,
    params: dict[str, Any],
    deps: tuple[str, ...],
) -> CompositionArtifact:
    return CompositionArtifact(
        key=key,
        content_fp=composition_content_fingerprint(frame_fps, preview_png, params),
        frame_fps=frame_fps,
        frame_keys=frame_keys,
        preview_png=preview_png,
        params=params,
        deps=deps,
    )


class ArtifactCache:
    """All immutable artifacts produced during a session."""

    def __init__(self) -> None:
        self.prompts: dict[str, PromptArtifact] = {}
        self.latents: dict[str, LatentArtifact] = {}
        self.decodes: dict[str, DecodedFrameArtifact] = {}
        self.compositions: dict[str, CompositionArtifact] = {}

    def put(self, artifact: Any) -> None:
        if isinstance(artifact, PromptArtifact):
            self.prompts[artifact.key] = artifact
        elif isinstance(artifact, LatentArtifact):
            self.latents[artifact.key] = artifact
        elif isinstance(artifact, DecodedFrameArtifact):
            self.decodes[artifact.key] = artifact
        elif isinstance(artifact, CompositionArtifact):
            self.compositions[artifact.key] = artifact
        else:  # pragma: no cover - internal misuse
            raise TypeError(f"unsupported artifact type: {type(artifact)!r}")

    def counts(self) -> dict[str, int]:
        return {
            "prompt": len(self.prompts),
            "latent": len(self.latents),
            "decode": len(self.decodes),
            "composition": len(self.compositions),
            "total": (
                len(self.prompts)
                + len(self.latents)
                + len(self.decodes)
                + len(self.compositions)
            ),
        }

    def save(self, destination: str | Path, include_keys: set[str] | None = None) -> dict[str, Any]:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        for directory in ("prompt", "latent", "decode", "composition"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []

        def included(key: str) -> bool:
            return include_keys is None or key in include_keys

        for key, artifact in sorted(self.prompts.items()):
            if not included(key):
                continue
            relative = f"prompt/{key}.json"
            write_json_fast(root / relative, {"key": key, **artifact.metadata()})
            entries.append(
                {
                    "stage": "prompt",
                    "key": key,
                    "content_fp": artifact.content_fp,
                    "file": relative,
                }
            )

        for key, artifact in sorted(self.latents.items()):
            if not included(key):
                continue
            array_rel = f"latent/{key}.npy"
            meta_rel = f"latent/{key}.json"
            np.save(root / array_rel, artifact.array, allow_pickle=False)
            write_json_fast(root / meta_rel, {"key": key, **artifact.metadata_dict()})
            entries.append(
                {
                    "stage": "latent",
                    "key": key,
                    "content_fp": artifact.content_fp,
                    "array_file": array_rel,
                    "meta_file": meta_rel,
                }
            )

        for key, artifact in sorted(self.decodes.items()):
            if not included(key):
                continue
            raw_rel = f"decode/{key}.raw"
            png_rel = f"decode/{key}.png"
            meta_rel = f"decode/{key}.json"
            write_bytes_fast(root / raw_rel, artifact.pixels.tobytes(order="C"))
            write_bytes_fast(root / png_rel, artifact.png)
            write_json_fast(root / meta_rel, {"key": key, **artifact.metadata_dict()})
            entries.append(
                {
                    "stage": "decode",
                    "key": key,
                    "content_fp": artifact.content_fp,
                    "pixels_file": raw_rel,
                    "png_file": png_rel,
                    "meta_file": meta_rel,
                }
            )

        for key, artifact in sorted(self.compositions.items()):
            if not included(key):
                continue
            png_rel = f"composition/{key}.png"
            meta_rel = f"composition/{key}.json"
            write_bytes_fast(root / png_rel, artifact.preview_png)
            write_json_fast(root / meta_rel, {"key": key, **artifact.metadata_dict()})
            entries.append(
                {
                    "stage": "composition",
                    "key": key,
                    "content_fp": artifact.content_fp,
                    "png_file": png_rel,
                    "meta_file": meta_rel,
                }
            )

        saved_counts = {
            "prompt": sum(1 for entry in entries if entry["stage"] == "prompt"),
            "latent": sum(1 for entry in entries if entry["stage"] == "latent"),
            "decode": sum(1 for entry in entries if entry["stage"] == "decode"),
            "composition": sum(1 for entry in entries if entry["stage"] == "composition"),
        }
        saved_counts["total"] = sum(saved_counts.values())
        index = {"schema_version": 1, "entries": entries, "counts": saved_counts}
        write_json_fast(root / "index.json", index)
        return index

    @classmethod
    def load(cls, source: str | Path) -> "ArtifactCache":
        root = Path(source)
        try:
            index = read_json(root / "index.json")
        except (OSError, json.JSONDecodeError) as exc:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed cache index is missing or unreadable",
                {"cache_root": str(root), "error": str(exc)},
            ) from exc
        if index.get("schema_version") != 1 or not isinstance(index.get("entries"), list):
            raise RenderError(E_SNAPSHOT_STALE, "committed cache index has an unsupported schema")
        cache = cls()
        try:
            for entry in index["entries"]:
                stage = entry["stage"]
                key = entry["key"]
                expected_fp = entry["content_fp"]
                if stage == "prompt":
                    meta = read_json(root / entry["file"])
                    params = meta["params"]
                    embedding_fp = tuple(int(value) for value in meta["embedding_fp"])
                    hits = tuple(meta["hits"])
                    unknown = tuple(meta["unknown_terms"])
                    artifact = make_prompt_artifact(
                        key=key,
                        normalized_prompt=meta["normalized_prompt"],
                        embedding_fp=embedding_fp,
                        hits=hits,
                        unknown_terms=unknown,
                        params=params,
                        deps=tuple(meta["deps"]),
                    )
                elif stage == "latent":
                    meta = read_json(root / entry["meta_file"])
                    array = np.load(root / entry["array_file"], allow_pickle=False)
                    if array.dtype != np.int64:
                        raise ValueError("latent cache array is not int64")
                    artifact = make_latent_artifact(
                        key=key,
                        stage=meta["stage"],
                        array=array,
                        params=meta["params"],
                        deps=tuple(meta["deps"]),
                        metadata=meta["metadata"],
                    )
                elif stage == "decode":
                    meta = read_json(root / entry["meta_file"])
                    shape = tuple(int(value) for value in meta["params"]["pixel_shape"])
                    raw = (root / entry["pixels_file"]).read_bytes()
                    pixels = np.frombuffer(raw, dtype=np.uint8)
                    if int(pixels.size) != int(np.prod(shape)):
                        raise ValueError("decode cache pixel length does not match pixel_shape")
                    pixels = pixels.reshape(shape).copy()
                    png = (root / entry["png_file"]).read_bytes()
                    artifact = make_decoded_artifact(
                        key=key,
                        pixels=pixels,
                        png=png,
                        blocks=tuple(meta["blocks"]),
                        params=meta["params"],
                        deps=tuple(meta["deps"]),
                    )
                elif stage == "composition":
                    meta = read_json(root / entry["meta_file"])
                    preview = (root / entry["png_file"]).read_bytes()
                    artifact = make_composition_artifact(
                        key=key,
                        frame_fps=tuple(meta["frame_fps"]),
                        frame_keys=tuple(meta["frame_keys"]),
                        preview_png=preview,
                        params=meta["params"],
                        deps=tuple(meta["deps"]),
                    )
                else:
                    raise ValueError(f"unknown cache stage: {stage}")
                if artifact.content_fp != expected_fp:
                    raise ValueError(
                        f"content fingerprint mismatch while loading {stage} artifact {key}"
                    )
                cache.put(artifact)
        except RenderError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed cache is corrupt",
                {"cache_root": str(root), "error": str(exc)},
            ) from exc
        return cache
