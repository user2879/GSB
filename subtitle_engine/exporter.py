"""Atomic frame/report export using versioned generations and a commit pointer."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Union

from ._numpy import np
from .util import atomic_write_json, ensure_dir, read_json, sha256_bytes


def _file_digest(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            import os

            os.fsync(dst.fileno())
        import os

        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _existing_commit(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "commit.json"
    if not path.exists():
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def repair_stable_mirrors(output_dir: Path) -> None:
    """Recover stable mirrors from the committed generation after a crash."""

    commit = _existing_commit(output_dir)
    if not commit:
        return
    generation = output_dir / str(commit.get("generation", ""))
    if not generation.is_dir():
        return
    for name in ("frames.npy", "report.json"):
        source = generation / name
        if source.exists():
            _atomic_copy(source, output_dir / name)


def _write_frames(source: Union["np.ndarray", Path], destination: Path) -> None:
    if isinstance(source, Path):
        temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temp)
            import os

            with temp.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temp, destination)
        finally:
            if temp.exists():
                temp.unlink()
        return
    with destination.open("wb") as handle:
        np.save(handle, source, allow_pickle=False)
        handle.flush()
        import os

        os.fsync(handle.fileno())


def export_outputs(
    output_dir: Path,
    frames: Union["np.ndarray", Path],
    report: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    repair_stable_mirrors(output_dir)
    version = str(report["version"])
    generations = ensure_dir(output_dir / "generations")
    generation = generations / f"{version}-{uuid.uuid4().hex[:12]}"
    generation.mkdir(parents=True, exist_ok=False)

    frames_path = generation / "frames.npy"
    report_path = generation / "report.json"
    _write_frames(frames, frames_path)
    atomic_write_json(report_path, report)
    frames_digest = _file_digest(frames_path)
    report_digest = _file_digest(report_path)
    relative_generation = str(generation.relative_to(output_dir))
    commit = {
        "version": version,
        "frame_count": int(report["frame_count"]),
        "frames_digest": frames_digest,
        "report_digest": report_digest,
        "generation": relative_generation,
    }
    atomic_write_json(output_dir / "commit.json", commit)

    # Stable mirrors are a convenience view.  The commit pointer above is the
    # atomic pair authority; these mirrors are repaired on the next run if a
    # process is killed between the two replacements.
    _atomic_copy(frames_path, output_dir / "frames.npy")
    _atomic_copy(report_path, output_dir / "report.json")
    return commit


def existing_output_matches(output_dir: Path, version: str) -> bool:
    commit = _existing_commit(output_dir)
    if not commit or commit.get("version") != version:
        return False
    generation = output_dir / str(commit.get("generation", ""))
    frames_path = generation / "frames.npy"
    report_path = generation / "report.json"
    if not frames_path.exists() or not report_path.exists():
        return False
    basic_match = (
        _file_digest(frames_path) == commit.get("frames_digest")
        and _file_digest(report_path) == commit.get("report_digest")
    )
    if not basic_match:
        return False
    try:
        report = read_json(report_path)
        frames = np.load(frames_path, mmap_mode="r", allow_pickle=False)
        frame_count = int(frames.shape[0])
        del frames
    except (OSError, ValueError):
        return False
    return (
        report.get("version") == version
        and report.get("frame_count") == commit.get("frame_count")
        and frame_count == commit.get("frame_count")
    )


def existing_output_version(output_dir: Path) -> str | None:
    commit = _existing_commit(output_dir)
    return str(commit.get("version")) if commit else None
