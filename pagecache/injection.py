from __future__ import annotations

import shutil
from pathlib import Path

from .cache import PageCache
from .errors import EngineError, ErrorCode
from .manifest import read_manifest


def truncate_manifest_copy(source: str | Path, target: str | Path, *, bytes_to_remove: int = 7) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    data = source_path.read_bytes()
    if bytes_to_remove <= 0 or bytes_to_remove >= len(data):
        raise EngineError(ErrorCode.E_FORMAT, "invalid tail truncation size", path="$injection.bytes_to_remove")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data[:-bytes_to_remove])
    return target_path


def corrupt_segment_checksum_copy(overflow_dir: str | Path, target: str | Path) -> Path:
    source_dir = Path(overflow_dir)
    segments = sorted(source_dir.glob("*.seg"), key=lambda item: item.name)
    if not segments:
        raise EngineError(ErrorCode.E_INTEGRITY, "no overflow segment is available for checksum injection", path=str(source_dir))
    source = segments[0]
    data = bytearray(source.read_bytes())
    if not data:
        raise EngineError(ErrorCode.E_INTEGRITY, "overflow segment is empty", path=str(source))
    data[-1] ^= 0x01
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(bytes(data))
    return target_path


def trigger_zero_budget(
    *,
    manifest_path: str | Path,
    pages_dir: str | Path,
    overflow_dir: str | Path,
    page_size: int,
) -> None:
    manifest = read_manifest(manifest_path)
    PageCache(manifest=manifest, pages_dir=pages_dir, overflow_dir=overflow_dir, budget_bytes=0, page_size=page_size)


def copy_segment_for_replay(overflow_dir: str | Path, target_dir: str | Path) -> list[Path]:
    """Utility used by tests and demos to create an isolated spill workspace."""

    source_dir = Path(overflow_dir)
    destination = Path(target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in sorted(source_dir.glob("*.seg"), key=lambda item: item.name):
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied
