"""Object file layout and crash-safe physical transfer between tier dirs.

Files are always named by their content digest.  A transfer never leaves an
object committed on both tiers:

1. write the payload to a unique ``.tmp`` file inside the *target* directory;
2. fsync and verify the digest of the temp file;
3. ``os.replace`` the temp file onto the final target path (atomic on the same
   filesystem and on Windows);
4. remove the source file only after the target is durably in place.

If a crash happens during step 1-3 the source is still the only valid copy and
the stray temp file is cleaned on recovery.  If a crash happens during step 4
both copies can transiently exist; recovery keeps the digest-valid target and
deletes the redundant source, never inventing a completion report.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .errors import IntegrityError

_TMP_SUFFIX = ".tmp"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tier_dir(workdir: str | Path, tier_id: str) -> Path:
    return Path(workdir) / "tiers" / tier_id


def staging_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "staging"


def object_path(workdir: str | Path, tier_id: str, digest: str) -> Path:
    return tier_dir(workdir, tier_id) / digest


def ensure_layout(workdir: str | Path, tier_ids: Iterable[str]) -> None:
    workdir = Path(workdir)
    staging_dir(workdir).mkdir(parents=True, exist_ok=True)
    for tier_id in tier_ids:
        tier_dir(workdir, tier_id).mkdir(parents=True, exist_ok=True)


def cleanup_temp_files(workdir: str | Path, tier_ids: Iterable[str]) -> list[str]:
    removed: list[str] = []
    roots = [staging_dir(workdir)] + [tier_dir(workdir, tid) for tid in tier_ids]
    for root in roots:
        if not root.exists():
            continue
        for tmp in root.glob(f"*{_TMP_SUFFIX}"):
            tmp.unlink()
            removed.append(str(tmp))
    return removed


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_place(workdir: str | Path, tier_id: str, source: str | Path, expected_digest: str) -> Path:
    """Move ``source`` into a tier atomically after verifying its digest."""
    source = Path(source)
    target_dir = tier_dir(workdir, tier_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / expected_digest

    fd, tmp_name = tempfile.mkstemp(prefix=f"{expected_digest}.", suffix=_TMP_SUFFIX, dir=str(target_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            with source.open("rb") as inp:
                while True:
                    chunk = inp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        actual_digest = sha256_file(tmp_path)
        if actual_digest != expected_digest:
            raise IntegrityError(
                f"digest mismatch while placing object into tier {tier_id}",
                {
                    "tier_id": tier_id,
                    "expected_digest": expected_digest,
                    "actual_digest": actual_digest,
                },
            )
        os.replace(tmp_path, final_path)
        _fsync_dir(target_dir)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path


def transfer_object(
    workdir: str | Path,
    key: str,
    source_tier: str,
    target_tier: str,
    expected_digest: str,
) -> Tuple[Path, Path | None]:
    """Physically move one object between tiers with verified atomic switch."""
    source_path = object_path(workdir, source_tier, expected_digest)
    target_path = object_path(workdir, target_tier, expected_digest)
    if not source_path.exists():
        # May already be on target after a crash between replace and delete.
        if target_path.exists() and sha256_file(target_path) == expected_digest:
            return target_path, source_path
        raise IntegrityError(
            f"source object file missing for key {key}",
            {
                "key": key,
                "source_tier": source_tier,
                "target_tier": target_tier,
                "expected_digest": expected_digest,
            },
        )

    atomic_place(workdir, target_tier, source_path, expected_digest)
    # Target is durable; removing the source is the final, non-atomic step.
    source_path.unlink(missing_ok=True)
    _fsync_dir(tier_dir(workdir, source_tier))
    if source_path.exists():
        raise IntegrityError(
            f"object {key} remains on both tiers after transfer",
            {
                "key": key,
                "source_tier": source_tier,
                "target_tier": target_tier,
                "source_digest": sha256_file(source_path),
                "target_digest": sha256_file(target_path),
            },
        )
    return target_path, None


def scan_tiers(workdir: str | Path, tier_ids: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """Map digest -> size for every committed object file, grouped by tier."""
    result: Dict[str, Dict[str, int]] = {}
    for tier_id in tier_ids:
        files: Dict[str, int] = {}
        directory = tier_dir(workdir, tier_id)
        if directory.exists():
            for entry in directory.iterdir():
                if entry.is_file() and not entry.name.endswith(_TMP_SUFFIX):
                    files[entry.name] = entry.stat().st_size
        result[tier_id] = files
    return result


def verify_object_file(workdir: str | Path, tier_id: str, expected_digest: str) -> str:
    path = object_path(workdir, tier_id, expected_digest)
    if not path.exists():
        raise IntegrityError(
            "object file missing during verification",
            {"tier_id": tier_id, "expected_digest": expected_digest},
        )
    actual = sha256_file(path)
    if actual != expected_digest:
        raise IntegrityError(
            "object file digest mismatch",
            {"tier_id": tier_id, "expected_digest": expected_digest, "actual_digest": actual},
        )
    return actual
