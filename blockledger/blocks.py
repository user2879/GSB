from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import block_seed, content_digest, sha256_hex


def deterministic_bytes(domain: str, seed_material: dict[str, Any], size: int) -> bytes:
    """Generate size stable bytes with NumPy PCG64 seeded by a SHA-256 subkey."""

    material = {"domain": domain, "size": size, "material": seed_material}
    seed = block_seed(sha256_hex(material))
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.bytes(size)


def initial_block_bytes(volume: dict[str, Any], block_index: int) -> bytes:
    return deterministic_bytes(
        "blockledger.initial-block.v1",
        {
            "volume_id": volume["id"],
            "master_seed": volume["seed"],
            "block_index": block_index,
            "baseline_version": volume["baseline_version"],
            "block_size": volume["block_size"],
        },
        int(volume["block_size"]),
    )


def write_block_bytes(
    volume: dict[str, Any], event_seq: int, write_seed: int, block_index: int
) -> bytes:
    return deterministic_bytes(
        "blockledger.write-block.v1",
        {
            "volume_id": volume["id"],
            "write_seed": write_seed,
            "event_seq": event_seq,
            "block_index": block_index,
            "block_size": volume["block_size"],
        },
        int(volume["block_size"]),
    )


class ContentStore:
    """Tiny content-addressed object store.

    Snapshot objects contain digest mappings only. The CAS permits rollback to
    resolve a frozen digest to exact bytes without duplicating objects per
    snapshot.
    """

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root is not None else None
        self._memory: dict[str, bytes] = {}
        if self.root is not None:
            (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def path_for_digest(self, digest: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / "objects" / digest[:2] / f"{digest}.bin"

    def put(self, digest: str, content: bytes) -> None:
        self._memory[digest] = content
        if self.root is None:
            return
        path = self.path_for_digest(digest)
        assert path is not None
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def get(self, digest: str) -> bytes | None:
        if digest in self._memory:
            return self._memory[digest]
        path = self.path_for_digest(digest)
        if path is None or not path.exists():
            return None
        content = path.read_bytes()
        self._memory[digest] = content
        return content


class MaterializedBlocks:
    """Files representing the currently restored volume blocks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def block_path(self, volume_id: str, block_index: int) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", volume_id).strip("._") or "volume"
        if safe_id != volume_id or safe_id in {".", ".."}:
            safe_id = f"{safe_id}-{sha256_hex(volume_id)[:12]}"
        volume_dir = self.root / safe_id
        volume_dir.mkdir(parents=True, exist_ok=True)
        return volume_dir / f"{block_index:012d}.bin"

    def write(self, volume_id: str, block_index: int, content: bytes) -> Path:
        path = self.block_path(volume_id, block_index)
        fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return path

    def read(self, volume_id: str, block_index: int) -> bytes:
        return self.block_path(volume_id, block_index).read_bytes()


def digest_file(path: str | Path, volume_id: str, block_index: int, baseline_version: int) -> str:
    return content_digest(volume_id, block_index, baseline_version, Path(path).read_bytes())


def raw_hash(content: bytes) -> str:
    return sha256_hex(content)
