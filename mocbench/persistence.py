import json
import os
import tempfile
from dataclasses import dataclass

from .canonical import canonical_json, sha256_text
from .errors import BenchError

MANIFEST_NAME = "manifest.json"
INTERRUPT_NAME = "interrupt.flag"


def state_files(state_dir):
    return {
        "manifest": os.path.join(state_dir, MANIFEST_NAME),
        "interrupt": os.path.join(state_dir, INTERRUPT_NAME),
        "chunk": lambda index: os.path.join(state_dir, f"chunk_{index:06d}.json"),
        "temp": lambda index: os.path.join(state_dir, f".chunk_{index:06d}.json.tmp"),
    }


def atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".atomic_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def chunk_digest(payload):
    body = {
        "version_digest": payload["version_digest"],
        "chunk_index": payload["chunk_index"],
        "frame_start": payload["frame_start"],
        "frame_count": payload["frame_count"],
        "frames": payload["frames"],
    }
    return sha256_text(canonical_json(body))


def write_chunk(state_dir, index, frame_start, version_digest, frames):
    payload = {
        "version_digest": version_digest,
        "chunk_index": index,
        "frame_start": frame_start,
        "frame_count": len(frames),
        "frames": frames,
    }
    payload["digest"] = chunk_digest(payload)
    path = state_files(state_dir)["chunk"](index)
    atomic_write_json(path, payload)
    return payload


def write_manifest(state_dir, version_digest, frame_count, chunk_size, completed):
    payload = {
        "version_digest": version_digest,
        "frame_count": frame_count,
        "chunk_size": chunk_size,
        "completed_chunks": sorted(completed),
    }
    payload["digest"] = sha256_text(
        canonical_json({k: payload[k] for k in ("version_digest", "frame_count", "chunk_size", "completed_chunks")})
    )
    atomic_write_json(state_files(state_dir)["manifest"], payload)
    return payload


def read_manifest(state_dir):
    path = state_files(state_dir)["manifest"]
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_chunk(state_dir, index):
    path = state_files(state_dir)["chunk"](index)
    if not os.path.exists(path):
        raise BenchError("E_DIGEST", f"chunk {index} is missing", {"chunk": index})
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BenchError("E_DIGEST", f"chunk {index} is unreadable", {"chunk": index, "detail": str(exc)})
    stored = payload.get("digest")
    actual = chunk_digest(payload)
    if stored != actual:
        raise BenchError("E_DIGEST", f"chunk {index} digest mismatch", {"chunk": index})
    return payload


def invalidate_chunk(state_dir, index):
    path = state_files(state_dir)["chunk"](index)
    if os.path.exists(path):
        os.replace(path, path + ".invalid")


def mark_interrupted(state_dir):
    path = state_files(state_dir)["interrupt"]
    atomic_write_json(path, {"interrupted": True})


def interrupted(state_dir):
    return os.path.exists(state_files(state_dir)["interrupt"])


def clear_interrupt(state_dir):
    path = state_files(state_dir)["interrupt"]
    if os.path.exists(path):
        os.unlink(path)


def reset_state(state_dir):
    if not os.path.isdir(state_dir):
        return
    for name in os.listdir(state_dir):
        os.unlink(os.path.join(state_dir, name))


def verify_state(state_dir, version_digest, frame_count, chunk_size):
    manifest = read_manifest(state_dir)
    if manifest is None:
        raise BenchError("E_STATE", "state manifest not found", {"state_dir": state_dir})
    if manifest.get("version_digest") != version_digest:
        raise BenchError(
            "E_DIGEST",
            "state version digest does not match input",
            {"state": manifest.get("version_digest"), "input": version_digest},
        )
    if manifest.get("frame_count") != frame_count or manifest.get("chunk_size") != chunk_size:
        raise BenchError("E_DIGEST", "state frame layout does not match input")
    for index in manifest.get("completed_chunks", []):
        payload = read_chunk(state_dir, index)
        start = index * chunk_size
        expected_count = min(chunk_size, frame_count - start)
        if payload["frame_start"] != start or payload["frame_count"] != expected_count:
            raise BenchError("E_DIGEST", "chunk declared frame range mismatch", {"chunk": index})
        if payload["version_digest"] != version_digest:
            raise BenchError("E_DIGEST", "chunk version mismatch", {"chunk": index})
    return manifest
