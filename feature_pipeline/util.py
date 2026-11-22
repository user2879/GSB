"""Shared deterministic serialization, hashing, and atomic-write helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    text = pretty_json(value) if pretty else canonical_json(value)
    atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_path(value: str) -> str:
    """Turn a stable sort key into a safe relative artifact path."""

    return "/".join(
        part.replace("%", "%25").replace("..", "%2E%2E")
        for part in value.split("/")
    )


def parallel_sorted_map(
    tasks: Sequence[tuple[str, Callable[[], T]]],
    *,
    max_workers: int,
    sort_key: Callable[[str], Any] | None = None,
) -> tuple[dict[str, T], int]:
    """Execute independent tasks concurrently and return stable sorted output.

    The returned second value is the maximum number of task bodies observed at
    once.  Callers do not depend on completion order: results are reinserted by
    the supplied task key.
    """

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not tasks:
        return {}, 0

    lock = threading.Lock()
    active = 0
    observed = 0

    def wrapped(func: Callable[[], T]) -> T:
        nonlocal active, observed
        with lock:
            active += 1
            observed = max(observed, active)
        try:
            return func()
        finally:
            with lock:
                active -= 1

    results: dict[str, T] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="feature-pipeline") as pool:
        futures = {pool.submit(wrapped, func): key for key, func in tasks}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    ordered_keys = sorted(results, key=sort_key or (lambda item: item))
    return {key: results[key] for key in ordered_keys}, observed


def stable_combinations(columns: Iterable[str], order: int) -> list[tuple[str, ...]]:
    from itertools import combinations

    return list(combinations(sorted(set(columns)), order))


def finite_float(value: Any) -> float:
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError("non-finite numeric value")
    return result


def jsonable_number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return finite_float(value)
