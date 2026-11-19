"""Deterministic local sample generation for demonstrations and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    canonical_json_bytes,
    compute_content_digest,
    compute_snapshot_fingerprint,
    encode_content,
    sha256_hex,
)


def make_entry(
    path: str,
    content: bytes | str,
    *,
    algorithm: str = "sha256",
    hmac_key: bytes | None = None,
    tampered: bool = False,
    digest: str | None = None,
    length: int | None = None,
) -> dict[str, Any]:
    if isinstance(content, str):
        raw_content = content.encode("utf-8")
    else:
        raw_content = content
    actual_digest = compute_content_digest(algorithm, raw_content, hmac_key).hex()
    entry: dict[str, Any] = {
        "content_b64": encode_content(raw_content),
        "digest": digest if digest is not None else f"{algorithm}:{actual_digest}",
        "length": len(raw_content) if length is None else length,
        "path": path,
        "tampered": tampered,
    }
    return entry


def make_snapshot(
    version: int,
    entries: list[dict[str, Any]],
    *,
    previous_fingerprint: str | None = None,
    previous_version: int | None = None,
    algorithm: str = "sha256",
    hmac_key: bytes | None = None,
    schema_version: int = 1,
) -> tuple[dict[str, Any], str]:
    snapshot: dict[str, Any] = {
        "algorithm": algorithm,
        "entries": entries,
        "previous_fingerprint": previous_fingerprint,
        "previous_version": previous_version,
        "schema_version": schema_version,
        "version": version,
    }
    fingerprint = compute_snapshot_fingerprint(snapshot, algorithm, hmac_key)
    return snapshot, fingerprint


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def generate_samples(output: str | Path) -> dict[str, Any]:
    """Generate deterministic valid and adversarial sample sets."""

    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def write(relative: str, value: Any) -> None:
        path = root / relative
        _write(path, value)
        files.append(relative)

    # A clean three-version incremental chain.
    v1, fp1 = make_snapshot(
        1,
        [
            make_entry("README.txt", "backup sample\n"),
            make_entry("data/a.txt", "alpha-1\n"),
            make_entry("data/nested/keep.txt", "keep\n"),
        ],
    )
    v2, fp2 = make_snapshot(
        2,
        [
            make_entry("data/a.txt", "alpha-2\n"),
            make_entry("data/b.txt", "beta-1\n"),
        ],
        previous_fingerprint=fp1,
        previous_version=1,
    )
    v3, fp3 = make_snapshot(
        3,
        [
            make_entry("README.txt", "backup sample v3\n"),
            make_entry("data/c.txt", "gamma-1\n"),
        ],
        previous_fingerprint=fp2,
        previous_version=2,
    )
    for name, value in (("01.json", v1), ("02.json", v2), ("03.json", v3)):
        write(f"valid/{name}", value)

    # A branch from version 1, used to test rollback after watermark 2/3.
    alt2, _ = make_snapshot(
        2,
        [
            make_entry("data/a.txt", "alpha-alt\n"),
            make_entry("data/alt.txt", "alternate\n"),
        ],
        previous_fingerprint=fp1,
        previous_version=1,
    )
    write("valid/02-alt.json", alt2)

    # Two distinct successors of the same predecessor.
    fork_b, fork_b_fp = make_snapshot(
        2,
        [make_entry("data/a.txt", "alpha-fork-b\n")],
        previous_fingerprint=fp1,
        previous_version=1,
    )
    for name, value in (("01.json", v1), ("02-a.json", v2), ("02-b.json", fork_b)):
        write(f"fork/{name}", value)

    # Tampering: marker, digest mismatch, length mismatch, plus one valid entry.
    bad_digest_entry = make_entry("data/a.txt", "alpha-tampered\n")
    bad_digest_entry["digest"] = "sha256:" + ("0" * 64)
    bad_length_entry = make_entry("data/b.txt", "beta-short\n")
    bad_length_entry["length"] = 999
    tampered2, _ = make_snapshot(
        2,
        [
            make_entry("README.txt", "marked tamper\n", tampered=True),
            bad_digest_entry,
            bad_length_entry,
            make_entry("data/valid.txt", "still applied\n"),
        ],
        previous_fingerprint=fp1,
        previous_version=1,
    )
    write("tamper/01.json", v1)
    write("tamper/02.json", tampered2)

    # Missing predecessor version: chain break.
    broken3, _ = make_snapshot(
        3,
        [make_entry("data/broken.txt", "broken\n")],
        previous_fingerprint=fp1,
        previous_version=2,
    )
    write("chain_break/01.json", v1)
    write("chain_break/03.json", broken3)

    # Known predecessor version but a wrong declared fingerprint.
    mismatch2, _ = make_snapshot(
        2,
        [make_entry("data/mismatch.txt", "mismatch\n")],
        previous_fingerprint="0" * 64,
        previous_version=1,
    )
    write("fingerprint_mismatch/01.json", v1)
    write("fingerprint_mismatch/02.json", mismatch2)

    # Two snapshots that point at each other's content fingerprints.
    cycle_a, cycle_a_fp = make_snapshot(
        1, [make_entry("cycle/a.txt", "cycle-a\n")]
    )
    cycle_b, cycle_b_fp = make_snapshot(
        2,
        [make_entry("cycle/b.txt", "cycle-b\n")],
        previous_fingerprint=cycle_a_fp,
        previous_version=1,
    )
    cycle_a["previous_fingerprint"] = cycle_b_fp
    cycle_a["previous_version"] = 2
    write("cycle/a.json", cycle_a)
    write("cycle/b.json", cycle_b)

    # A self-referencing snapshot. The fingerprint excludes chain links, so the
    # link can be set after computing it without changing the fingerprint.
    self_ref, self_ref_fp = make_snapshot(
        1, [make_entry("self/loop.txt", "self\n")]
    )
    self_ref["previous_fingerprint"] = self_ref_fp
    self_ref["previous_version"] = 1
    write("self_reference/a.json", self_ref)

    manifest_files = []
    for relative in sorted(files):
        data = (root / relative).read_bytes()
        manifest_files.append({"path": relative, "sha256": sha256_hex(data)})
    manifest = {
        "description": "deterministic offline backup engine samples",
        "files": manifest_files,
        "schema_version": 1,
    }
    write("manifest.json", manifest)
    return {
        "ok": True,
        "output": str(root),
        "files": sorted(files),
        "manifest": manifest,
    }
