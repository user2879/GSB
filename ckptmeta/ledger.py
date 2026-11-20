from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, digest_json, load_json, pretty_json_bytes, sha256_hex
from .config import InteropConfig, load_config
from .errors import fail
from .formats import decode_manifest_data, write_manifest
from .generator import build_step_core
from .view import canonical_view, view_bytes, view_digest


LEDGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LedgerEntry:
    global_step: int
    manifest_id: str
    view: dict[str, Any]
    view_digest: str
    shard_digests: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "manifest_id": self.manifest_id,
            "view": self.view,
            "view_digest": self.view_digest,
            "shards": list(self.shard_digests),
        }


@dataclass(frozen=True)
class Ledger:
    config: InteropConfig
    entries: dict[int, LedgerEntry]
    config_digest: str
    ledger_digest: str

    def entry(self, step: int) -> LedgerEntry:
        if step not in self.entries:
            fail(
                "E_INTEGRITY",
                f"ledger has no truth entry for global step {step}",
                details={"global_step": step},
            )
        return self.entries[step]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "config_digest": self.config_digest,
            "entries": [self.entries[step].to_dict() for step in sorted(self.entries)],
        }

    def save(self, path: str | Path) -> None:
        atomic_write_bytes(Path(path), pretty_json_bytes(self.to_dict()))


def build_ledger(config: InteropConfig) -> Ledger:
    entries: dict[int, LedgerEntry] = {}
    previous_view: dict[str, Any] | None = None
    previous_digest: str | None = None
    for step in sorted(config.steps):
        core = build_step_core(config, step)
        full_view = dict(core)
        full_view["parent_manifest"] = (
            None
            if previous_view is None
            else {
                "manifest_id": previous_view["manifest_id"],
                "digest": previous_digest,
            }
        )
        canonical = canonical_view(full_view)
        digest = view_digest(canonical)
        entry = LedgerEntry(
            global_step=step,
            manifest_id=canonical["manifest_id"],
            view=canonical,
            view_digest=digest,
            shard_digests=tuple(
                {
                    "shard_index": shard["shard_index"],
                    "parameter_digest": shard["parameter_digest"],
                    "momentum_digest": shard["momentum_digest"],
                }
                for shard in canonical["shards"]
            ),
        )
        entries[step] = entry
        previous_view = canonical
        previous_digest = digest

    config_digest = digest_json(config.to_dict())
    provisional = Ledger(
        config=config,
        entries=entries,
        config_digest=config_digest,
        ledger_digest="",
    )
    ledger_digest = digest_json(provisional.to_dict())
    return Ledger(
        config=config,
        entries=entries,
        config_digest=config_digest,
        ledger_digest=ledger_digest,
    )


def ledger_from_dict(raw: dict[str, Any]) -> Ledger:
    if raw.get("schema_version") != LEDGER_SCHEMA_VERSION:
        fail("E_FORMAT", "unsupported ledger schema_version", path="schema_version")
    config = load_config(raw.get("config"))
    rebuilt = build_ledger(config)
    expected_config_digest = digest_json(config.to_dict())
    if raw.get("config_digest") != expected_config_digest:
        fail(
            "E_INTEGRITY",
            "ledger config_digest does not match embedded config",
            path="config_digest",
        )
    if raw.get("entries") != [rebuilt.entries[step].to_dict() for step in sorted(rebuilt.entries)]:
        fail(
            "E_INTEGRITY",
            "ledger entries do not reproduce deterministically from embedded config",
            path="entries",
        )
    stored_digest = digest_json(raw)
    rebuilt_digest = digest_json(rebuilt.to_dict())
    if stored_digest != rebuilt_digest:
        fail("E_INTEGRITY", "ledger digest mismatch", path="$")
    return rebuilt


def load_ledger(path: str | Path) -> Ledger:
    path = Path(path)
    raw = load_json(path)
    if not isinstance(raw, dict):
        fail("E_FORMAT", "ledger root must be a JSON object", path="$", manifest_path=str(path))
    return ledger_from_dict(raw)


def ledger_file_digest(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())


def verify_manifest_against_ledger(manifest: Any, ledger: Ledger, *, manifest_path: str | None = None) -> int:
    truth = ledger.entry(int(manifest.global_step))
    expected = truth.view
    verified = 0
    if manifest.shard_count != expected["shard_count"]:
        fail(
            "E_INTEGRITY",
            "manifest shard_count differs from ledger",
            path="shard_count",
            manifest_path=manifest_path,
            details={"field": "shard_count", "expected": expected["shard_count"], "actual": manifest.shard_count},
        )
    for shard in manifest.shards:
        expected_shard = expected["shards"][shard.index]
        if shard.parameter_digest != expected_shard["parameter_digest"]:
            fail(
                "E_INTEGRITY",
                f"parameter digest mismatch for shard {shard.index}",
                path=f"shards[{shard.index}].parameter_digest",
                manifest_path=manifest_path,
                shard_index=shard.index,
                details={
                    "field": "parameter_digest",
                    "global_step": manifest.global_step,
                    "expected": expected_shard["parameter_digest"],
                    "actual": shard.parameter_digest,
                },
            )
        verified += 1
        if shard.momentum_digest is not None:
            if shard.momentum_digest != expected_shard["momentum_digest"]:
                fail(
                    "E_INTEGRITY",
                    f"momentum digest mismatch for shard {shard.index}",
                    path=f"shards[{shard.index}].momentum_digest",
                    manifest_path=manifest_path,
                    shard_index=shard.index,
                    details={
                        "field": "momentum_digest",
                        "global_step": manifest.global_step,
                        "expected": expected_shard["momentum_digest"],
                        "actual": shard.momentum_digest,
                    },
                )
            verified += 1
        if shard.data_cursor is not None and shard.data_cursor != expected_shard["data_cursor"]:
            fail(
                "E_INTEGRITY",
                f"data cursor mismatch for shard {shard.index}",
                path=f"shards[{shard.index}].data_cursor",
                manifest_path=manifest_path,
                shard_index=shard.index,
                details={"field": "data_cursor", "global_step": manifest.global_step},
            )
    if manifest.rng_state is not None and manifest.rng_state != expected["rng_state"]:
        fail(
            "E_INTEGRITY",
            "random state differs from ledger",
            path="rng_state",
            manifest_path=manifest_path,
            details={"field": "rng_state", "global_step": manifest.global_step},
        )
    if manifest.partition_mode is not None and manifest.partition_mode != expected["partition_mode"]:
        fail(
            "E_INTEGRITY",
            "partition_mode differs from ledger",
            path="partition_mode",
            manifest_path=manifest_path,
            details={"field": "partition_mode", "global_step": manifest.global_step},
        )
    if manifest.parent_manifest is not None and manifest.parent_manifest != expected["parent_manifest"]:
        fail(
            "E_INTEGRITY",
            "parent_manifest differs from ledger",
            path="parent_manifest",
            manifest_path=manifest_path,
            details={"field": "parent_manifest", "global_step": manifest.global_step},
        )
    return verified


def direct_manifest_for_step(
    ledger: Ledger,
    step: int,
    *,
    source_path: str | Path | None = None,
) -> Any:
    truth = ledger.entry(step)
    return decode_manifest_data(
        {"format_version": 3, **truth.view},
        source_path=Path(source_path) if source_path is not None else None,
    )


def write_direct_manifest(path: str | Path, ledger: Ledger, step: int) -> Any:
    destination = Path(path)
    manifest = direct_manifest_for_step(ledger, step, source_path=destination)
    write_manifest(destination, manifest)
    return manifest
