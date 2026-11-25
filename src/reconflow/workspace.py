from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .specs import SpecBundle, load_bundle


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.domains_path = self.root / "specs" / "domains.json"
        self.config_path = self.root / "config" / "config.json"
        self.candidates_path = self.root / "config" / "candidates.json"
        self.batches_path = self.root / "data" / "batches.ndjson"
        self.ledger_path = self.root / "ledger" / "reconflow.sqlite"
        self.fast_view_path = self.root / "cache" / "fast_view.json"
        self.fast_conclusion_path = self.root / "cache" / "fast_conclusion.canonical.json"
        self.authoritative_path = self.root / "cache" / "authoritative_conclusion.canonical.json"
        self.propagation_path = self.root / "cache" / "last_propagation.json"
        self.snapshot_dir = self.root / "snapshots"

    def ensure_dirs(self) -> None:
        for path in (self.config_path.parent, self.batches_path.parent, self.ledger_path.parent, self.fast_view_path.parent, self.snapshot_dir):
            path.mkdir(parents=True, exist_ok=True)

    def load_bundle(self, *, verify_digests: bool = True) -> SpecBundle:
        return load_bundle(self.root, verify_digests=verify_digests)

    def load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_config(self, config: dict[str, Any]) -> None:
        self.ensure_dirs()
        atomic_write_json(self.config_path, config)

    def load_candidates(self) -> list[dict[str, Any]]:
        if not self.candidates_path.exists():
            return []
        payload = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        return list(payload.get("candidates", []))

    def save_candidates(self, candidates: list[dict[str, Any]]) -> None:
        atomic_write_json(self.candidates_path, {"format_version": 1, "candidates": candidates})

    def read_fast_view_bytes(self) -> bytes:
        return self.fast_view_path.read_bytes()

    def load_fast_view(self) -> dict[str, Any]:
        return json.loads(self.read_fast_view_bytes())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_bytes(value))


def default_candidates() -> list[dict[str, Any]]:
    domains = ["alpha", "bravo", "cyan", "delta"]
    return [
        {
            "candidate_id": "cand_a_alpha_plus",
            "weights_bp": {"alpha": 2_600, "bravo": 2_400, "cyan": 2_500, "delta": 2_500},
        },
        {
            "candidate_id": "cand_b_minimal_shift",
            "weights_bp": {"alpha": 2_540, "bravo": 2_460, "cyan": 2_500, "delta": 2_500},
        },
    ]


def write_default_candidates(root: str | Path) -> None:
    workspace = Workspace(root)
    workspace.ensure_dirs()
    workspace.save_candidates(default_candidates())
