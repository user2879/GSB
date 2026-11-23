from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from gsbcache.content import artifact_digest
from gsbcache.ledger import Ledger
from gsbcache.sample import create_sample
from gsbcache.spec import Spec, load_spec
from gsbcache.util import monotonic_ns, now_ms, write_json_file


def make_sample(root: Path) -> tuple[Path, Spec, dict[str, Any]]:
    report = create_sample(root)
    spec_path = root / "spec.json"
    return spec_path, load_spec(spec_path), report


def mutate_spec(root: Path, mutation) -> tuple[Path, dict[str, Any]]:
    source, _spec, _report = make_sample(root)
    raw = json.loads(source.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(raw)
    mutation(mutated)
    target = root / "mutated-spec.json"
    write_json_file(target, mutated)
    return target, mutated


def commit_manual_lease(
    ledger: Ledger,
    spec: Spec,
    session_id: str,
    *,
    key: str = "src:main",
    node_id: str = "node-a",
    lease_id: str = "lease-manual-1",
    request_id: str = "request-manual-1",
    arbitration_index: int = 1,
    lease_ms: int = 1000,
) -> dict[str, Any]:
    attempt_id = ledger.begin_attempt(
        request_id,
        key,
        node_id,
        owner_session=session_id,
        details={"manual_test": True},
    )
    digest = artifact_digest(spec.node_by_id[node_id], key)
    return ledger.commit_attempt(
        attempt_id,
        node_id=node_id,
        lease_id=lease_id,
        request_id=request_id,
        artifact_keys=[key],
        digests={key: digest},
        expected_digest=spec.artifacts[key].expected_digest,
        expires_at_ms=now_ms() + lease_ms,
        deadline_mono_ns=monotonic_ns() + lease_ms * 1_000_000,
        owner_session=session_id,
        arbitration_index=arbitration_index,
        details={"manual_test": True},
    )
