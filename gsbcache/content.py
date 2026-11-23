"""The explicitly documented deterministic artifact content rule."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import ReplyError
from .spec import ArtifactSpec, NodeSpec, Spec
from .util import atomic_write, canonical_bytes, sha256_bytes

CONTENT_VERSION = 1


def artifact_content(node: NodeSpec, artifact_key: str) -> bytes:
    """Canonical persisted content.

    The rule is intentionally simple and has no platform-dependent fields:
    SHA-256 over canonical JSON:
      {"artifact_key": key, "content_version": 1, "node_seed": seed}
    Node identity is deliberately excluded, so two nodes with the same seed
    produce byte-identical replicas and can compete for the same lease.
    """
    return canonical_bytes(
        {
            "artifact_key": artifact_key,
            "content_version": CONTENT_VERSION,
            "node_seed": node.seed,
        }
    )


def artifact_digest(node: NodeSpec, artifact_key: str) -> str:
    return sha256_bytes(artifact_content(node, artifact_key))


def artifact_filename(artifact_key: str) -> str:
    return "artifact-" + hashlib.sha256(artifact_key.encode("utf-8")).hexdigest() + ".bin"


@dataclass
class NodeStore:
    """Persistent per-node artifact store.

    Existing bytes are always authoritative.  Missing files are generated and
    written atomically.  A restart therefore re-hashes the bytes left by the
    previous process instead of silently regenerating them.
    """

    root: Path
    spec: Spec
    node: NodeSpec

    @property
    def artifact_dir(self) -> Path:
        return self.root / "artifacts"

    def path_for(self, artifact_key: str) -> Path:
        return self.artifact_dir / artifact_filename(artifact_key)

    def ensure(self, artifact_key: str) -> bytes:
        if artifact_key not in self.spec.artifacts:
            raise ReplyError(
                "unknown artifact key",
                artifact_key=artifact_key,
                node_id=self.node.node_id,
                field="artifact_key",
            )
        path = self.path_for(artifact_key)
        if path.exists():
            return path.read_bytes()
        data = artifact_content(self.node, artifact_key)
        atomic_write(path, data)
        return data

    def digest(self, artifact_key: str) -> str:
        return sha256_bytes(self.ensure(artifact_key))

    def persisted_digest(self, artifact_key: str) -> str | None:
        path = self.path_for(artifact_key)
        if not path.exists():
            return None
        return sha256_bytes(path.read_bytes())

    def inventory(self) -> dict[str, str]:
        return {key: self.digest(key) for key in sorted(self.spec.artifacts)}

    def reusability(self, artifact_key: str, digests: dict[str, str] | None = None) -> dict:
        """Evaluate digest plus the complete reference closure."""
        artifact = self.spec.artifacts[artifact_key]
        digests = digests if digests is not None else self.inventory()
        actual = digests.get(artifact_key, self.digest(artifact_key))
        if actual != artifact.expected_digest:
            return {
                "reusable": False,
                "artifact_key": artifact_key,
                "node_id": self.node.node_id,
                "expected_digest": artifact.expected_digest,
                "actual_digest": actual,
                "reason": "digest_mismatch",
                "bad_references": [],
            }
        bad: list[str] = []
        for ref in artifact.refs:
            ref_artifact = self.spec.artifacts[ref]
            ref_actual = digests.get(ref, self.digest(ref))
            if ref_actual != ref_artifact.expected_digest:
                bad.append(ref)
        if bad:
            return {
                "reusable": False,
                "artifact_key": artifact_key,
                "node_id": self.node.node_id,
                "expected_digest": artifact.expected_digest,
                "actual_digest": actual,
                "reason": "reference_digest_mismatch",
                "bad_references": sorted(bad),
            }
        return {
            "reusable": True,
            "artifact_key": artifact_key,
            "node_id": self.node.node_id,
            "expected_digest": artifact.expected_digest,
            "actual_digest": actual,
            "reason": "digest_and_references_match",
            "bad_references": [],
        }

    def reusable_keys(self, keys: Iterable[str], digests: dict[str, str] | None = None) -> list[str]:
        digests = digests if digests is not None else self.inventory()
        return [key for key in keys if self.reusability(key, digests)["reusable"]]


def spec_published_files(spec: Spec, output_dir: str | Path) -> list[dict]:
    """Materialize the manifest-declared replica bytes for inspection/tests."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for artifact in spec.ordered_artifacts:
        nodes = [node for node in spec.nodes if artifact_digest(node, artifact.key) == artifact.expected_digest]
        if not nodes:
            continue
        data = artifact_content(nodes[0], artifact.key)
        path = output / artifact_filename(artifact.key)
        atomic_write(path, data)
        rows.append(
            {
                "artifact_key": artifact.key,
                "expected_digest": artifact.expected_digest,
                "path": str(path),
                "matching_nodes": [node.node_id for node in nodes],
            }
        )
    return rows
