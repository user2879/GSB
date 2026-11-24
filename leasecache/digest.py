from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .spec import ArtifactSpec, NodeSpec, PipelineSpec
from .util import atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_hex


DIGEST_RULE = (
    "sha256(canonical-json({format,seed,key,vector})); "
    "vector=PCG64(SeedSequence([seed, sha256(key)[:8]])).integers(0,2^32,16,uint32)"
)
ARTIFACT_FORMAT = "leasecache-artifact/v1"


@dataclass(frozen=True)
class GeneratedArtifact:
    key: str
    digest: str
    content: bytes


def generate_artifact(node: NodeSpec, key: str) -> GeneratedArtifact:
    """Generate a node-local artifact copy using an explicit stable rule.

    NumPy is deliberately used for the byte vector so that the rule is easy to
    reproduce outside SQL/Python dictionary ordering.  NumPy's SeedSequence and
    PCG64 algorithms are stable for the pinned major version.
    """

    key_word = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    sequence = np.random.SeedSequence([int(node.random_seed), key_word])
    generator = np.random.Generator(np.random.PCG64(sequence))
    vector = generator.integers(0, 2**32, size=16, dtype=np.uint32)
    payload = {
        "format": ARTIFACT_FORMAT,
        "key": key,
        "seed": int(node.random_seed),
        "vector": [int(value) for value in vector.tolist()],
    }
    content = canonical_json_bytes(payload)
    return GeneratedArtifact(key=key, digest=sha256_hex(content), content=content)


def digest_file(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())


class ArtifactStore:
    """Workspace-backed artifact copies and node manifests."""

    def __init__(self, workspace: str | Path, node: NodeSpec) -> None:
        self.workspace = Path(workspace).resolve()
        self.node = node
        self.node_dir = self.workspace / "node_cache" / node.node_id
        self.node_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.node_dir / "manifest.json"

    def path_for_key(self, key: str) -> Path:
        name = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] + ".artifact"
        return self.node_dir / name

    def materialize(
        self, artifacts: Iterable[ArtifactSpec]
    ) -> tuple[dict[str, str], dict[str, list[str]], list[dict[str, object]]]:
        ordered = sorted(artifacts, key=lambda artifact: artifact.key)
        digests: dict[str, str] = {}
        refs: dict[str, list[str]] = {}
        manifest_artifacts: list[dict[str, object]] = []
        for artifact in ordered:
            generated = generate_artifact(self.node, artifact.key)
            path = self.path_for_key(artifact.key)
            # Always rewrite from the deterministic rule.  A stale/tampered local
            # file therefore cannot masquerade as a restarted node's copy.
            atomic_write_bytes(path, generated.content)
            verified_digest = digest_file(path)
            if verified_digest != generated.digest:
                raise RuntimeError(f"artifact write verification failed for {artifact.key}")
            digests[artifact.key] = generated.digest
            refs[artifact.key] = list(artifact.references)
            manifest_artifacts.append(
                {
                    "key": artifact.key,
                    "digest": generated.digest,
                    "file": path.relative_to(self.workspace).as_posix(),
                    "references": list(artifact.references),
                }
            )
        manifest = {
            "format_version": 1,
            "node_id": self.node.node_id,
            "random_seed": self.node.random_seed,
            "digest_rule": DIGEST_RULE,
            "artifacts": manifest_artifacts,
        }
        atomic_write_json(self.manifest_path, manifest)
        return digests, refs, manifest_artifacts


def render_artifact_preview(content: bytes, limit: int = 180) -> str:
    try:
        text = json.dumps(json.loads(content), ensure_ascii=False, sort_keys=True)
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = content.hex()
    return text if len(text) <= limit else text[: limit - 3] + "..."
