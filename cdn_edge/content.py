from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import PipelineError, e_spec
from .spec import ContentEntry, Dependency, SourceUpdateSpec, Spec


@dataclass(frozen=True)
class ContentVersion:
    path: str
    etag: str
    body: bytes
    ttl_seconds: int
    dependencies: tuple[Dependency, ...]
    effective_from_ms: int
    ordinal: int


@dataclass(frozen=True)
class EffectiveContent:
    path: str
    declared_etag: str
    etag: str
    body: bytes
    ttl_seconds: int
    dependencies: tuple[Dependency, ...]
    derived: bool

    @property
    def body_sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_digest_from_versions(versions: dict[str, dict[str, Any]]) -> str:
    return sha256_text(canonical_json(versions))


class ContentRepository:
    """Time-indexed deterministic content versions.

    A request carries a virtual monotonic timestamp.  Version selection uses the
    latest update whose effective_from_ms is <= that timestamp, which makes
    outcomes independent of process scheduling and request interleaving.
    """

    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self._history: dict[str, list[ContentVersion]] = {}
        for path, entry in sorted(spec.content.items()):
            self._history[path] = [
                ContentVersion(
                    path=path,
                    etag=entry.etag,
                    body=entry.body,
                    ttl_seconds=entry.ttl_seconds,
                    dependencies=entry.dependencies,
                    effective_from_ms=0,
                    ordinal=0,
                )
            ]
        self._ordinal = {path: 1 for path in self._history}
        self._applied_updates: list[SourceUpdateSpec] = []

    @property
    def applied_updates(self) -> tuple[SourceUpdateSpec, ...]:
        return tuple(self._applied_updates)

    def _version_at(self, path: str, at_ms: int) -> ContentVersion:
        if path not in self._history:
            raise PipelineError("E_PROTOCOL", "unknown content path", path=path, field="path")
        selected = None
        for version in self._history[path]:
            if version.effective_from_ms <= at_ms:
                selected = version
            else:
                break
        if selected is None:
            selected = self._history[path][0]
        return selected

    def _current_dependencies_at(self, at_ms: int) -> dict[str, tuple[Dependency, ...]]:
        return {
            path: self._version_at(path, at_ms).dependencies
            for path in sorted(self._history)
        }

    def _assert_acyclic_at(self, at_ms: int) -> None:
        graph = self._current_dependencies_at(at_ms)
        state = {path: 0 for path in graph}
        stack: list[str] = []

        def visit(path: str) -> list[str] | None:
            state[path] = 1
            stack.append(path)
            for dep in graph[path]:
                if dep.path not in graph:
                    raise e_spec(
                        "source update references an unknown derivation path",
                        f"source_update[path={path}]",
                        "derivation.inputs.path",
                        referenced_path=dep.path,
                    )
                if state[dep.path] == 1:
                    start = stack.index(dep.path)
                    return stack[start:] + [dep.path]
                if state[dep.path] == 0:
                    found = visit(dep.path)
                    if found:
                        return found
            stack.pop()
            state[path] = 2
            return None

        for path in sorted(graph):
            if state[path] == 0:
                cycle = visit(path)
                if cycle:
                    raise PipelineError(
                        "E_CYCLE",
                        "source update would create a derivation cycle",
                        path=f"source_update[path={cycle[0]}].derivation",
                        field="inputs",
                        details={"cycle": cycle, "spec_rejected": True},
                    )

    def apply_update(self, update: SourceUpdateSpec) -> None:
        if update.path not in self._history:
            raise e_spec("source update references an unknown content path", "source_update", "path")
        previous = self._version_at(update.path, update.at_ms)
        version = ContentVersion(
            path=update.path,
            etag=update.etag if update.etag is not None else previous.etag,
            body=update.body if update.body is not None else previous.body,
            ttl_seconds=update.ttl_seconds if update.ttl_seconds is not None else previous.ttl_seconds,
            dependencies=(
                update.dependencies
                if update.dependencies is not None
                else previous.dependencies
            ),
            effective_from_ms=update.at_ms,
            ordinal=self._ordinal[update.path],
        )
        self._ordinal[update.path] += 1
        history = self._history[update.path]
        # Updates arrive in non-decreasing time order in normal orchestration.
        insert_at = len(history)
        for index, existing in enumerate(history):
            if existing.effective_from_ms > update.at_ms:
                insert_at = index
                break
        history.insert(insert_at, version)
        self._assert_acyclic_at(update.at_ms)
        self._applied_updates.append(update)

    def effective(self, path: str, at_ms: int, _memo: dict[str, EffectiveContent] | None = None) -> EffectiveContent:
        memo = _memo if _memo is not None else {}
        if path in memo:
            return memo[path]
        version = self._version_at(path, at_ms)
        if not version.dependencies:
            result = EffectiveContent(
                path=path,
                declared_etag=version.etag,
                etag=version.etag,
                body=version.body,
                ttl_seconds=version.ttl_seconds,
                dependencies=(),
                derived=False,
            )
            memo[path] = result
            return result

        child_values: list[tuple[int, EffectiveContent]] = []
        max_len = 0
        for dep in version.dependencies:
            if dep.path == path:
                raise PipelineError(
                    "E_CYCLE",
                    "self-referential derivation",
                    path=f"content[path={path}].derivation",
                    field="inputs",
                    details={"cycle": [path, path]},
                )
            child = self.effective(dep.path, at_ms, memo)
            child_values.append((dep.coefficient, child))
            max_len = max(max_len, len(child.body))
        accumulator = [0] * max_len
        for coefficient, child in child_values:
            for index, byte in enumerate(child.body):
                accumulator[index] += coefficient * byte
        body = bytes(value % 256 for value in accumulator)
        etag = f"{version.etag}~{hashlib.sha256(body).hexdigest()[:16]}"
        result = EffectiveContent(
            path=path,
            declared_etag=version.etag,
            etag=etag,
            body=body,
            ttl_seconds=version.ttl_seconds,
            dependencies=version.dependencies,
            derived=True,
        )
        memo[path] = result
        return result

    def get(self, path: str, at_ms: int) -> EffectiveContent:
        return self.effective(path, at_ms)

    def versions_at(self, at_ms: int) -> dict[str, dict[str, Any]]:
        memo: dict[str, EffectiveContent] = {}
        result: dict[str, dict[str, Any]] = {}
        for path in sorted(self._history):
            item = self.effective(path, at_ms, memo)
            result[path] = {
                "etag": item.etag,
                "body_sha256": item.body_sha256,
                "ttl_seconds": item.ttl_seconds,
                "derived": item.derived,
                "dependencies": [dep.as_dict() for dep in item.dependencies],
            }
        return result

    def snapshot(self, at_ms: int) -> dict[str, Any]:
        versions = self.versions_at(at_ms)
        return {
            "at_ms": at_ms,
            "content_digest": content_digest_from_versions(versions),
            "content_versions": versions,
        }


def initial_versions(spec: Spec, updates: Iterable[SourceUpdateSpec] = (), at_ms: int = 0) -> dict[str, Any]:
    repository = ContentRepository(spec)
    for update in updates:
        repository.apply_update(update)
    return repository.snapshot(at_ms)


def content_digest_for_spec(spec: Spec, updates: Iterable[SourceUpdateSpec] = (), at_ms: int = 0) -> str:
    return str(initial_versions(spec, updates, at_ms)["content_digest"])


def topology_from_versions(versions: dict[str, dict[str, Any]]) -> dict[str, tuple[Dependency, ...]]:
    topology: dict[str, tuple[Dependency, ...]] = {}
    for path, item in versions.items():
        topology[path] = tuple(
            Dependency(str(dep["path"]), int(dep["coefficient"]))
            for dep in item.get("dependencies", [])
        )
    return topology
