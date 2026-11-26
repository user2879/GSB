from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import Codes, DiagnosticError
from .jsonloc import LocatedDict, LocatedList, load_located_json


SUPPORTED_SCALES = (1, 10, 100, 1_000, 10_000, 1_000_000)


@dataclass(frozen=True)
class NodeSpec:
    id: str
    seq: int
    host: str
    port: int
    concurrency_quota: int
    lease_limit: int
    fixed_scale: int
    source: str
    line: int


@dataclass(frozen=True)
class SampleSpec:
    id: str
    group: str
    values: tuple[int, ...]
    seed: int | None = None
    source: str = ""
    line: int = 1


@dataclass(frozen=True)
class ShardSpec:
    id: str
    node_id: str
    batch_id: str
    seq_start: int
    seq_end: int
    resource_units: int
    dependencies: tuple[str, ...]
    sample_ids: tuple[str, ...]
    source: str
    line: int


@dataclass(frozen=True)
class NodeLimit:
    concurrency_quota: int
    lease_limit: int


@dataclass(frozen=True)
class CapacityVersion:
    version: int
    parent_version: int | None
    budget_units: int
    node_limits: dict[str, NodeLimit]
    source: str
    line: int


@dataclass(frozen=True)
class FixedRevision:
    revision: int
    parent_revision: int | None
    scale_factor: int
    source: str
    line: int


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    shard_id: str | None = None
    node_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    line: int = 1


@dataclass(frozen=True)
class FormatSpec:
    d1_widths: tuple[int, int, int, int] = (4, 5, 5, 16)
    d1_fields: tuple[str, ...] = ("node_seq", "seq_start", "seq_end", "summary_prefix")
    prefix_chars: int = 16

    @property
    def d1_line_width(self) -> int:
        return sum(self.d1_widths)


@dataclass
class SpecBundle:
    root: Path
    manifest: dict[str, Any]
    spec_format_version: int
    node_set_version: int
    active_capacity_version: int
    active_fixed_revision: int
    name: str
    nodes: list[NodeSpec]
    samples: list[SampleSpec]
    shards: list[ShardSpec]
    capacity_versions: list[CapacityVersion]
    fixed_revisions: list[FixedRevision]
    events: list[EventSpec]
    format_spec: FormatSpec
    known_delivery_refs: frozenset[str]
    content_digest: str

    @property
    def nodes_by_id(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def nodes_by_seq(self) -> dict[int, NodeSpec]:
        return {node.seq: node for node in self.nodes}

    @property
    def shards_by_id(self) -> dict[str, ShardSpec]:
        return {shard.id: shard for shard in self.shards}

    @property
    def samples_by_id(self) -> dict[str, SampleSpec]:
        return {sample.id: sample for sample in self.samples}

    @property
    def capacity_by_version(self) -> dict[int, CapacityVersion]:
        return {item.version: item for item in self.capacity_versions}

    @property
    def fixed_by_revision(self) -> dict[int, FixedRevision]:
        return {item.revision: item for item in self.fixed_revisions}

    def active_capacity(self) -> CapacityVersion:
        return self.capacity_by_version[self.active_capacity_version]

    def active_fixed(self) -> FixedRevision:
        return self.fixed_by_revision[self.active_fixed_revision]


def _spec_error(
    path: str | Path,
    data: Any,
    key: str | None,
    message: str,
    *,
    line: int | None = None,
) -> DiagnosticError:
    resolved_line = line
    if resolved_line is None and isinstance(data, LocatedDict) and key is not None:
        resolved_line = data.field_line(key)
    return DiagnosticError(Codes.SPEC, message, field=key, line=resolved_line or 1, source=str(path))


def _require_object(data: Any, path: Path) -> LocatedDict:
    if not isinstance(data, LocatedDict):
        raise DiagnosticError(Codes.SPEC, "top-level JSON value must be an object", field="$", line=1, source=str(path))
    return data


def _require_list(obj: LocatedDict, key: str, path: Path) -> LocatedList:
    value = obj.get(key)
    if not isinstance(value, LocatedList):
        raise _spec_error(path, obj, key, f"{key} must be a list")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_str(item: LocatedDict, key: str, path: Path) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise _spec_error(path, item, key, f"{key} must be a non-empty string")
    return value


def _require_int(item: LocatedDict, key: str, path: Path, *, minimum: int | None = None) -> int:
    value = item.get(key)
    if not _is_int(value):
        raise _spec_error(path, item, key, f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise _spec_error(path, item, key, f"{key} must be >= {minimum}")
    return value


def _optional_str_list(item: LocatedDict, key: str, path: Path) -> tuple[str, ...]:
    if key not in item:
        return ()
    value = item[key]
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise _spec_error(path, item, key, f"{key} must be a list of non-empty strings")
    return tuple(value)


def _digest_dir(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name.endswith(".pyc"):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    data, _ = load_located_json(path)
    return data


def load_spec_bundle(root: str | Path) -> SpecBundle:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest, _ = load_located_json(manifest_path)
    manifest = _require_object(manifest, manifest_path)
    spec_format_version = _require_int(manifest, "spec_format_version", manifest_path, minimum=1)
    node_set_version = _require_int(manifest, "node_set_version", manifest_path, minimum=1)
    active_capacity_version = _require_int(manifest, "active_capacity_version", manifest_path, minimum=1)
    active_fixed_revision = _require_int(manifest, "active_fixed_revision", manifest_path, minimum=1)
    name = manifest.get("name", root.name)
    if not isinstance(name, str):
        raise _spec_error(manifest_path, manifest, "name", "name must be a string")

    nodes_path = root / "nodes.json"
    nodes_data = _require_object(load_located_json(nodes_path)[0], nodes_path)
    raw_nodes = _require_list(nodes_data, "nodes", nodes_path)
    nodes: list[NodeSpec] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    seen_seqs: set[int] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, LocatedDict):
            raise DiagnosticError(Codes.SPEC, "node entry must be an object", field=f"nodes[{index}]", line=raw_nodes.item_line(index), source=str(nodes_path))
        node_id = _require_str(raw, "id", nodes_path)
        if node_id in seen_node_ids:
            raise _spec_error(nodes_path, raw, "id", f"duplicate node id: {node_id}")
        seen_node_ids.add(node_id)
        seq = _require_int(raw, "seq", nodes_path, minimum=1)
        if seq in seen_seqs:
            raise _spec_error(nodes_path, raw, "seq", f"duplicate node seq: {seq}")
        seen_seqs.add(seq)
        host = _require_str(raw, "host", nodes_path)
        if host != "127.0.0.1":
            raise _spec_error(nodes_path, raw, "host", "host must be the literal 127.0.0.1")
        port = _require_int(raw, "port", nodes_path, minimum=1)
        if port > 65535:
            raise _spec_error(nodes_path, raw, "port", "port must be <= 65535")
        if port in seen_ports:
            raise _spec_error(nodes_path, raw, "port", f"duplicate loopback port: {port}")
        seen_ports.add(port)
        quota = _require_int(raw, "concurrency_quota", nodes_path, minimum=0)
        lease_limit = _require_int(raw, "lease_limit", nodes_path, minimum=0)
        fixed_scale = _require_int(raw, "fixed_scale", nodes_path, minimum=1)
        if fixed_scale not in SUPPORTED_SCALES:
            raise _spec_error(nodes_path, raw, "fixed_scale", f"fixed_scale must be one of {SUPPORTED_SCALES}")
        nodes.append(NodeSpec(node_id, seq, host, port, quota, lease_limit, fixed_scale, str(nodes_path), raw.line))

    samples_path = root / "samples.json"
    samples: list[SampleSpec] = []
    if samples_path.exists():
        samples_data = _require_object(load_located_json(samples_path)[0], samples_path)
        raw_samples = _require_list(samples_data, "samples", samples_path)
        seen_sample_ids: set[str] = set()
        for index, raw in enumerate(raw_samples):
            if not isinstance(raw, LocatedDict):
                raise DiagnosticError(Codes.SPEC, "sample entry must be an object", field=f"samples[{index}]", line=raw_samples.item_line(index), source=str(samples_path))
            sample_id = _require_str(raw, "id", samples_path)
            if sample_id in seen_sample_ids:
                raise _spec_error(samples_path, raw, "id", f"duplicate sample id: {sample_id}")
            seen_sample_ids.add(sample_id)
            group = raw.get("group", "default")
            if not isinstance(group, str) or not group:
                raise _spec_error(samples_path, raw, "group", "group must be a non-empty string")
            values = raw.get("values")
            if not isinstance(values, list) or not values or not all(_is_int(v) for v in values):
                raise _spec_error(samples_path, raw, "values", "values must be a non-empty integer list")
            seed = raw.get("seed")
            if seed is not None and not _is_int(seed):
                raise _spec_error(samples_path, raw, "seed", "seed must be an integer")
            samples.append(SampleSpec(sample_id, group, tuple(values), seed, str(samples_path), raw.line))

    shards_path = root / "shards.json"
    shards_data = _require_object(load_located_json(shards_path)[0], shards_path)
    raw_shards = _require_list(shards_data, "shards", shards_path)
    shards: list[ShardSpec] = []
    seen_shard_ids: set[str] = set()
    node_ids = {node.id for node in nodes}
    sample_ids = {sample.id for sample in samples}
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, LocatedDict):
            raise DiagnosticError(Codes.SPEC, "shard entry must be an object", field=f"shards[{index}]", line=raw_shards.item_line(index), source=str(shards_path))
        shard_id = _require_str(raw, "id", shards_path)
        if shard_id in seen_shard_ids:
            raise _spec_error(shards_path, raw, "id", f"duplicate shard id: {shard_id}")
        seen_shard_ids.add(shard_id)
        node_id = _require_str(raw, "node_id", shards_path)
        if node_id not in node_ids:
            raise _spec_error(shards_path, raw, "node_id", f"node reference does not exist: {node_id}")
        batch_id = raw.get("batch_id", "default-batch")
        if not isinstance(batch_id, str) or not batch_id:
            raise _spec_error(shards_path, raw, "batch_id", "batch_id must be a non-empty string")
        seq_start = _require_int(raw, "seq_start", shards_path, minimum=0)
        seq_end = _require_int(raw, "seq_end", shards_path, minimum=0)
        if seq_start > seq_end:
            raise _spec_error(shards_path, raw, "seq_start", "seq_start must be <= seq_end")
        units = _require_int(raw, "resource_units", shards_path, minimum=1)
        deps = _optional_str_list(raw, "dependencies", shards_path)
        if shard_id in deps:
            raise _spec_error(shards_path, raw, "dependencies", "shard cannot depend on itself")
        refs = _optional_str_list(raw, "sample_ids", shards_path)
        if not refs:
            raise _spec_error(shards_path, raw, "sample_ids", "sample_ids must not be empty")
        missing_sample = next((ref for ref in refs if ref not in sample_ids), None)
        if missing_sample:
            raise _spec_error(shards_path, raw, "sample_ids", f"sample reference does not exist: {missing_sample}")
        shards.append(ShardSpec(shard_id, node_id, batch_id, seq_start, seq_end, units, deps, refs, str(shards_path), raw.line))

    shard_ids = {shard.id for shard in shards}
    for shard in shards:
        missing_dep = next((dep for dep in shard.dependencies if dep not in shard_ids), None)
        if missing_dep:
            raise DiagnosticError(
                Codes.SPEC,
                f"dependency reference does not exist: {missing_dep}",
                field="dependencies",
                line=shard.line,
                source=shard.source,
            )
    _assert_acyclic(shards)

    cap_path = root / "capacity_versions.json"
    cap_data = _require_object(load_located_json(cap_path)[0], cap_path)
    raw_versions = _require_list(cap_data, "versions", cap_path)
    capacities: list[CapacityVersion] = []
    seen_versions: set[int] = set()
    for index, raw in enumerate(raw_versions):
        if not isinstance(raw, LocatedDict):
            raise DiagnosticError(Codes.SPEC, "capacity version must be an object", field=f"versions[{index}]", line=raw_versions.item_line(index), source=str(cap_path))
        version = _require_int(raw, "version", cap_path, minimum=1)
        if version in seen_versions:
            raise _spec_error(cap_path, raw, "version", f"duplicate capacity version: {version}")
        seen_versions.add(version)
        parent = raw.get("parent_version")
        if parent is not None and not _is_int(parent):
            raise _spec_error(cap_path, raw, "parent_version", "parent_version must be an integer or null")
        budget = _require_int(raw, "budget_units", cap_path, minimum=0)
        raw_limits = raw.get("node_limits")
        if not isinstance(raw_limits, LocatedDict):
            raise _spec_error(cap_path, raw, "node_limits", "node_limits must be an object keyed by node id")
        limits: dict[str, NodeLimit] = {}
        for node in nodes:
            raw_limit = raw_limits.get(node.id)
            if not isinstance(raw_limit, LocatedDict):
                line = raw_limits.field_line(node.id) if isinstance(raw_limits, LocatedDict) else raw.line
                raise DiagnosticError(Codes.SPEC, f"missing node_limits entry for {node.id}", field=f"node_limits.{node.id}", line=line, source=str(cap_path))
            quota = _require_int(raw_limit, "concurrency_quota", cap_path, minimum=0)
            lease_limit = _require_int(raw_limit, "lease_limit", cap_path, minimum=0)
            limits[node.id] = NodeLimit(quota, lease_limit)
        extra = next((key for key in raw_limits if key not in node_ids), None)
        if extra:
            line = raw_limits.key_lines.get(extra, raw.line)
            raise DiagnosticError(Codes.SPEC, f"node_limits references unknown node: {extra}", field=f"node_limits.{extra}", line=line, source=str(cap_path))
        capacities.append(CapacityVersion(version, parent, budget, limits, str(cap_path), raw.line))
    cap_lines = {v.version: v.line for v in capacities}
    _validate_parent_chain(
        [v.version for v in capacities],
        {v.version: v.parent_version for v in capacities},
        "capacity version",
        cap_path,
        cap_lines,
        "version",
        "parent_version",
    )
    if active_capacity_version not in seen_versions:
        raise _spec_error(manifest_path, manifest, "active_capacity_version", "active capacity version is absent from chain")

    fixed_path = root / "fixed_revisions.json"
    fixed_data = _require_object(load_located_json(fixed_path)[0], fixed_path)
    raw_revisions = _require_list(fixed_data, "revisions", fixed_path)
    revisions: list[FixedRevision] = []
    seen_revisions: set[int] = set()
    for index, raw in enumerate(raw_revisions):
        if not isinstance(raw, LocatedDict):
            raise DiagnosticError(Codes.SPEC, "fixed revision must be an object", field=f"revisions[{index}]", line=raw_revisions.item_line(index), source=str(fixed_path))
        revision = _require_int(raw, "revision", fixed_path, minimum=1)
        if revision in seen_revisions:
            raise _spec_error(fixed_path, raw, "revision", f"duplicate fixed revision: {revision}")
        seen_revisions.add(revision)
        parent = raw.get("parent_revision")
        if parent is not None and not _is_int(parent):
            raise _spec_error(fixed_path, raw, "parent_revision", "parent_revision must be an integer or null")
        scale = _require_int(raw, "scale_factor", fixed_path, minimum=1)
        if scale not in SUPPORTED_SCALES:
            raise _spec_error(fixed_path, raw, "scale_factor", f"scale_factor must be one of {SUPPORTED_SCALES}")
        revisions.append(FixedRevision(revision, parent, scale, str(fixed_path), raw.line))
    revision_lines = {r.revision: r.line for r in revisions}
    _validate_parent_chain(
        [r.revision for r in revisions],
        {r.revision: r.parent_revision for r in revisions},
        "fixed revision",
        fixed_path,
        revision_lines,
        "revision",
        "parent_revision",
    )
    if active_fixed_revision not in seen_revisions:
        raise _spec_error(manifest_path, manifest, "active_fixed_revision", "active fixed revision is absent from chain")

    events_path = root / "event_stream.json"
    events: list[EventSpec] = []
    if events_path.exists():
        events_data = _require_object(load_located_json(events_path)[0], events_path)
        raw_events = _require_list(events_data, "events", events_path)
        last_seq = 0
        event_seqs: set[int] = set()
        for index, raw in enumerate(raw_events):
            if not isinstance(raw, LocatedDict):
                raise DiagnosticError(Codes.SPEC, "event must be an object", field=f"events[{index}]", line=raw_events.item_line(index), source=str(events_path))
            seq = _require_int(raw, "seq", events_path, minimum=1)
            if seq in event_seqs:
                raise _spec_error(events_path, raw, "seq", f"duplicate event seq: {seq}")
            if seq <= last_seq:
                raise _spec_error(events_path, raw, "seq", "event seq must be strictly increasing in stream order")
            event_seqs.add(seq)
            last_seq = seq
            event_type = _require_str(raw, "type", events_path)
            shard_id = raw.get("shard_id")
            if shard_id is not None and (not isinstance(shard_id, str) or shard_id not in shard_ids):
                raise _spec_error(events_path, raw, "shard_id", f"event shard reference does not exist: {shard_id}")
            node_id = raw.get("node_id")
            if node_id is not None and (not isinstance(node_id, str) or node_id not in node_ids):
                raise _spec_error(events_path, raw, "node_id", f"event node reference does not exist: {node_id}")
            detail = raw.get("detail", {})
            if not isinstance(detail, dict):
                raise _spec_error(events_path, raw, "detail", "detail must be an object")
            events.append(EventSpec(seq, event_type, shard_id if isinstance(shard_id, str) else None, node_id if isinstance(node_id, str) else None, dict(detail), raw.line))

    format_spec = FormatSpec()
    formats_path = root / "formats.json"
    if formats_path.exists():
        formats_data = _require_object(load_located_json(formats_path)[0], formats_path)
        d1 = formats_data.get("d1", {})
        if not isinstance(d1, LocatedDict):
            raise DiagnosticError(Codes.SPEC, "formats.d1 must be an object", field="d1", line=1, source=str(formats_path))
        widths = d1.get("widths", list(FormatSpec().d1_widths))
        if not isinstance(widths, list) or len(widths) != 4 or not all(_is_int(w) and w > 0 for w in widths):
            raise _spec_error(formats_path, d1, "widths", "d1.widths must contain four positive integers")
        prefix_chars = _require_int(d1, "prefix_chars", formats_path, minimum=1) if "prefix_chars" in d1 else 16
        if widths[3] != prefix_chars:
            raise _spec_error(formats_path, d1, "prefix_chars", "last d1 width must equal prefix_chars")
        format_spec = FormatSpec(tuple(widths), tuple(FormatSpec().d1_fields), prefix_chars)  # type: ignore[arg-type]

    refs_path = root / "delivery_refs.json"
    known_refs: frozenset[str] = frozenset()
    if refs_path.exists():
        refs_data = _require_object(load_located_json(refs_path)[0], refs_path)
        raw_refs = refs_data.get("known_refs", [])
        if not isinstance(raw_refs, list) or not all(isinstance(x, str) and x for x in raw_refs):
            raise _spec_error(refs_path, refs_data, "known_refs", "known_refs must be a list of strings")
        known_refs = frozenset(raw_refs)

    return SpecBundle(
        root=root,
        manifest=dict(manifest),
        spec_format_version=spec_format_version,
        node_set_version=node_set_version,
        active_capacity_version=active_capacity_version,
        active_fixed_revision=active_fixed_revision,
        name=name,
        nodes=nodes,
        samples=samples,
        shards=shards,
        capacity_versions=sorted(capacities, key=lambda item: item.version),
        fixed_revisions=sorted(revisions, key=lambda item: item.revision),
        events=events,
        format_spec=format_spec,
        known_delivery_refs=known_refs,
        content_digest=_digest_dir(root),
    )


def _assert_acyclic(shards: Iterable[ShardSpec]) -> None:
    by_id = {shard.id: shard for shard in shards}
    state: dict[str, int] = {}

    def visit(shard_id: str, stack: list[str]) -> None:
        status = state.get(shard_id, 0)
        if status == 1:
            cycle = " -> ".join(stack + [shard_id])
            shard = by_id[shard_id]
            raise DiagnosticError(Codes.SPEC, f"dependency graph contains a cycle: {cycle}", field="dependencies", line=shard.line, source=shard.source)
        if status == 2:
            return
        state[shard_id] = 1
        for dep in by_id[shard_id].dependencies:
            visit(dep, stack + [shard_id])
        state[shard_id] = 2

    for shard in by_id.values():
        visit(shard.id, [])


def _validate_parent_chain(
    ids: list[int],
    parents: dict[int, int | None],
    label: str,
    path: Path,
    lines: dict[int, int],
    id_field: str,
    parent_field: str,
) -> None:
    id_set = set(ids)
    previous: int | None = None
    for item_id in ids:
        if previous is not None and item_id <= previous:
            raise DiagnosticError(Codes.SPEC, f"{label} ids must be strictly increasing in declaration order", field="version", line=lines.get(item_id, 1), source=str(path))
        previous = item_id
        parent = parents[item_id]
        if parent is None:
            continue
        if parent not in id_set:
            raise DiagnosticError(Codes.SPEC, f"{label} {item_id} points to missing parent {parent}", field=parent_field, line=lines.get(item_id, 1), source=str(path))
        if parent >= item_id:
            raise DiagnosticError(Codes.SPEC, f"{label} {item_id} must be greater than parent {parent}", field=id_field, line=lines.get(item_id, 1), source=str(path))
        seen: set[int] = set()
        current: int | None = item_id
        while current is not None:
            if current in seen:
                raise DiagnosticError(Codes.SPEC, f"{label} chain has a cycle at {current}", field=parent_field, line=lines.get(current, 1), source=str(path))
            seen.add(current)
            current = parents.get(current)


def stable_topological_order(shards: Iterable[ShardSpec]) -> list[str]:
    by_id = {shard.id: shard for shard in shards}
    remaining_deps = {shard_id: set(shard.dependencies) for shard_id, shard in by_id.items()}
    ready = sorted(shard_id for shard_id, deps in remaining_deps.items() if not deps)
    order: list[str] = []
    dependents: dict[str, list[str]] = {shard_id: [] for shard_id in by_id}
    for shard in by_id.values():
        for dep in shard.dependencies:
            dependents[dep].append(shard.id)
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dependent in dependents[current]:
            remaining_deps[dependent].discard(current)
            if not remaining_deps[dependent]:
                ready.append(dependent)
                ready.sort()
    if len(order) != len(by_id):
        raise DiagnosticError(Codes.SPEC, "dependency graph contains a cycle", field="dependencies")
    return order


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec_fingerprint(bundle: SpecBundle) -> dict[str, Any]:
    return {
        "spec_format_version": bundle.spec_format_version,
        "node_set_version": bundle.node_set_version,
        "content_digest": bundle.content_digest,
        "active_capacity_version": bundle.active_capacity_version,
        "active_fixed_revision": bundle.active_fixed_revision,
    }


def dataclass_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {k: dataclass_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_dict(v) for v in value]
    return value
