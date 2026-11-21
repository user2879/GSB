from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes, digest_bytes, digest_canonical
from .errors import ArbiterError, spec_error


FORMAT_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9._:@/-]{1,128}$")
_NODE_KEYS = {
    "node_id",
    "host",
    "port",
    "propagation_rule_version",
    "sample_ratio_bps",
    "random_seed",
}
_NODE_FILE_KEYS = {"format_version", "ruleset_version", "nodes"}
_SPAN_KEYS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "node_id",
    "timestamp_ms",
    "duration_us",
    "sampled",
}
_SPAN_REQUIRED = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "node_id",
    "timestamp_ms",
    "duration_us",
    "sampled",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_id(value: Any, *, path: str, line: int | None = None) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise spec_error(
            "identifier must match [A-Za-z0-9._:@/-]{1,128}",
            path=path,
            line=line,
        )
    return value


def _check_int(
    value: Any,
    *,
    path: str,
    line: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not _is_int(value):
        raise spec_error("value must be an integer", path=path, line=line)
    if minimum is not None and value < minimum:
        raise spec_error(f"value must be >= {minimum}", path=path, line=line)
    if maximum is not None and value > maximum:
        raise spec_error(f"value must be <= {maximum}", path=path, line=line)
    return value


@dataclass(frozen=True)
class Node:
    node_id: str
    host: str
    port: int
    propagation_rule_version: int
    sample_ratio_bps: int
    random_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "propagation_rule_version": self.propagation_rule_version,
            "sample_ratio_bps": self.sample_ratio_bps,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    node_id: str
    timestamp_ms: int
    duration_us: int
    sampled: bool
    line: int

    def as_dict(self, *, include_line: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "node_id": self.node_id,
            "timestamp_ms": self.timestamp_ms,
            "duration_us": self.duration_us,
            "sampled": self.sampled,
        }
        if include_line:
            result["line"] = self.line
        return result


@dataclass(frozen=True)
class Dataset:
    format_version: int
    ruleset_version: int
    nodes: tuple[Node, ...]
    spans: tuple[Span, ...]
    nodes_path: str
    spans_path: str
    nodes_digest: str
    spans_digest: str
    content_digest: str
    source_payload: dict[str, Any]

    @property
    def node_map(self) -> dict[str, Node]:
        return {node.node_id: node for node in self.nodes}

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(sorted({node.propagation_rule_version for node in self.nodes}))

    @property
    def version_rules(self) -> dict[int, tuple[int, int]]:
        return {
            node.propagation_rule_version: (node.sample_ratio_bps, node.random_seed)
            for node in self.nodes
        }

    def nodes_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "ruleset_version": self.ruleset_version,
            "nodes": [
                node.as_dict()
                for node in sorted(
                    self.nodes,
                    key=lambda item: (item.propagation_rule_version, item.node_id),
                )
            ],
        }

    def spans_payload(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self.spans,
            key=lambda span: (span.trace_id, span.timestamp_ms, span.span_id),
        )
        return [span.as_dict() for span in ordered]


def _read_json_object(path: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ArbiterError("E_IO", f"cannot read node declaration: {exc}", path=path) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"invalid node declaration JSON: {exc.msg}",
            path=path,
            line=exc.lineno,
        ) from exc
    if not isinstance(value, dict):
        raise spec_error("node declaration must be a JSON object", path=path)
    return value


def _validate_nodes(payload: dict[str, Any], path: str) -> tuple[int, int, tuple[Node, ...]]:
    unknown = sorted(set(payload) - _NODE_FILE_KEYS)
    if unknown:
        raise spec_error(f"unknown field: {unknown[0]}", path=f"{path}/{unknown[0]}")
    for key in ("format_version", "ruleset_version", "nodes"):
        if key not in payload:
            raise spec_error("missing required field", path=f"{path}/{key}")

    format_version = _check_int(payload["format_version"], path=f"{path}/format_version", minimum=1)
    if format_version != FORMAT_VERSION:
        raise spec_error(
            f"unsupported format_version {format_version}; expected {FORMAT_VERSION}",
            path=f"{path}/format_version",
        )
    ruleset_version = _check_int(
        payload["ruleset_version"],
        path=f"{path}/ruleset_version",
        minimum=1,
    )
    raw_nodes = payload["nodes"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise spec_error("nodes must be a non-empty array", path=f"{path}/nodes")

    nodes: list[Node] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    version_rules: dict[int, tuple[int, int]] = {}
    previous_version = 0
    for index, raw in enumerate(raw_nodes):
        item_path = f"{path}/nodes/{index}"
        if not isinstance(raw, dict):
            raise spec_error("node must be an object", path=item_path)
        unknown = sorted(set(raw) - _NODE_KEYS)
        if unknown:
            raise spec_error(f"unknown field: {unknown[0]}", path=f"{item_path}/{unknown[0]}")
        for key in (
            "node_id",
            "host",
            "port",
            "propagation_rule_version",
            "sample_ratio_bps",
            "random_seed",
        ):
            if key not in raw:
                raise spec_error("missing required field", path=f"{item_path}/{key}")

        node_id = _check_id(raw["node_id"], path=f"{item_path}/node_id")
        if node_id in seen_ids:
            raise spec_error("duplicate node_id", path=f"{item_path}/node_id")
        seen_ids.add(node_id)

        host = raw["host"]
        if host != "127.0.0.1":
            raise spec_error(
                "host must be the literal loopback address 127.0.0.1",
                path=f"{item_path}/host",
            )

        port = _check_int(
            raw["port"],
            path=f"{item_path}/port",
            minimum=1024,
            maximum=65535,
        )
        if port in seen_ports:
            raise spec_error("duplicate loopback port", path=f"{item_path}/port")
        seen_ports.add(port)

        version = _check_int(
            raw["propagation_rule_version"],
            path=f"{item_path}/propagation_rule_version",
            minimum=1,
        )
        if version < previous_version:
            raise spec_error(
                "propagation_rule_version must be non-decreasing in declaration order",
                path=f"{item_path}/propagation_rule_version",
            )
        previous_version = version

        ratio = _check_int(
            raw["sample_ratio_bps"],
            path=f"{item_path}/sample_ratio_bps",
            minimum=0,
            maximum=10_000,
        )
        seed = _check_int(raw["random_seed"], path=f"{item_path}/random_seed")
        if version in version_rules and version_rules[version] != (ratio, seed):
            raise spec_error(
                "nodes sharing a propagation_rule_version must share sample_ratio_bps and random_seed",
                path=f"{item_path}/sample_ratio_bps",
            )
        version_rules.setdefault(version, (ratio, seed))

        nodes.append(
            Node(
                node_id=node_id,
                host=host,
                port=port,
                propagation_rule_version=version,
                sample_ratio_bps=ratio,
                random_seed=seed,
            )
        )
    return format_version, ruleset_version, tuple(nodes)


def _parse_span(raw: Any, *, file_path: str, line: int, node_ids: set[str]) -> Span:
    if not isinstance(raw, dict):
        raise spec_error("span line must be a JSON object", path=file_path, line=line)
    unknown = sorted(set(raw) - _SPAN_KEYS)
    if unknown:
        raise spec_error(
            f"unknown field: {unknown[0]}",
            path=f"{file_path}/{unknown[0]}",
            line=line,
        )
    for key in _SPAN_REQUIRED:
        if key not in raw:
            raise spec_error(
                "missing required field",
                path=f"{file_path}/{key}",
                line=line,
            )

    trace_id = _check_id(raw["trace_id"], path=f"{file_path}/trace_id", line=line)
    span_id = _check_id(raw["span_id"], path=f"{file_path}/span_id", line=line)
    parent_raw = raw["parent_span_id"]
    if parent_raw in (None, ""):
        parent_span_id = None
    else:
        parent_span_id = _check_id(
            parent_raw,
            path=f"{file_path}/parent_span_id",
            line=line,
        )
    node_id = _check_id(raw["node_id"], path=f"{file_path}/node_id", line=line)
    if node_id not in node_ids:
        raise spec_error(
            "node_id is not declared in the node file",
            path=f"{file_path}/node_id",
            line=line,
        )
    timestamp_ms = _check_int(
        raw["timestamp_ms"],
        path=f"{file_path}/timestamp_ms",
        line=line,
        minimum=0,
    )
    duration_us = _check_int(
        raw["duration_us"],
        path=f"{file_path}/duration_us",
        line=line,
        minimum=0,
    )
    if not isinstance(raw["sampled"], bool):
        raise spec_error("sampled must be a JSON boolean", path=f"{file_path}/sampled", line=line)
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        node_id=node_id,
        timestamp_ms=timestamp_ms,
        duration_us=duration_us,
        sampled=raw["sampled"],
        line=line,
    )


def _earliest(errors: list[ArbiterError]) -> ArbiterError:
    return min(errors, key=lambda err: (err.line or 10**18, err.path or ""))


def _validate_trace_graph(trace_id: str, spans: list[Span], *, spans_path: str) -> None:
    by_id: dict[str, Span] = {}
    for span in spans:
        if span.span_id in by_id:
            raise spec_error(
                "duplicate span_id within trace",
                path=f"{spans_path}/{trace_id}/{span.span_id}",
                line=span.line,
                field="span_id",
            )
        by_id[span.span_id] = span

    errors: list[ArbiterError] = []
    roots: list[Span] = []
    for span in spans:
        if span.parent_span_id is None:
            roots.append(span)
        elif span.parent_span_id not in by_id:
            errors.append(
                ArbiterError(
                    "E_SPEC",
                    "parent_span_id does not exist in the same trace",
                    path=f"{spans_path}/{trace_id}/{span.span_id}/parent_span_id",
                    line=span.line,
                    trace_id=trace_id,
                    field="parent_span_id",
                )
            )
    if errors:
        raise _earliest(errors)

    state: dict[str, int] = {}
    stack: list[str] = []
    cycle_lines: list[int] = []

    def visit(span_id: str) -> None:
        status = state.get(span_id, 0)
        if status == 1:
            start = stack.index(span_id)
            cycle_lines.extend(by_id[item].line for item in stack[start:])
            return
        if status == 2:
            return
        state[span_id] = 1
        stack.append(span_id)
        parent = by_id[span_id].parent_span_id
        if parent is not None:
            visit(parent)
        stack.pop()
        state[span_id] = 2

    for span_id in sorted(by_id):
        visit(span_id)
    if cycle_lines:
        raise spec_error(
            "parent_span_id cycle detected",
            path=f"{spans_path}/{trace_id}/parent_span_id",
            line=min(cycle_lines),
            trace_id=trace_id,
            field="parent_span_id",
        )

    if len(roots) != 1:
        candidate = min(roots, key=lambda span: span.line) if roots else min(spans, key=lambda s: s.line)
        raise spec_error(
            "each trace must have exactly one root span",
            path=f"{spans_path}/{trace_id}/root_span_id",
            line=candidate.line,
            field="root_span_id",
        )

    for span in spans:
        if span.parent_span_id is None:
            continue
        parent = by_id[span.parent_span_id]
        if span.timestamp_ms < parent.timestamp_ms:
            raise spec_error(
                "timestamp_ms must be non-decreasing from parent to child",
                path=f"{spans_path}/{trace_id}/{span.span_id}/timestamp_ms",
                line=span.line,
                trace_id=trace_id,
                field="timestamp_ms",
            )


def load_dataset(
    nodes_path: str,
    spans_path: str,
    *,
    inject: str | None = None,
) -> Dataset:
    """Load and strongly validate a node declaration and NDJSON stream.

    ``inject`` is restricted to documented fault-injection modes. Normal
    parent-reference errors are E_SPEC; the explicit missing-parent injection
    exercises the E_CHAIN recovery path.
    """

    payload = _read_json_object(nodes_path)
    if inject not in (None, "corrupt-message", "parent-missing"):
        raise spec_error("unsupported injection mode for dataset loading", path=nodes_path)
    format_version, ruleset_version, nodes = _validate_nodes(payload, "")
    node_ids = {node.node_id for node in nodes}

    try:
        raw_text = Path(spans_path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ArbiterError("E_IO", f"cannot read span stream: {exc}", path=spans_path) from exc

    raw_lines = raw_text.splitlines()
    if inject == "corrupt-message":
        if not raw_lines:
            raw_lines = ["{"]
        else:
            raw_lines[0] = '{"trace_id": "corrupt"'

    spans: list[Span] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            raw_span = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise spec_error(
                f"invalid NDJSON span: {exc.msg}",
                path=spans_path,
                line=line_number,
            ) from exc
        spans.append(
            _parse_span(raw_span, file_path=spans_path, line=line_number, node_ids=node_ids)
        )

    if not spans:
        raise spec_error("span stream contains no records", path=spans_path, line=1)

    if inject == "parent-missing":
        candidates = [span for span in spans if span.parent_span_id is not None]
        target = candidates[0] if candidates else spans[0]
        replaced: list[Span] = []
        for span in spans:
            if span is target:
                replaced.append(
                    Span(
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        parent_span_id="__injected_missing_parent__",
                        node_id=span.node_id,
                        timestamp_ms=span.timestamp_ms,
                        duration_us=span.duration_us,
                        sampled=span.sampled,
                        line=span.line,
                    )
                )
            else:
                replaced.append(span)
        spans = replaced

    grouped: dict[str, list[Span]] = {}
    for span in spans:
        grouped.setdefault(span.trace_id, []).append(span)

    graph_errors: list[ArbiterError] = []
    for trace_id in sorted(grouped):
        try:
            _validate_trace_graph(trace_id, grouped[trace_id], spans_path=spans_path)
        except ArbiterError as exc:
            graph_errors.append(exc)
    if graph_errors:
        exc = _earliest(graph_errors)
        if inject == "parent-missing" and exc.field == "parent_span_id":
            exc.code = "E_CHAIN"
        raise exc

    ordered_nodes = tuple(
        sorted(nodes, key=lambda node: (node.propagation_rule_version, node.node_id))
    )
    ordered_spans = tuple(sorted(spans, key=lambda span: (span.trace_id, span.timestamp_ms, span.span_id)))
    nodes_payload = {
        "format_version": format_version,
        "ruleset_version": ruleset_version,
        "nodes": [node.as_dict() for node in ordered_nodes],
    }
    spans_payload = [span.as_dict() for span in ordered_spans]
    nodes_digest = digest_canonical(nodes_payload)
    spans_digest = digest_canonical(spans_payload)
    content_digest = digest_bytes(
        canonical_bytes(
            {
                "format_version": format_version,
                "ruleset_version": ruleset_version,
                "nodes": nodes_payload["nodes"],
                "spans": spans_payload,
            }
        )
    )
    return Dataset(
        format_version=format_version,
        ruleset_version=ruleset_version,
        nodes=ordered_nodes,
        spans=ordered_spans,
        nodes_path=str(Path(nodes_path).resolve()),
        spans_path=str(Path(spans_path).resolve()),
        nodes_digest=nodes_digest,
        spans_digest=spans_digest,
        content_digest=content_digest,
        source_payload={
            "format_version": format_version,
            "ruleset_version": ruleset_version,
            "nodes": [node.as_dict() for node in ordered_nodes],
            "spans": spans_payload,
        },
    )


def dataset_from_payload(payload: dict[str, Any]) -> Dataset:
    """Rebuild a validated dataset from a snapshot payload."""

    nodes = tuple(
        Node(
            node_id=node["node_id"],
            host=node["host"],
            port=node["port"],
            propagation_rule_version=node["propagation_rule_version"],
            sample_ratio_bps=node["sample_ratio_bps"],
            random_seed=node["random_seed"],
        )
        for node in payload["nodes"]
    )
    spans = tuple(
        Span(
            trace_id=span["trace_id"],
            span_id=span["span_id"],
            parent_span_id=span["parent_span_id"],
            node_id=span["node_id"],
            timestamp_ms=span["timestamp_ms"],
            duration_us=span["duration_us"],
            sampled=span["sampled"],
            line=index + 1,
        )
        for index, span in enumerate(payload["spans"])
    )
    nodes_payload = {
        "format_version": payload["format_version"],
        "ruleset_version": payload["ruleset_version"],
        "nodes": [
            node.as_dict()
            for node in sorted(
                nodes,
                key=lambda item: (item.propagation_rule_version, item.node_id),
            )
        ],
    }
    spans_payload = [span.as_dict() for span in spans]
    content_digest = digest_bytes(
        canonical_bytes(
            {
                "format_version": payload["format_version"],
                "ruleset_version": payload["ruleset_version"],
                "nodes": nodes_payload["nodes"],
                "spans": spans_payload,
            }
        )
    )
    return Dataset(
        format_version=payload["format_version"],
        ruleset_version=payload["ruleset_version"],
        nodes=tuple(
            sorted(nodes, key=lambda item: (item.propagation_rule_version, item.node_id))
        ),
        spans=spans,
        nodes_path="<snapshot:nodes>",
        spans_path="<snapshot:spans>",
        nodes_digest=digest_canonical(nodes_payload),
        spans_digest=digest_canonical(spans_payload),
        content_digest=content_digest,
        source_payload=payload,
    )


def relevant_rules(nodes: Iterable[Node]) -> dict[str, tuple[int, int, int]]:
    return {
        node.node_id: (
            node.propagation_rule_version,
            node.sample_ratio_bps,
            node.random_seed,
        )
        for node in nodes
    }
