"""One-pass bucket expansion, deterministic packing and mergeable statistics."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .canonical import digest, dumps
from .errors import E_DIVERGE, E_WIDTH, fail
from .graph import DependencyGraph, build_graph, stable_edges
from .numeric import bincount, error_bound, quantize, sum_counts
from .scheduler import ScheduleEntry, ScheduleResult, schedule_packs
from .source import SourceData, load_source
from .spec import PackingSpec, RuleSpec, Spec

NULL_WIDTH_POLICY = (
    "Every expanded bucket cell reserves one presence/null bit plus its fixed "
    "bucket-id width. A null cell still consumes the full slot width, so row "
    "width does not depend on values or null positions."
)
BUCKET_RULE = (
    "q = trunc_toward_zero(value * 10^scale), saturated to signed int64; "
    "bucket = bisect_right(boundaries, q); bucket 0 is (-inf,b0), middle buckets "
    "are [b[i-1],b[i]), and the final bucket is [b[-1],+inf)."
)


def _fraction_dict(value: Fraction) -> dict[str, Any]:
    return {
        "fraction": str(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


@dataclass(frozen=True)
class ExpandedBucket:
    bucket_id: str
    source_id: str
    input_kind: str
    input_name: str
    boundaries: tuple[int, ...]
    scale: int
    bucket_count: int
    width_bits: int
    cell_width: int
    values: tuple[int, ...]
    nulls: tuple[bool, ...]
    counts: tuple[int, ...]
    null_count: int
    observed_max_error: Fraction
    error_upper_bound: Fraction
    saturated_count: int
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "source_id": self.source_id,
            "input": {"kind": self.input_kind, "name": self.input_name},
            "boundaries": list(self.boundaries),
            "scale": self.scale,
            "bucket_count": self.bucket_count,
            "width_bits": self.width_bits,
            "cell_width": self.cell_width,
            "values": list(self.values),
            "null_bitmap": "".join("1" if bit else "0" for bit in self.nulls),
            "counts": list(self.counts),
            "null_count": self.null_count,
            "quantization": {
                "mode": "truncate_toward_zero",
                "overflow": "saturate_int64",
                "observed_max_error": _fraction_dict(self.observed_max_error),
                "error_upper_bound": _fraction_dict(self.error_upper_bound),
                "saturated_values": self.saturated_count,
            },
            "bucket_version": self.version,
        }


@dataclass(frozen=True)
class ExpandedSource:
    source_id: str
    source_digest: str
    rows: int
    row_keys: tuple[str, ...]
    row_width: int
    buckets: Mapping[str, ExpandedBucket]
    topo_order: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        input_summary: dict[str, list[dict[str, Any]]] = {}
        for bucket_id in sorted(self.buckets):
            bucket = self.buckets[bucket_id]
            input_summary.setdefault(bucket.input_name, []).append(
                {
                    "bucket_column": bucket.bucket_id,
                    "input_kind": bucket.input_kind,
                    "bucket_count": bucket.bucket_count,
                    "boundaries": list(bucket.boundaries),
                    "error_upper_bound": _fraction_dict(bucket.error_upper_bound),
                }
            )
        return {
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "rows": self.rows,
            "row_keys": list(self.row_keys),
            "row_width": self.row_width,
            "bucket_columns": {
                bucket_id: self.buckets[bucket_id].to_dict() for bucket_id in sorted(self.buckets)
            },
            "input_column_summary": {
                key: input_summary[key] for key in sorted(input_summary)
            },
            "topo_order": list(self.topo_order),
            "expansion_digest": self.digest,
        }


@dataclass(frozen=True)
class PackPlan:
    pack_id: str
    source_id: str
    index: int
    row_start: int
    row_end: int
    width_used: int


@dataclass(frozen=True)
class Pack:
    pack_id: str
    source_id: str
    index: int
    row_start: int
    row_end: int
    row_keys: tuple[str, ...]
    width_used: int
    cost_micros: int
    bucket_counts: Mapping[str, tuple[int, ...]]
    null_counts: Mapping[str, int]
    bucket_versions: Mapping[str, str]
    content_digest: str
    digest: str

    @property
    def row_count(self) -> int:
        return self.row_end - self.row_start

    def summary_core(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "source_id": self.source_id,
            "index": self.index,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "row_count": self.row_count,
            "width_used": self.width_used,
            "cost_micros": self.cost_micros,
            "bucket_counts": {
                bucket_id: list(self.bucket_counts[bucket_id])
                for bucket_id in sorted(self.bucket_counts)
            },
            "null_counts": {
                bucket_id: self.null_counts[bucket_id] for bucket_id in sorted(self.null_counts)
            },
            "bucket_versions": {
                bucket_id: self.bucket_versions[bucket_id]
                for bucket_id in sorted(self.bucket_versions)
            },
            "content_digest": self.content_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.summary_core()
        out["row_keys"] = list(self.row_keys)
        out["pack_digest"] = self.digest
        return out


@dataclass(frozen=True)
class PipelineState:
    spec: Spec
    graph: DependencyGraph
    source_data: Mapping[str, SourceData]
    expansions: Mapping[str, ExpandedSource]
    packs: tuple[Pack, ...]
    schedule: ScheduleResult
    result: Mapping[str, Any]


def _bucket_version(
    rule: RuleSpec,
    values: Sequence[int],
    nulls: Sequence[bool],
) -> str:
    return digest(
        {
            "bucket_id": rule.id,
            "source_id": rule.source,
            "input": {"kind": rule.input_kind, "name": rule.input_name},
            "boundaries": list(rule.boundaries),
            "scale": rule.scale,
            "width_bits": rule.effective_width_bits,
            "values": list(values),
            "nulls": "".join("1" if bit else "0" for bit in nulls),
        }
    )


def _slice_bucket_version(bucket: ExpandedBucket, row_start: int, row_end: int) -> str:
    """Version only the rows owned by a pack.

    A pack must remain byte-identical when a different pack of the same bucket
    column changes.  Therefore pack summaries carry pack-local slice versions;
    the whole-column version remains in the source expansion.
    """

    return digest(
        {
            "bucket_id": bucket.bucket_id,
            "source_id": bucket.source_id,
            "input": {"kind": bucket.input_kind, "name": bucket.input_name},
            "boundaries": list(bucket.boundaries),
            "scale": bucket.scale,
            "width_bits": bucket.width_bits,
            "values": list(bucket.values[row_start:row_end]),
            "nulls": "".join(
                "1" if bit else "0" for bit in bucket.nulls[row_start:row_end]
            ),
        }
    )


def expand_source(
    spec: Spec,
    graph: DependencyGraph,
    data: SourceData,
    *,
    baseline: ExpandedSource | None = None,
    invalidated: Iterable[str] = (),
    counters: dict[str, list[str]] | None = None,
) -> ExpandedSource:
    """Expand all bucket rules for one source in a single row traversal."""

    source = spec.source(data.source_id)
    assert source is not None
    invalidated_set = set(invalidated)
    source_rule_ids = tuple(
        rule_id for rule_id in graph.topo if spec.rule(rule_id).source == source.id  # type: ignore[union-attr]
    )
    source_rules = tuple(spec.rule(rule_id) for rule_id in source_rule_ids)
    assert all(rule is not None for rule in source_rules)
    rules: tuple[RuleSpec, ...] = tuple(rule for rule in source_rules if rule is not None)
    if counters is None:
        counters = {"recomputed": [], "reused": []}

    reuse: dict[str, ExpandedBucket] = {}
    if baseline is not None:
        for rule in rules:
            old = baseline.buckets.get(rule.id)
            if old is None or rule.id in invalidated_set:
                continue
            same_config = (
                old.boundaries == rule.boundaries
                and old.scale == rule.scale
                and old.width_bits == rule.effective_width_bits
                and old.input_kind == rule.input_kind
                and old.input_name == rule.input_name
                and len(old.values) == data.rows
            )
            if same_config:
                reuse[rule.id] = old

    value_arrays: dict[str, list[int] | tuple[int, ...]] = {}
    null_arrays: dict[str, list[bool] | tuple[bool, ...]] = {}
    count_arrays: dict[str, list[int]] = {}
    null_counts: dict[str, int] = {}
    observed: dict[str, Fraction] = {}
    saturated: dict[str, int] = {}
    recompute_ids: list[str] = []

    for rule in rules:
        if rule.id in reuse:
            old = reuse[rule.id]
            value_arrays[rule.id] = old.values
            null_arrays[rule.id] = old.nulls
            count_arrays[rule.id] = list(old.counts)
            null_counts[rule.id] = old.null_count
            observed[rule.id] = old.observed_max_error
            saturated[rule.id] = old.saturated_count
            counters["reused"].append(rule.id)
        else:
            value_arrays[rule.id] = [0] * data.rows
            null_arrays[rule.id] = [False] * data.rows
            count_arrays[rule.id] = [0] * rule.bucket_count
            null_counts[rule.id] = 0
            observed[rule.id] = Fraction(0)
            saturated[rule.id] = 0
            recompute_ids.append(rule.id)
            counters["recomputed"].append(rule.id)

    for row in range(data.rows):
        for rule in rules:
            if rule.id in reuse:
                continue
            if rule.input_kind == "column":
                column = source.column(rule.input_name)
                assert column is not None
                is_null = data.null_flags[rule.input_name][row]
                raw_value = column.null_fill if is_null else data.raw_rows[row][rule.input_name]
            else:
                upstream = value_arrays[rule.input_name]
                upstream_nulls = null_arrays[rule.input_name]
                raw_value = upstream[row]
                is_null = bool(upstream_nulls[row])
            assert raw_value is not None
            quantized = quantize(raw_value, rule.scale)
            bucket = bisect_right(rule.boundaries, quantized.value)
            bucket_values = value_arrays[rule.id]
            bucket_nulls = null_arrays[rule.id]
            assert isinstance(bucket_values, list)
            assert isinstance(bucket_nulls, list)
            bucket_values[row] = bucket
            bucket_nulls[row] = bool(is_null)
            count_arrays[rule.id][bucket] += 1
            if is_null:
                null_counts[rule.id] += 1
            observed[rule.id] = max(observed[rule.id], quantized.error)
            if quantized.saturated:
                saturated[rule.id] += 1

    buckets: dict[str, ExpandedBucket] = {}
    for rule in rules:
        old = reuse.get(rule.id)
        if old is not None:
            buckets[rule.id] = old
            continue
        values = tuple(value_arrays[rule.id])
        nulls = tuple(bool(bit) for bit in null_arrays[rule.id])
        is_float_input = False
        if rule.input_kind == "column":
            column = source.column(rule.input_name)
            assert column is not None
            is_float_input = column.type == "float"
        if is_float_input:
            bound = error_bound(rule.scale, observed[rule.id], saturated[rule.id] > 0)
        else:
            bound = Fraction(0)
        buckets[rule.id] = ExpandedBucket(
            bucket_id=rule.id,
            source_id=rule.source,
            input_kind=rule.input_kind,
            input_name=rule.input_name,
            boundaries=rule.boundaries,
            scale=rule.scale,
            bucket_count=rule.bucket_count,
            width_bits=rule.effective_width_bits,
            cell_width=rule.cell_width,
            values=values,
            nulls=nulls,
            counts=tuple(count_arrays[rule.id]),
            null_count=null_counts[rule.id],
            observed_max_error=observed[rule.id],
            error_upper_bound=bound,
            saturated_count=saturated[rule.id],
            version=_bucket_version(rule, values, nulls),
        )

    row_width = sum(rule.cell_width for rule in rules)
    expansion_digest = digest(
        {
            "source_id": source.id,
            "source_digest": data.digest,
            "row_width": row_width,
            "bucket_versions": {bucket_id: buckets[bucket_id].version for bucket_id in sorted(buckets)},
        }
    )
    return ExpandedSource(
        source_id=source.id,
        source_digest=data.digest,
        rows=data.rows,
        row_keys=data.row_keys,
        row_width=row_width,
        buckets=buckets,
        topo_order=tuple(rule.id for rule in rules),
        digest=expansion_digest,
    )


def make_pack_plans(expanded: ExpandedSource, packing: PackingSpec) -> tuple[PackPlan, ...]:
    """Greedy ordered split by max rows and maximum total pack cell width."""

    if expanded.rows == 0:
        return ()
    if expanded.row_width > packing.pack_width_budget:
        fail(
            E_WIDTH,
            "$.packing.pack_width_budget",
            "one expanded row exceeds the pack width budget",
            source_id=expanded.source_id,
            row_index=0,
            primary_key=expanded.row_keys[0],
            row_width=expanded.row_width,
            pack_width_budget=packing.pack_width_budget,
        )
    plans: list[PackPlan] = []
    start = 0
    index = 0
    while start < expanded.rows:
        end = start
        width = 0
        while end < expanded.rows and end - start < packing.rows_per_pack:
            candidate = width + expanded.row_width
            if candidate > packing.pack_width_budget:
                break
            width = candidate
            end += 1
        if end == start:
            fail(
                E_WIDTH,
                "$.packing.pack_width_budget",
                "one expanded row exceeds the pack width budget",
                source_id=expanded.source_id,
                row_index=start,
                primary_key=expanded.row_keys[start],
                row_width=expanded.row_width,
                pack_width_budget=packing.pack_width_budget,
            )
        pack_id = f"{expanded.source_id}:p{index:05d}"
        plans.append(
            PackPlan(
                pack_id=pack_id,
                source_id=expanded.source_id,
                index=index,
                row_start=start,
                row_end=end,
                width_used=width,
            )
        )
        start = end
        index += 1
    return tuple(plans)


def compute_pack(plan: PackPlan, expanded: ExpandedSource, unit_cost_micros: int) -> Pack:
    bucket_ids = sorted(expanded.buckets)
    bucket_counts: dict[str, tuple[int, ...]] = {}
    null_counts: dict[str, int] = {}
    bucket_versions: dict[str, str] = {}
    for bucket_id in bucket_ids:
        bucket = expanded.buckets[bucket_id]
        slice_values = bucket.values[plan.row_start : plan.row_end]
        bucket_counts[bucket_id] = tuple(bincount(slice_values, bucket.bucket_count))
        null_counts[bucket_id] = sum(1 for bit in bucket.nulls[plan.row_start : plan.row_end] if bit)
        bucket_versions[bucket_id] = _slice_bucket_version(
            bucket, plan.row_start, plan.row_end
        )
    rows_payload = []
    for row in range(plan.row_start, plan.row_end):
        rows_payload.append(
            {
                "key": expanded.row_keys[row],
                "values": [expanded.buckets[bucket_id].values[row] for bucket_id in bucket_ids],
                "nulls": "".join(
                    "1" if expanded.buckets[bucket_id].nulls[row] else "0"
                    for bucket_id in bucket_ids
                ),
            }
        )
    content_digest = digest(
        {
            "bucket_ids": bucket_ids,
            "bucket_versions": bucket_versions,
            "rows": rows_payload,
        }
    )
    width_used = (plan.row_end - plan.row_start) * expanded.row_width
    cost_micros = width_used * unit_cost_micros
    core = {
        "pack_id": plan.pack_id,
        "source_id": plan.source_id,
        "index": plan.index,
        "row_start": plan.row_start,
        "row_end": plan.row_end,
        "row_count": plan.row_end - plan.row_start,
        "width_used": width_used,
        "cost_micros": cost_micros,
        "bucket_counts": {bucket_id: list(bucket_counts[bucket_id]) for bucket_id in bucket_ids},
        "null_counts": null_counts,
        "bucket_versions": bucket_versions,
        "content_digest": content_digest,
    }
    pack_digest = digest(core)
    return Pack(
        pack_id=plan.pack_id,
        source_id=plan.source_id,
        index=plan.index,
        row_start=plan.row_start,
        row_end=plan.row_end,
        row_keys=expanded.row_keys[plan.row_start : plan.row_end],
        width_used=width_used,
        cost_micros=cost_micros,
        bucket_counts=bucket_counts,
        null_counts=null_counts,
        bucket_versions=bucket_versions,
        content_digest=content_digest,
        digest=pack_digest,
    )


def _empty_stats_like(spec: Spec) -> tuple[dict[str, Any], dict[str, Any]]:
    sources: dict[str, Any] = {}
    global_counts: dict[str, list[int]] = {}
    global_nulls: dict[str, int] = {}
    for source in spec.sources:
        counts: dict[str, list[int]] = {}
        nulls: dict[str, int] = {}
        for rule in spec.rules_for_source(source.id):
            counts[rule.id] = [0] * rule.bucket_count
            nulls[rule.id] = 0
            global_counts[rule.id] = [0] * rule.bucket_count
            global_nulls[rule.id] = 0
        sources[source.id] = {
            "row_count": 0,
            "width_used": 0,
            "bucket_counts": {key: counts[key] for key in sorted(counts)},
            "null_counts": {key: nulls[key] for key in sorted(nulls)},
        }
    return (
        {
            "row_count": 0,
            "width_used": 0,
            "bucket_counts": {key: global_counts[key] for key in sorted(global_counts)},
            "null_counts": {key: global_nulls[key] for key in sorted(global_nulls)},
        },
        sources,
    )


def merge_pack_summaries(spec: Spec, packs: Sequence[Pack]) -> dict[str, Any]:
    global_stats, source_stats = _empty_stats_like(spec)
    for pack in sorted(packs, key=lambda item: item.pack_id):
        source_stat = source_stats[pack.source_id]
        source_stat["row_count"] += pack.row_count
        source_stat["width_used"] += pack.width_used
        for bucket_id in sorted(pack.bucket_counts):
            incoming = list(pack.bucket_counts[bucket_id])
            source_stat["bucket_counts"][bucket_id] = sum_counts(
                [source_stat["bucket_counts"][bucket_id], incoming],
                len(source_stat["bucket_counts"][bucket_id]),
            )
            global_stats["bucket_counts"][bucket_id] = sum_counts(
                [global_stats["bucket_counts"][bucket_id], incoming],
                len(global_stats["bucket_counts"][bucket_id]),
            )
            source_stat["null_counts"][bucket_id] += pack.null_counts[bucket_id]
            global_stats["null_counts"][bucket_id] += pack.null_counts[bucket_id]
        global_stats["row_count"] += pack.row_count
        global_stats["width_used"] += pack.width_used
    return {"global": global_stats, "sources": {key: source_stats[key] for key in sorted(source_stats)}}


def direct_statistics(spec: Spec, expansions: Mapping[str, ExpandedSource]) -> dict[str, Any]:
    global_stats, source_stats = _empty_stats_like(spec)
    for source_id in sorted(expansions):
        expanded = expansions[source_id]
        source_stat = source_stats[source_id]
        source_stat["row_count"] = expanded.rows
        source_stat["width_used"] = expanded.rows * expanded.row_width
        global_stats["row_count"] += expanded.rows
        global_stats["width_used"] += expanded.rows * expanded.row_width
        for bucket_id in sorted(expanded.buckets):
            bucket = expanded.buckets[bucket_id]
            source_stat["bucket_counts"][bucket_id] = list(bucket.counts)
            source_stat["null_counts"][bucket_id] = bucket.null_count
            vectors = [global_stats["bucket_counts"][bucket_id], list(bucket.counts)]
            global_stats["bucket_counts"][bucket_id] = sum_counts(
                vectors, len(global_stats["bucket_counts"][bucket_id])
            )
            global_stats["null_counts"][bucket_id] += bucket.null_count
    return {"global": global_stats, "sources": {key: source_stats[key] for key in sorted(source_stats)}}


def _first_pack_for(packs: Sequence[Pack], source_id: str | None, bucket_id: str | None) -> str:
    ordered = sorted(packs, key=lambda item: item.pack_id)
    for pack in ordered:
        if source_id is not None and pack.source_id != source_id:
            continue
        if bucket_id is not None and bucket_id not in pack.bucket_counts:
            continue
        return pack.pack_id
    return ordered[0].pack_id if ordered else ""


def assert_merge_consistency(
    merged: Mapping[str, Any],
    direct: Mapping[str, Any],
    packs: Sequence[Pack],
) -> None:
    if dumps(merged) == dumps(direct):
        return
    saw_bucket_diff = False
    for source_id in sorted(set(merged["sources"]) | set(direct["sources"])):
        m_source = merged["sources"].get(source_id, {})
        d_source = direct["sources"].get(source_id, {})
        bucket_ids = sorted(
            set(m_source.get("bucket_counts", {})) | set(d_source.get("bucket_counts", {}))
        )
        for bucket_id in bucket_ids:
            if m_source.get("bucket_counts", {}).get(bucket_id) != d_source.get(
                "bucket_counts", {}
            ).get(bucket_id) or m_source.get("null_counts", {}).get(bucket_id) != d_source.get(
                "null_counts", {}
            ).get(bucket_id):
                saw_bucket_diff = True
                raise_divergence(
                    first_pack=_first_pack_for(packs, source_id, bucket_id),
                    bucket_id=bucket_id,
                    source_id=source_id,
                    field="bucket_counts/null_counts",
                    merged=m_source,
                    direct=d_source,
                )
        if m_source.get("row_count") != d_source.get("row_count") or m_source.get(
            "width_used"
        ) != d_source.get("width_used"):
            raise_divergence(
                first_pack=_first_pack_for(packs, source_id, None),
                bucket_id=None,
                source_id=source_id,
                field="row_count/width_used",
                merged=m_source,
                direct=d_source,
            )
    if not saw_bucket_diff:
        raise_divergence(
            first_pack=_first_pack_for(packs, None, None),
            bucket_id=None,
            source_id=None,
            field="global",
            merged=merged.get("global"),
            direct=direct.get("global"),
        )


def raise_divergence(
    *,
    first_pack: str,
    bucket_id: str | None,
    source_id: str | None,
    field: str,
    merged: Any,
    direct: Any,
) -> None:
    fail(
        E_DIVERGE,
        f"$.packs[{first_pack}].{field}",
        "merged pack statistics differ from direct expansion statistics",
        first_differing_pack_id=first_pack,
        first_differing_bucket_id=bucket_id,
        source_id=source_id,
        field=field,
        merged=merged,
        direct=direct,
    )


def assert_pack_layout(
    spec: Spec,
    expansions: Mapping[str, ExpandedSource],
    packs: Sequence[Pack],
) -> None:
    grouped: dict[str, list[Pack]] = {}
    for pack in packs:
        grouped.setdefault(pack.source_id, []).append(pack)
    for source_id in sorted(grouped):
        ordered = sorted(grouped[source_id], key=lambda item: item.index)
        cursor = 0
        for pack in ordered:
            if pack.row_start != cursor or pack.row_end < pack.row_start:
                fail(
                    E_DIVERGE,
                    f"$.packs[{pack.pack_id}].row_start",
                    "pack row intervals are not contiguous",
                    source_id=source_id,
                    expected_row_start=cursor,
                    actual_row_start=pack.row_start,
                    first_pack=pack.pack_id,
                )
            if pack.width_used > spec.packing.pack_width_budget:
                fail(
                    E_WIDTH,
                    f"$.packs[{pack.pack_id}].width_used",
                    "pack width exceeds the pack width budget",
                    pack_id=pack.pack_id,
                    width_used=pack.width_used,
                    pack_width_budget=spec.packing.pack_width_budget,
                )
            cursor = pack.row_end
        expected_rows = expansions[source_id].rows
        if cursor != expected_rows:
            fail(
                E_DIVERGE,
                f"$.sources[{source_id}].rows",
                "pack row intervals do not cover every row",
                source_id=source_id,
                expected_rows=expected_rows,
                covered_rows=cursor,
                first_pack=ordered[-1].pack_id if ordered else "",
            )


def derive_statistics(spec: Spec, merged: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stat in spec.statistics:
        total = 0
        refs_counts: dict[str, list[int]] = {}
        source_stat = merged["sources"][stat.source]
        for ref in sorted(stat.refs):
            counts = list(source_stat["bucket_counts"][ref])
            refs_counts[ref] = counts
            total += sum(counts)
        out[stat.id] = {
            "id": stat.id,
            "source_id": stat.source,
            "metric": stat.metric,
            "refs": sorted(stat.refs),
            "value": total,
            "per_bucket": refs_counts,
        }
    return {key: out[key] for key in sorted(out)}


def cost_ledger(spec: Spec, packs: Sequence[Pack]) -> dict[str, Any]:
    entries = [
        {
            "pack_id": pack.pack_id,
            "source_id": pack.source_id,
            "width_used": pack.width_used,
            "cost_micros": pack.cost_micros,
        }
        for pack in sorted(packs, key=lambda item: item.pack_id)
    ]
    total = sum(entry["cost_micros"] for entry in entries)
    return {
        "unit_cost_micros": spec.accounting.unit_cost_micros,
        "budget_micros": spec.accounting.budget_micros,
        "total_cost_micros": total,
        "within_budget": total <= spec.accounting.budget_micros,
        "packs": entries,
    }


def graph_view(spec: Spec, graph: DependencyGraph) -> dict[str, Any]:
    nodes = []
    for rule_id in sorted(graph.parents):
        rule = spec.rule(rule_id)
        assert rule is not None
        nodes.append(
            {
                "id": rule.id,
                "source_id": rule.source,
                "input": {"kind": rule.input_kind, "name": rule.input_name},
                "boundaries": list(rule.boundaries),
                "bucket_count": rule.bucket_count,
                "width_bits": rule.effective_width_bits,
                "cell_width": rule.cell_width,
                "parents": list(graph.parents[rule_id]),
            }
        )
    return {
        "nodes": nodes,
        "edges": stable_edges(graph),
        "topological_order": list(graph.topo),
        "statistics": {
            stat.id: list(stat.refs) for stat in sorted(spec.statistics, key=lambda item: item.id)
        },
    }


def deterministic_view(result: Mapping[str, Any]) -> dict[str, Any]:
    view = dict(result)
    view.pop("schedule", None)
    view.pop("recompute", None)
    view.pop("content_digest", None)
    return view


def assemble_result(
    spec: Spec,
    graph: DependencyGraph,
    source_data: Mapping[str, SourceData],
    expansions: Mapping[str, ExpandedSource],
    packs: Sequence[Pack],
    schedule: ScheduleResult,
    recompute: Mapping[str, Any],
) -> dict[str, Any]:
    merged = merge_pack_summaries(spec, packs)
    direct = direct_statistics(spec, expansions)
    assert_merge_consistency(merged, direct, packs)
    assert_pack_layout(spec, expansions, packs)
    derived = derive_statistics(spec, merged)
    cost = cost_ledger(spec, packs)
    result: dict[str, Any] = {
        "format_version": 1,
        "spec_digest": spec.digest,
        "graph": graph_view(spec, graph),
        "sources": {
            source_id: expansions[source_id].to_dict() for source_id in sorted(expansions)
        },
        "packs": [pack.to_dict() for pack in sorted(packs, key=lambda item: item.pack_id)],
        "merged": merged,
        "direct": direct,
        "derived_statistics": derived,
        "cost_ledger": cost,
        "schedule": schedule.to_dict(),
        "recompute": dict(recompute),
        "policy": {
            "quantization": "Decimal(str(value)); truncate toward zero; saturate to signed int64",
            "bucket_rule": BUCKET_RULE,
            "width_rule": NULL_WIDTH_POLICY,
            "packing": "ordered greedy split; rows_per_pack maximum and total row-width maximum",
            "pack_version_scope": "pack-local row-slice bucket versions keep unaffected packs byte-stable",
            "merge": "bucket counts, null counts and width are additive across packs",
        },
        "consistency": {
            "merge_matches_direct": True,
            "pack_ranges_contiguous": True,
            "cost_within_budget": cost["within_budget"],
        },
    }
    result["content_digest"] = digest(deterministic_view(result))
    return result


def run_full(
    spec: Spec,
    *,
    value_overrides: Mapping[str, Mapping[tuple[int, str], Any]] | None = None,
    null_overrides: Mapping[str, Mapping[tuple[int, str], bool]] | None = None,
    on_plans_ready: Callable[[Sequence[PackPlan]], None] | None = None,
    on_pack_complete: Callable[[Pack, ScheduleEntry], None] | None = None,
) -> PipelineState:
    graph = build_graph(spec)
    value_overrides = value_overrides or {}
    null_overrides = null_overrides or {}
    source_data: dict[str, SourceData] = {}
    expansions: dict[str, ExpandedSource] = {}
    counters: dict[str, list[str]] = {"recomputed": [], "reused": []}
    for source in spec.sources:
        data = load_source(
            spec,
            source,
            value_overrides=value_overrides.get(source.id, {}),
            null_overrides=null_overrides.get(source.id, {}),
        )
        source_data[source.id] = data
        expansions[source.id] = expand_source(spec, graph, data, counters=counters)
    plans: list[PackPlan] = []
    for source in spec.sources:
        plans.extend(make_pack_plans(expansions[source.id], spec.packing))
    plans.sort(key=lambda item: item.pack_id)
    if on_plans_ready is not None:
        on_plans_ready(tuple(plans))

    def compute(plan: PackPlan) -> Pack:
        return compute_pack(
            plan,
            expansions[plan.source_id],
            spec.accounting.unit_cost_micros,
        )

    schedule, computed = schedule_packs(plans, spec.packing, compute, on_result=on_pack_complete)
    packs = tuple(computed[plan.pack_id] for plan in plans)
    recompute = {
        "mode": "full",
        "triggers": [],
        "recomputed_bucket_columns": sorted(counters["recomputed"]),
        "reused_bucket_columns": sorted(counters["reused"]),
        "recomputed_packs": sorted(pack.pack_id for pack in packs),
        "reused_packs": [],
        "trace": [],
    }
    result = assemble_result(spec, graph, source_data, expansions, packs, schedule, recompute)
    return PipelineState(
        spec=spec,
        graph=graph,
        source_data=source_data,
        expansions=expansions,
        packs=tuple(sorted(packs, key=lambda item: item.pack_id)),
        schedule=schedule,
        result=result,
    )
