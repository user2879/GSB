"""Fixed-point operator chain execution engine with incremental ledger.

Stage artifact shapes
---------------------
embed        : {"rows": [[int]], "tokens": [int], "row_fps": [str]}
segment_pool : {"sum": [int], "count": int, "segment": [start, end]} (one/segment)
multihead    : {"heads": [[[int]]], "vectors": [[int]]}
layer_norm   : {"rows": [[int]], "means": [int], "variances": [int]}
final        : {"rows": [[int]], "per_token": [{"token","values","residual"}]}

Each artifact keeps a content-derived cache key plus token/op lineage, so an
edit invalidates exactly the entries that reference the changed token or op.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PipelineModel
from .kernel import (
    QuantContext,
    embed_lookup,
    layer_norm,
    make_context,
    multihead_attention,
    segment_pool,
)
from .ledger import (
    Ledger,
    LedgerEntry,
    derive_cache_key,
    make_entry_key,
    output_fingerprint,
    param_fingerprint,
    token_row_fingerprint,
)

STAGE_EMBED = "embed"
STAGE_POOL = "segment_pool"
STAGE_ATTN = "multihead_attn"
STAGE_NORM = "layer_norm"
STAGE_FINAL = "final"
STAGES = (STAGE_EMBED, STAGE_POOL, STAGE_ATTN, STAGE_NORM, STAGE_FINAL)
# Pool uses an explicit pool stage even when the op type is segment_pool.
STAGE_BY_TYPE = {
    "embed": STAGE_EMBED,
    "segment_pool": STAGE_POOL,
    "multihead_attn": STAGE_ATTN,
    "layer_norm": STAGE_NORM,
    "final": STAGE_FINAL,
}


@dataclass
class EnginePlan:
    to_compute: list[dict] = field(default_factory=list)
    reusable: list[str] = field(default_factory=list)
    invalidated: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, model: PipelineModel, exchange_version: str, embedding_table: dict[int, list[int]]):
        self.model = model
        self.config = model.config
        self.exchange_version = exchange_version
        self.embedding_table = embedding_table
        self.ledger = Ledger()
        self._op_map = {op.node_id(): op for op in model.all_ops()}
        self.stage_points = {stage: 0 for stage in STAGES}

    # ------------------------------------------------------------------ #
    def context_for(self, op) -> QuantContext:
        width = op.quant.width or 32
        permille = op.quant.permille if op.quant.permille is not None else 1000
        return make_context(self.config, width, permille)

    def _op_stage(self, op) -> str:
        return STAGE_BY_TYPE[op.op_type]

    def _entry_tokens(self, op, segment: tuple[int, int] | None = None) -> list[int]:
        if op.op_type == "segment_pool" and segment is not None:
            return list(range(segment[0], segment[1]))
        return list(range(len(self.model.tokens)))

    def _entry_key(self, op, segment: tuple[int, int] | None = None) -> str:
        return make_entry_key(op.node_id(), self._op_stage(op), self._entry_tokens(op, segment))

    def _upstream_entry(self, ref_op, segment: tuple[int, int] | None = None) -> LedgerEntry | None:
        return self.ledger.get(self._entry_key(ref_op, segment))

    def _upstream_keys(self, op, segment: tuple[int, int] | None) -> list[str]:
        keys: list[str] = []
        token_hint = self._entry_tokens(op, segment)
        for ref in op.inputs:
            ref_op = self._op_map.get(f"c{op.chain_id}op{ref}")
            if ref_op is None:
                continue
            if ref_op.op_type == "segment_pool":
                for seg in self.model.segments:
                    seg_tokens = set(range(seg[0], seg[1]))
                    if seg_tokens & set(token_hint):
                        keys.append(self._entry_key(ref_op, seg))
            else:
                keys.append(self._entry_key(ref_op, None))
        return keys

    def _referenced(self, op, token_indices: list[int], upstream_keys: list[str]):
        tokens = set(token_indices)
        ops = {op.node_id()}
        for ref in op.inputs:
            ops.add(f"c{op.chain_id}op{ref}")
        for key in upstream_keys:
            prior = self.ledger.get(key)
            if prior:
                tokens.update(prior.referenced_tokens)
                ops.update(prior.referenced_ops)
        return sorted(tokens), sorted(ops)

    def _make_entry(self, op, token_indices, output, upstream_keys, cost, segment=None) -> LedgerEntry:
        pfp = param_fingerprint(op)
        ckey = derive_cache_key(
            upstream_keys, pfp, self.config.gear, self.config.frac_bits, self.exchange_version
        )
        ref_tokens, ref_ops = self._referenced(op, token_indices, upstream_keys)
        entry = LedgerEntry(
            entry_key=self._entry_key(op, segment),
            node_id=op.node_id(),
            chain_id=op.chain_id,
            seq=op.seq,
            op_type=op.op_type,
            stage=self._op_stage(op),
            token_indices=list(token_indices),
            cache_key=ckey,
            output_fp=output_fingerprint(output),
            output=output,
            upstream_keys=upstream_keys,
            param_fp=pfp,
            gear=self.config.gear,
            exchange_version=self.exchange_version,
            frac_bits=self.config.frac_bits,
            referenced_tokens=ref_tokens,
            referenced_ops=ref_ops,
            cost_points=cost,
        )
        self.stage_points[entry.stage] = self.stage_points.get(entry.stage, 0) + cost
        return entry

    # ------------------------------------------------------------------ #
    # Invalidation
    # ------------------------------------------------------------------ #
    def expected_cache_key(self, entry: LedgerEntry) -> str:
        """Recompute the cache key an entry would have under current state.

        This captures edits whose final state matches the baseline (e.g. a
        gear that is changed and then changed back): the op-param fingerprint
        or upstream keys still differ, so the entry is correctly invalidated.
        """
        op = self._op_map.get(entry.node_id)
        if op is None:
            return entry.cache_key + "|op-removed"
        segment = tuple(entry.token_indices[0:1] and (entry.token_indices[0],
                                                       entry.token_indices[-1] + 1)) if entry.stage == STAGE_POOL else None
        upstream_keys = self._upstream_keys(op, segment)
        return derive_cache_key(
            upstream_keys, param_fingerprint(op),
            self.config.gear, self.config.frac_bits, self.exchange_version,
        )

    def reconcile(self, changed_tokens: set[int], changed_ops: set[str]) -> list[str]:
        """Exact content-fingerprint invalidation after any edit sequence."""
        removed: list[str] = []
        for entry in self.ledger.all_entries():
            reason = None
            if entry.node_id in changed_ops or any(op in changed_ops for op in entry.referenced_ops):
                reason = "op-edit"
            elif changed_tokens and any(tok in changed_tokens for tok in entry.referenced_tokens):
                reason = "token-edit"
            elif (entry.gear != self.config.gear or entry.frac_bits != self.config.frac_bits
                  or entry.exchange_version != self.exchange_version):
                reason = "quant-or-version-changed"
            else:
                expected = self.expected_cache_key(entry)
                if expected != entry.cache_key:
                    reason = "content-fingerprint-mismatch"
            if reason and self.ledger.invalidate(entry.entry_key, reason):
                removed.append(entry.entry_key)
        return removed

    # ------------------------------------------------------------------ #
    # Planning: which entries are missing vs reusable
    # ------------------------------------------------------------------ #
    def _topo_levels(self) -> dict[str, int]:
        """Longest-input-path level per node; guarantees upstream-first order."""
        levels: dict[str, int] = {}
        for op in self.model.all_ops():
            self._level_for(op, levels)
        return levels

    def _level_for(self, op, levels: dict[str, int]) -> int:
        if op.node_id() in levels:
            return levels[op.node_id()]
        if not op.inputs:
            levels[op.node_id()] = 0
            return 0
        parent_levels = []
        for ref in op.inputs:
            parent = self._op_map.get(f"c{op.chain_id}op{ref}")
            parent_levels.append(0 if parent is None else self._level_for(parent, levels))
        levels[op.node_id()] = 1 + max(parent_levels)
        return levels[op.node_id()]

    def plan(self) -> EnginePlan:
        plan = EnginePlan()
        levels = self._topo_levels()
        units: list[dict] = []
        for op in self.model.all_ops():
            scopes = self.model.segments if op.op_type == "segment_pool" else [None]
            for segment in scopes:
                key = self._entry_key(op, segment)
                if self.ledger.get(key) is not None:
                    plan.reusable.append(key)
                    self.ledger.record_reuse(self.ledger.get(key), "cache-hit")
                else:
                    units.append({
                        "node_id": op.node_id(),
                        "chain_id": op.chain_id,
                        "seq": op.seq,
                        "op_type": op.op_type,
                        "stage": self._op_stage(op),
                        "segment": segment,
                        "entry_key": key,
                        "level": levels[op.node_id()],
                    })
        # Topological level first, then stable chain/op/token order.
        units.sort(key=lambda item: (item["level"], item["chain_id"], item["seq"],
                                     item["segment"] or (-1, -1)))
        plan.to_compute = [{k: v for k, v in unit.items() if k != "level"} for unit in units]
        return plan

    # ------------------------------------------------------------------ #
    # Compute one planned unit. Upstream entries already exist in ledger.
    # ------------------------------------------------------------------ #
    def compute_unit(self, unit: dict) -> LedgerEntry:
        op = self._op_map[unit["node_id"]]
        segment = unit["segment"]
        upstream_keys = self._upstream_keys(op, segment)
        ctx = self.context_for(op)
        dim = self.config.dim

        if op.op_type == "embed":
            rows, row_fps, tokens = [], [], []
            for token_index, token in enumerate(self.model.tokens):
                raw = self.embedding_table[token]
                rows.append(embed_lookup(token, raw, ctx))
                row_fps.append(token_row_fingerprint(token, raw))
                tokens.append(token)
            output = {"rows": rows, "tokens": tokens, "row_fps": row_fps}
            cost = dim * len(tokens)
            entry = self._make_entry(op, list(range(len(tokens))), output, upstream_keys, cost)

        elif op.op_type == "segment_pool":
            upstream = self._first_upstream(op, segment)
            start, end = segment
            seg_rows = [upstream.output["rows"][index] for index in range(start, end)]
            pooled = segment_pool(seg_rows, ctx)
            output = {"sum": pooled["sum"], "count": pooled["count"], "segment": [start, end]}
            cost = dim * (end - start)
            entry = self._make_entry(op, list(range(start, end)), output, upstream_keys, cost, segment)

        elif op.op_type == "multihead_attn":
            upstream = self._first_upstream(op)
            rows = self._value_rows(upstream)
            attended = multihead_attention(op.node_id(), rows, self.config.heads, ctx)
            cost = dim * len(rows) * self.config.heads
            entry = self._make_entry(op, list(range(len(rows))), attended, upstream_keys, cost)

        elif op.op_type == "layer_norm":
            upstream = self._first_upstream(op)
            rows = self._value_rows(upstream)
            normalized = layer_norm(rows, ctx)
            cost = dim * len(rows) * 2
            entry = self._make_entry(op, list(range(len(rows))), normalized, upstream_keys, cost)

        elif op.op_type == "final":
            upstream = self._first_upstream(op)
            rows = self._value_rows(upstream)
            per_token = []
            for token_index, row in enumerate(rows):
                residual = sum(row) - sum(
                    ctx.saturate(value) for value in row
                )
                per_token.append({
                    "token_index": token_index,
                    "token": self.model.tokens[token_index],
                    "values": list(row),
                    "residual": residual,
                })
            output = {"rows": rows, "per_token": per_token}
            cost = dim * len(rows)
            entry = self._make_entry(op, list(range(len(rows))), output, upstream_keys, cost)
        else:  # pragma: no cover - guarded by loader validation
            raise ValueError(f"unknown op type {op.op_type}")

        self.ledger.upsert(entry)
        return entry

    def _first_upstream(self, op, segment: tuple[int, int] | None = None) -> LedgerEntry:
        for ref in op.inputs:
            ref_op = self._op_map[f"c{op.chain_id}op{ref}"]
            # Only a pooling upstream is keyed per segment; other stages use
            # the full token scope regardless of the caller's segment.
            ref_segment = segment if ref_op.op_type == "segment_pool" else None
            entry = self._upstream_entry(ref_op, ref_segment)
            if entry is not None:
                return entry
        raise KeyError(f"no upstream entry ready for {op.node_id()}")

    @staticmethod
    def _value_rows(entry: LedgerEntry) -> list[list[int]]:
        output = entry.output
        if "rows" in output:
            return output["rows"]
        if "vectors" in output:
            return output["vectors"]
        if "sum" in output:
            return [output["sum"]]
        raise KeyError(f"entry {entry.entry_key} has no row-shaped output")

    def execute_plan(self, plan: EnginePlan, compute_callback=None) -> list[LedgerEntry]:
        produced: list[LedgerEntry] = []
        for unit in plan.to_compute:
            entry = compute_callback(unit) if compute_callback else self.compute_unit(unit)
            produced.append(entry)
        return produced

    def all_outputs(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for entry in self.ledger.all_entries():
            result[entry.entry_key] = {
                "cache_key": entry.cache_key,
                "output_fp": entry.output_fp,
                "stage": entry.stage,
                "token_indices": entry.token_indices,
                "referenced_tokens": entry.referenced_tokens,
                "referenced_ops": entry.referenced_ops,
                "output": entry.output,
            }
        return result
