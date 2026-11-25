"""Loading + validation layer for graph, tokens, segments and edits.

Bad entries emit an issue (e.g. ``E_SCHEMA``) and are skipped; sibling entries
keep loading. The e1/e2 quant records are merged onto the structural operator
graph so both exchange versions converge on one internal representation.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import (
    Chain,
    OpSpec,
    PipelineConfig,
    PipelineModel,
    validate_chain,
    validate_tokens_segments,
)
from .errors import Code, IssueLog
from .exchange import QuantRecord, parse_e1_file, parse_e2_file, read_bypass

VALID_OP_TYPES = {"embed", "segment_pool", "multihead_attn", "layer_norm", "final"}


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tokens(path: str | Path, config: PipelineConfig, log: IssueLog) -> list[int] | None:
    payload = load_json(path)
    tokens = payload.get("tokens") if isinstance(payload, dict) else payload
    if not isinstance(tokens, list):
        log.add(Code.E_SCHEMA, "tokens", "payload needs a token list")
        return None
    segments_raw = payload.get("segments") if isinstance(payload, dict) else None
    segments = _coerce_segments(segments_raw, log)
    if not validate_tokens_segments(tokens, segments, config.vocab_size, log):
        return None
    return tokens


def _coerce_segments(raw: object, log: IssueLog) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        log.add(Code.E_SCHEMA, "segments", "segments must be a list of [start,end]")
        return []
    segments: list[tuple[int, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            log.add(Code.E_SCHEMA, f"segments[{index}]", "segment must be [start,end]")
            continue
        segments.append((int(item[0]), int(item[1])))
    return segments


def load_segments(path: str | Path, token_count: int, log: IssueLog) -> list[tuple[int, int]] | None:
    payload = load_json(path)
    raw = payload.get("segments") if isinstance(payload, dict) else payload
    segments = _coerce_segments(raw, log)
    ok = True
    expected_start = 0
    for index, (start, end) in enumerate(segments):
        if start != expected_start or end > token_count or start >= end:
            log.add(Code.E_SCHEMA, f"segments[{index}]", "boundaries must be strictly increasing and cover tokens")
            ok = False
        expected_start = end
    if ok and (not segments or segments[-1][1] != token_count):
        log.add(Code.E_SCHEMA, "segments", f"must cover all {token_count} tokens")
        ok = False
    return segments if ok else None


def _quant_index(records: list[QuantRecord]) -> dict[str, QuantRecord]:
    return {record.node_id(): record for record in records}


def load_graph(
    path: str | Path,
    quant_records: list[QuantRecord],
    log: IssueLog,
) -> dict[int, Chain]:
    payload = load_json(path)
    raw_chains = payload.get("chains", []) if isinstance(payload, dict) else payload
    quant_by_node = _quant_index(quant_records)
    chains: dict[int, Chain] = {}
    seen_chain_ids: set[int] = set()
    for chain_index, raw_chain in enumerate(raw_chains):
        chain_id = raw_chain.get("chain_id")
        if not isinstance(chain_id, int):
            log.add(Code.E_SCHEMA, f"graph.chains[{chain_index}]", "chain_id missing", raw_chain)
            continue
        if chain_id in seen_chain_ids:
            log.add(Code.E_DUPLICATE, f"graph.chain[{chain_id}]", "duplicate chain id")
            continue
        seen_chain_ids.add(chain_id)
        ops: list[OpSpec] = []
        for raw_op in raw_chain.get("ops", []):
            seq = raw_op.get("seq")
            op_type = raw_op.get("op_type")
            if not isinstance(seq, int) or op_type not in VALID_OP_TYPES:
                log.add(Code.E_SCHEMA, f"chain[{chain_id}].op", "bad seq/op_type", raw_op)
                continue
            node_id = f"c{chain_id}op{seq}"
            record = quant_by_node.get(node_id)
            if record is None:
                log.add(Code.E_MISSING_FIELD, node_id, "no quant record from exchange file")
                continue
            params = raw_op.get("params", {})
            if not isinstance(params, dict):
                log.add(Code.E_SCHEMA, node_id, "params must be an object")
                continue
            inputs = tuple(sorted(int(ref) for ref in raw_op.get("inputs", []) if isinstance(ref, int)))
            op = OpSpec(
                chain_id=chain_id,
                seq=seq,
                op_type=op_type,
                inputs=inputs,
                params=params,
                quant=record.quant,
                source_version=record.source_version,
                missing_fields=tuple(record.quant.missing),
            )
            ops.append(op)
        if ops and validate_chain(chain_id, ops, log):
            chains[chain_id] = Chain(chain_id=chain_id, ops=ops)
        elif ops:
            log.add(Code.E_SCHEMA, f"chain[{chain_id}]", "chain rejected due to validation failure")
    return chains


def load_pipeline(
    config_path: str | Path,
    tokens_path: str | Path,
    segments_path: str | Path,
    graph_path: str | Path,
    e1_path: str | Path | None = None,
    e2_path: str | Path | None = None,
    bypass_path: str | Path | None = None,
) -> PipelineModel:
    log = IssueLog()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = PipelineConfig.from_dict(config)
    cfg.validate(log, "config")

    token_payload = load_json(tokens_path)
    tokens = token_payload.get("tokens") if isinstance(token_payload, dict) else token_payload
    if not isinstance(tokens, list):
        log.add(Code.E_SCHEMA, "tokens", "token list required")
        tokens = []
    segments = load_segments(segments_path, len(tokens), log) or []

    records: list[QuantRecord] = []
    bypass_map = read_bypass(bypass_path) if bypass_path else {}
    if e1_path and Path(e1_path).exists():
        records.extend(parse_e1_file(e1_path, log))
    if e2_path and Path(e2_path).exists():
        records.extend(parse_e2_file(e2_path, log, bypass_map))
    chains = load_graph(graph_path, records, log) if records else {}
    return PipelineModel(config=cfg, tokens=tokens, segments=segments, chains=chains, log=log)


def load_edits(path: str | Path, log: IssueLog) -> list[dict]:
    payload = load_json(path)
    edits = payload.get("edits", []) if isinstance(payload, dict) else payload
    valid: list[dict] = []
    allowed = {
        "update_op_param", "add_op", "remove_op", "replace_token",
        "remove_token", "set_segments", "set_quant", "set_version",
        "set_gear", "set_frac_bits",
    }
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or edit.get("cmd") not in allowed:
            log.add(Code.E_SCHEMA, f"edits[{index}]", "unknown edit command", edit)
            continue
        valid.append(edit)
    return valid
