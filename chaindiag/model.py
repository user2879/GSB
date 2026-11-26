from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import E_CHAIN_FORMAT, ChainFormatError, Issue
from .fixedpoint import (
    ALLOWED_FRAC_BITS,
    ALLOWED_INT_BITS,
    ALLOWED_QUANT_TIERS,
    SCHEME_TO_INT_BITS,
)


KIND_ALIASES = {
    "embedding": "embedding",
    "embed": "embedding",
    "segment_pool": "segment_pool",
    "pool": "segment_pool",
    "multi_head_attention": "attention",
    "attention": "attention",
    "mha": "attention",
    "layer_norm": "layer_norm",
    "layernorm": "layer_norm",
    "ln": "layer_norm",
}

ALLOWED_KINDS = tuple(sorted(set(KIND_ALIASES.values())))
VERSION_RE = re.compile(r"^(e[12])(?:[-_/]v?(\d+)(?:\.(\d+))?)?$", re.IGNORECASE)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issue = Issue(
            code=E_CHAIN_FORMAT,
            message=f"JSON 解析失败: {exc.msg}",
            location=str(path),
            line=exc.lineno,
            field=f"column:{exc.colno}",
        )
        raise ChainFormatError(issue) from exc
    except OSError as exc:
        issue = Issue(code=E_CHAIN_FORMAT, message=f"无法读取文件: {exc}", location=str(path))
        raise ChainFormatError(issue) from exc


def parse_version_string(raw: Any, location: str = "$.version") -> str:
    if not isinstance(raw, str):
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "版本串必须是字符串", field=location, location=location))
    match = VERSION_RE.match(raw.strip())
    if not match:
        raise ChainFormatError(
            Issue(
                E_CHAIN_FORMAT,
                f"版本串解析失败: {raw!r}; 支持 e1/e2 或 e1-1.0/e2-1.0",
                field=location,
                location=location,
            )
        )
    return match.group(1).lower()


def _issue(errors: list[Issue], message: str, location: str, field: str | None = None) -> None:
    errors.append(Issue(E_CHAIN_FORMAT, message, location=location, field=field or location))


def validate_config(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[Issue] = []
    if not isinstance(data, dict):
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "配置根必须是 JSON 对象", location="$"))

    cfg = dict(data)
    cfg.setdefault("schema_version", "1.0")
    cfg.setdefault("seed", 20261127)
    cfg.setdefault("vocab_size", 32)
    cfg.setdefault("embedding_dim", 6)
    cfg.setdefault("heads", 2)
    cfg.setdefault("head_dim", 3)
    cfg.setdefault("frac_bits", 8)
    cfg.setdefault("quant_tier", "q2")
    cfg.setdefault("exchange_version", "e2")
    cfg.setdefault("point_budget", 512)
    cfg.setdefault("operator_quota", 128)
    cfg.setdefault("max_workers", 4)
    cfg.setdefault("width_tolerance", "0.0005")
    cfg.setdefault("tie_rule", "half_away_from_zero")

    for key in ("seed", "vocab_size", "embedding_dim", "heads", "head_dim", "point_budget", "operator_quota", "max_workers"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            _issue(errors, f"{key} 必须是正整数", "$", key)

    if cfg.get("frac_bits") not in ALLOWED_FRAC_BITS:
        _issue(errors, f"frac_bits 必须是 {ALLOWED_FRAC_BITS} 之一", "$.frac_bits", "frac_bits")
    if cfg.get("quant_tier") not in ALLOWED_QUANT_TIERS:
        _issue(errors, f"quant_tier 必须是 {ALLOWED_QUANT_TIERS} 之一", "$.quant_tier", "quant_tier")
    try:
        parse_version_string(cfg.get("exchange_version"), "$.exchange_version")
    except ChainFormatError as exc:
        errors.extend(exc.issues)
    if cfg.get("tie_rule") not in ("half_away_from_zero", "half_even"):
        _issue(errors, "tie_rule 仅支持 half_away_from_zero/half_even", "$.tie_rule", "tie_rule")
    if not isinstance(cfg.get("width_tolerance"), (str, int, float)):
        _issue(errors, "width_tolerance 必须是十进制数或字符串", "$.width_tolerance", "width_tolerance")
    if isinstance(cfg.get("heads"), int) and isinstance(cfg.get("head_dim"), int) and isinstance(cfg.get("embedding_dim"), int):
        if cfg["heads"] * cfg["head_dim"] != cfg["embedding_dim"]:
            _issue(
                errors,
                f"heads*head_dim={cfg['heads'] * cfg['head_dim']} 必须等于 embedding_dim={cfg['embedding_dim']}",
                "$.embedding_dim",
                "embedding_dim",
            )
    if errors:
        raise ChainFormatError(errors)
    return cfg


def normalize_tokens(data: Any, vocab_size: int | None = None) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        raw_tokens = data.get("tokens", [])
    else:
        raw_tokens = data
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "tokens 必须是非空数组", location="tokens"))

    errors: list[Issue] = []
    tokens: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_seqs: set[int] = set()
    for index, item in enumerate(raw_tokens):
        loc = f"tokens[{index}]"
        if not isinstance(item, dict):
            _issue(errors, "令牌条目必须是对象", loc)
            continue
        uid = item.get("uid", f"t{index:04d}")
        token = item.get("token", item.get("token_id"))
        seq = item.get("seq", index)
        if not isinstance(uid, str) or not uid:
            _issue(errors, "uid 必须是非空字符串", f"{loc}.uid", "uid")
        elif uid in seen_uids:
            _issue(errors, f"令牌 uid 重复: {uid}", f"{loc}.uid", "uid")
        else:
            seen_uids.add(uid)
        if not isinstance(token, int) or isinstance(token, bool):
            _issue(errors, "token 必须是整数种子表序号", f"{loc}.token", "token")
        elif vocab_size is not None and not 0 <= token < vocab_size:
            _issue(errors, f"token={token} 超出 vocab_size={vocab_size}", f"{loc}.token", "token")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            _issue(errors, "seq 必须是非负整数", f"{loc}.seq", "seq")
        elif seq in seen_seqs:
            _issue(errors, f"令牌 seq 重复: {seq}", f"{loc}.seq", "seq")
        else:
            seen_seqs.add(seq)
        if not errors or all(e.location != loc for e in errors):
            tokens.append({"seq": int(seq) if isinstance(seq, int) else index, "uid": uid, "token": token})

    if errors:
        raise ChainFormatError(errors)
    tokens.sort(key=lambda item: (item["seq"], item["uid"]))
    for index, token in enumerate(tokens):
        token["seq"] = index
    return tokens


def normalize_boundaries(data: Any, token_count: int) -> list[int]:
    if isinstance(data, dict) and "boundaries" in data:
        raw = data["boundaries"]
        location = "segments.boundaries"
    elif isinstance(data, dict) and "segments" in data:
        raw = data["segments"]
        location = "segments"
    else:
        raw = data
        location = "segments"

    errors: list[Issue] = []
    boundaries: list[int]
    if isinstance(raw, list) and raw and all(isinstance(x, int) and not isinstance(x, bool) for x in raw):
        boundaries = list(raw)
    elif isinstance(raw, list):
        boundaries = [0]
        for index, segment in enumerate(raw):
            loc = f"{location}[{index}]"
            if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
                _issue(errors, "段必须包含 start/end，或直接提供 boundaries 整数数组", loc)
                continue
            if segment["start"] != boundaries[-1]:
                _issue(errors, f"段 start={segment['start']} 未紧接前一边界 {boundaries[-1]}", f"{loc}.start", "start")
            boundaries.append(segment["end"])
    else:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "段边界必须是整数数组或段对象数组", location=location))

    if errors:
        raise ChainFormatError(errors)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in boundaries):
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "边界必须全部是整数", location=location))
    if boundaries[0] != 0:
        _issue(errors, "首个边界必须为 0", f"{location}[0]", "boundaries[0]")
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            _issue(
                errors,
                f"边界必须严格递增: {boundaries[index - 1]} -> {boundaries[index]}",
                f"{location}[{index}]",
                f"boundaries[{index}]",
            )
    if not boundaries or boundaries[-1] != token_count:
        _issue(errors, f"末边界 {boundaries[-1] if boundaries else None} 必须等于令牌数 {token_count}", f"{location}[-1]", "boundaries[-1]")
    if any(x < 0 or x > token_count for x in boundaries):
        _issue(errors, f"存在越界边界，合法范围为 [0,{token_count}]", location, "boundaries")
    if errors:
        raise ChainFormatError(errors)
    return boundaries


def _normalize_operator(raw: dict[str, Any], index: int, errors: list[Issue]) -> dict[str, Any] | None:
    loc = f"operators[{index}]"
    if not isinstance(raw, dict):
        _issue(errors, "算子必须是对象", loc)
        return None
    seq = raw.get("seq", raw.get("op_seq", index))
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        _issue(errors, "算子序号必须是非负整数", f"{loc}.seq", "seq")
        seq = index

    raw_kind = raw.get("kind", raw.get("type"))
    kind = KIND_ALIASES.get(str(raw_kind).strip().lower()) if raw_kind is not None else None
    if kind is None:
        _issue(errors, f"未知算子树/算子类型: {raw_kind!r}; 允许 {ALLOWED_KINDS}", f"{loc}.kind", "kind")

    inputs = raw.get("inputs", raw.get("input_refs", []))
    if not isinstance(inputs, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in inputs):
        _issue(errors, "inputs 必须是整数算子序号数组", f"{loc}.inputs", "inputs")
        inputs = []

    int_bits = raw.get("int_bits", raw.get("quant_bits"))
    scheme = raw.get("quant_scheme", raw.get("named_quantization"))
    if int_bits is None and isinstance(scheme, str):
        int_bits = SCHEME_TO_INT_BITS.get(scheme)
    if int_bits is None:
        int_bits = 16
    if int_bits not in ALLOWED_INT_BITS:
        _issue(errors, f"整数量化位宽必须是 {ALLOWED_INT_BITS} 或已命名方案", f"{loc}.int_bits", "int_bits")

    scale_permille = raw.get("scale_permille")
    scale_float = raw.get("scale_float")
    if scale_permille is None and scale_float is not None:
        try:
            from fractions import Fraction

            scale_permille = int((Fraction(str(scale_float)) * 1000).numerator // (Fraction(str(scale_float)) * 1000).denominator)
        except Exception:  # noqa: BLE001 - validation reports below
            _issue(errors, "scale_float 不是有限数值", f"{loc}.scale_float", "scale_float")
    if scale_permille is None:
        scale_permille = 1000
    if not isinstance(scale_permille, int) or isinstance(scale_permille, bool) or scale_permille <= 0:
        _issue(errors, "scale_permille 必须是正整数千分比", f"{loc}.scale_permille", "scale_permille")

    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        _issue(errors, "params 必须是对象", f"{loc}.params", "params")
        params = {}

    if errors:
        # Still return a best-effort object for duplicate analysis when possible.
        if kind is None or not isinstance(seq, int):
            return None
    return {
        "seq": int(seq),
        "kind": kind,
        "name": str(raw.get("name", kind or f"op{seq}")),
        "inputs": list(inputs),
        "scale_permille": int(scale_permille) if isinstance(scale_permille, int) else 1000,
        "scale_float": scale_float,
        "int_bits": int(int_bits) if int_bits in ALLOWED_INT_BITS else 16,
        "quant_scheme": scheme,
        "params": params,
        "attention_snapshot": raw.get("attention_snapshot", params.get("attention_snapshot")),
        "exchange_meta": raw.get("exchange_meta", {}),
        "source_line": raw.get("source_line"),
    }


def validate_chain(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("chains"), list):
        chains = data["chains"]
        chain_seq_values = [chain.get("seq", chain.get("chain_seq")) if isinstance(chain, dict) else None for chain in chains]
        seen: set[Any] = set()
        errors: list[Issue] = []
        for index, seq in enumerate(chain_seq_values):
            if seq in seen:
                _issue(errors, f"链序号唯一校验失败，重复 chain seq={seq}", f"chains[{index}].seq", "chain_seq")
            seen.add(seq)
        if errors:
            raise ChainFormatError(errors)
        active = data.get("active_chain_seq", chain_seq_values[0] if chain_seq_values else 1)
        selected = next((chain for chain in chains if isinstance(chain, dict) and chain.get("seq", chain.get("chain_seq")) == active), None)
        if selected is None:
            raise ChainFormatError(Issue(E_CHAIN_FORMAT, f"active_chain_seq={active} 不存在", "$.active_chain_seq", "active_chain_seq"))
        meta = {"chain_seq": active, "root": data}
        return _validate_operator_list(selected.get("operators", []), "chains[active].operators"), meta

    if isinstance(data, dict):
        version = data.get("version") or data.get("exchange_version")
        if version is not None:
            parse_version_string(version, "$.version")
        meta = {"chain_seq": data.get("chain_seq", data.get("seq", 1)), "root": data}
        return _validate_operator_list(data.get("operators", []), "$.operators"), meta

    if isinstance(data, list):
        return _validate_operator_list(data, "$"), {"chain_seq": 1, "root": {"operators": data}}

    raise ChainFormatError(Issue(E_CHAIN_FORMAT, "算子链必须是对象或数组", location="$"))


def _validate_operator_list(raw_operators: Any, base_location: str) -> list[dict[str, Any]]:
    if not isinstance(raw_operators, list) or not raw_operators:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "算子链必须包含非空 operators 数组", location=base_location))

    errors: list[Issue] = []
    operators: list[dict[str, Any]] = []
    seen_seq: set[int] = set()
    for index, raw in enumerate(raw_operators):
        op = _normalize_operator(raw, index, errors)
        if op is None:
            continue
        op["location"] = f"{base_location}[{index}]"
        if op["seq"] in seen_seq:
            _issue(errors, f"算子序号重复: {op['seq']}", f"{base_location}[{index}].seq", "seq")
        seen_seq.add(op["seq"])
        operators.append(op)

    if not errors:
        expected = list(range(len(operators)))
        actual = sorted(op["seq"] for op in operators)
        if actual != expected:
            _issue(errors, f"算子序号必须从 0 连续且无缺口，期望 {expected}，实际 {actual}", base_location, "seq")

    by_seq = {op["seq"]: op for op in operators}
    for op in operators:
        for ref in op["inputs"]:
            if ref not in by_seq:
                _issue(errors, f"算子 {op['seq']} 的输入引用 {ref} 不存在", f"{base_location}[{op['seq']}].inputs", "inputs")
            elif ref == op["seq"]:
                _issue(errors, f"算子 {op['seq']} 不能引用自身", f"{base_location}[{op['seq']}].inputs", "inputs")

    # Cycle detection with explicit location.
    graph = {op["seq"]: [ref for ref in op["inputs"] if ref in by_seq] for op in operators}
    state: dict[int, int] = {}

    def visit(node: int, stack: list[int]) -> None:
        state[node] = 1
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1:
                _issue(errors, f"算子图存在环: {' -> '.join(map(str, stack + [node, nxt]))}", f"{base_location}[{node}].inputs", "inputs")
            elif state.get(nxt, 0) == 0:
                visit(nxt, stack + [node])
        state[node] = 2

    for seq in sorted(graph):
        if state.get(seq, 0) == 0:
            visit(seq, [])

    expected_input_counts = {"embedding": 0, "segment_pool": 1, "attention": 2, "layer_norm": 1}
    for op in operators:
        count = expected_input_counts.get(op["kind"])
        if count is not None and len(op["inputs"]) != count:
            _issue(
                errors,
                f"{op['kind']} 期望 {count} 个输入，实际 {len(op['inputs'])}",
                f"{base_location}[{op['seq']}].inputs",
                "inputs",
            )

    if errors:
        raise ChainFormatError(errors)
    operators.sort(key=lambda op: op["seq"])
    return operators


def topological_order(operators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seq = {op["seq"]: op for op in operators}
    ready = sorted([op["seq"] for op in operators if not op["inputs"]])
    visited: list[int] = []
    while ready:
        seq = ready.pop(0)
        if seq in visited:
            continue
        if any(ref not in visited for ref in by_seq[seq]["inputs"]):
            ready.append(seq)
            continue
        visited.append(seq)
        for candidate in sorted(by_seq):
            if candidate not in visited and all(ref in visited for ref in by_seq[candidate]["inputs"]):
                if candidate not in ready:
                    ready.append(candidate)
        ready = sorted(set(ready) - set(visited))
    if len(visited) != len(operators):
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "算子图拓扑排序失败，可能存在环", location="operators"))
    return [by_seq[seq] for seq in visited]
