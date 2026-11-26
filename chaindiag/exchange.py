from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any

from .errors import E_CHAIN_FORMAT, ChainFormatError, Issue
from .fixedpoint import INT_BITS_TO_SCHEME


# Fixed character slices.  The widths are part of the exchange contract.
E1_FIELDS = (
    ("version", 0, 2, "E1"),
    ("op_seq", 2, 6, None),
    ("scale_permille", 6, 12, None),
    ("int_bits", 12, 15, None),
)
E1_LINE_WIDTH = 15
E1_VERSION = "E1"


def parse_e1_text(text: str, location: str = "chain.e1.txt") -> list[dict[str, Any]]:
    issues: list[Issue] = []
    operators: list[dict[str, Any]] = []
    lines = text.splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        line_loc = f"{location}:{line_number}"
        values: dict[str, Any] = {}
        for field_name, start, end, literal in E1_FIELDS:
            field_loc = f"{line_loc}:{field_name}"
            if len(line) < end:
                issues.append(
                    Issue(
                        E_CHAIN_FORMAT,
                        f"定宽字段宽度不足: {field_name} 需要至少 {end} 字符，实际 {len(line)}",
                        location=line_loc,
                        line=line_number,
                        field=field_name,
                        context={"expected_end": end, "actual_length": len(line)},
                    )
                )
                continue
            piece = line[start:end]
            if literal is not None:
                if piece != literal:
                    issues.append(
                        Issue(
                            E_CHAIN_FORMAT,
                            f"版本字段解析失败: 期望 {literal!r}，实际 {piece!r}",
                            location=field_loc,
                            line=line_number,
                            field=field_name,
                        )
                    )
                values[field_name] = piece
                continue
            if not piece.isdigit():
                issues.append(
                    Issue(
                        E_CHAIN_FORMAT,
                        f"定宽整数字段含非数字内容: {piece!r}",
                        location=field_loc,
                        line=line_number,
                        field=field_name,
                    )
                )
                continue
            values[field_name] = int(piece)

        if len(line) > E1_LINE_WIDTH:
            issues.append(
                Issue(
                    E_CHAIN_FORMAT,
                    f"定宽行宽度超出契约: 期望 {E1_LINE_WIDTH}，实际 {len(line)}",
                    location=line_loc,
                    line=line_number,
                    field="record",
                )
            )

        if {"op_seq", "scale_permille", "int_bits"} <= values.keys():
            operators.append(
                {
                    "seq": values["op_seq"],
                    "scale_permille": values["scale_permille"],
                    "int_bits": values["int_bits"],
                    "source_line": line_number,
                    "source_version": "e1",
                }
            )

    if issues:
        raise ChainFormatError(issues)
    if not operators:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, "e1 文件没有有效算子记录", location=location, field="body"))
    return operators


def format_e1_line(op: dict[str, Any]) -> str:
    seq = int(op.get("seq", op.get("op_seq", 0)))
    permille = int(op.get("scale_permille", 1000))
    bits = int(op.get("int_bits", 16))
    if not 0 <= seq <= 9999:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, f"op_seq={seq} 超出 e1 定宽字段", field="op_seq"))
    if not 0 <= permille <= 999999:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, f"scale_permille={permille} 超出 e1 定宽字段", field="scale_permille"))
    if not 0 <= bits <= 999:
        raise ChainFormatError(Issue(E_CHAIN_FORMAT, f"int_bits={bits} 超出 e1 定宽字段", field="int_bits"))
    return f"{E1_VERSION}{seq:04d}{permille:06d}{bits:03d}"


def _operators_from_e2(e2: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(e2, list):
        return e2
    if isinstance(e2, dict) and isinstance(e2.get("operators"), list):
        return e2["operators"]
    if isinstance(e2, dict) and isinstance(e2.get("chains"), list):
        active = e2.get("active_chain_seq")
        for chain in e2["chains"]:
            if active is None or chain.get("seq", chain.get("chain_seq")) == active:
                return chain.get("operators", [])
    raise ChainFormatError(Issue(E_CHAIN_FORMAT, "e2 JSON 不包含 operators", location="$.operators"))


def e2_to_e1(e2: dict[str, Any] | list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    operators = _operators_from_e2(e2)
    lines: list[str] = []
    bypass_entries: dict[str, Any] = {}
    for raw_op in operators:
        op = copy.deepcopy(raw_op)
        seq = int(op.get("seq", op.get("op_seq", len(lines))))
        if "scale_float" in op and op["scale_float"] is not None:
            original_scale = Fraction(str(op["scale_float"]))
        else:
            original_scale = Fraction(int(op.get("scale_permille", 1000)), 1000)
        rounded_permille = int(original_scale * 1000 + Fraction(1, 2))
        e1_op = {
            "seq": seq,
            "scale_permille": rounded_permille,
            "int_bits": int(op.get("int_bits", 16)),
        }
        lines.append(format_e1_line(e1_op))
        residual = original_scale - Fraction(rounded_permille, 1000)
        bypass_entries[str(seq)] = {
            "schema": "chaindiag-e1-bypass-entry-v1",
            "e1_fields": e1_op,
            "original_scale_float": float(original_scale),
            "original_scale_rational": f"{original_scale.numerator}/{original_scale.denominator}",
            "scale_round_residual": {
                "exact": f"{residual.numerator}/{residual.denominator}",
                "float": float(residual),
            },
            "precision_tag": (op.get("exchange_meta") or {}).get("precision_tag"),
            "quant_scheme": op.get("quant_scheme"),
            "overwritten_references": {
                "inputs": op.get("inputs", []),
                "attention_snapshot": op.get("attention_snapshot")
                or (op.get("params") or {}).get("attention_snapshot"),
            },
            "operator": op,
        }

    sidecar = {
        "schema": "chaindiag-e1-bypass-v1",
        "format": "chain-e1-bypass",
        "version": "bypass-1.0",
        "entries": bypass_entries,
    }
    return "\n".join(lines) + "\n", sidecar


def e1_to_e2(text: str, sidecar: dict[str, Any] | None = None, chain_seq: int = 1) -> dict[str, Any]:
    parsed = parse_e1_text(text)
    entries = (sidecar or {}).get("entries", {}) if isinstance(sidecar, dict) else {}
    operators: list[dict[str, Any]] = []
    for raw in parsed:
        seq = raw["seq"]
        permille = raw["scale_permille"]
        bits = raw["int_bits"]
        rational = Fraction(permille, 1000)
        scheme = INT_BITS_TO_SCHEME.get(bits, f"qint{bits}")
        op: dict[str, Any] = {
            "seq": seq,
            "kind": "unknown",
            "name": f"op{seq}",
            "inputs": [],
            "scale_permille": permille,
            "scale_float": float(rational),
            "scale_rational": f"{rational.numerator}/{rational.denominator}",
            "int_bits": bits,
            "quant_scheme": scheme,
            "exchange_version": "e2",
            "exchange_meta": {
                "precision_tag": f"e1-exact:{permille}/1000",
                "e1_precision_tag": f"e1-exact:{permille}/1000",
            },
            "source_line": raw.get("source_line"),
        }
        entry = entries.get(str(seq)) or entries.get(seq)
        if entry:
            restored = copy.deepcopy(entry.get("operator", {}))
            # Restore all e2-only/covered fields, while retaining parsed e1 facts.
            for key, value in restored.items():
                if key not in ("seq", "scale_permille", "int_bits", "source_line"):
                    op[key] = value
            op["restored_from_bypass"] = True
            op["bypass_residual"] = entry.get("scale_round_residual")
            op["exchange_meta"] = dict(op.get("exchange_meta") or {})
            if entry.get("precision_tag"):
                op["exchange_meta"]["restored_precision_tag"] = entry["precision_tag"]
            op["exchange_meta"]["bypass_entry_schema"] = entry.get("schema")
        operators.append(op)

    return {
        "format": "chain-e2",
        "version": "e2-1.0",
        "chain_seq": chain_seq,
        "exchange_version": "e2",
        "operators": operators,
        "bypass_used": bool(entries),
    }


def roundtrip_e2(e2: dict[str, Any]) -> dict[str, Any]:
    e1_text, sidecar = e2_to_e1(e2)
    restored = e1_to_e2(e1_text, sidecar)
    return {"e1_text": e1_text, "sidecar": sidecar, "restored_e2": restored}
