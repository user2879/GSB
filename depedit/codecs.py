"""v1 定宽文本与 v2 JSON 的解析 / 序列化。

v1：定宽字段拼接，1 起始序号，头词缺失用固定填充串，千分比整数得分，短码关系。
v2：JSON，可配置编号起点（默认 0 起始浮点偏移），命名关系，0~1 浮点得分，精度标签。
统一表示：1 起始序号，头词 0 表示根缺失，命名关系，0~1 浮点得分。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import AppConfig, FIXED_ORDER
from .errors import (
    EFixedWidthField,
    EHeadOutOfRange,
    ERelationUnknown,
    EScoreNonFinite,
    EScoreRange,
)
from .models import Arc, HEAD_FILLER, MISSING_HEAD, Sentence, Word


def _pad(value: str, width: int) -> str:
    if len(value) > width:
        # 定宽列无法容纳时由调用方在降级清单保留原值；此处安全截断以维持行结构。
        value = value[:width]
    return value.ljust(width)



def self_widths_relation(cfg: "AppConfig") -> int:
    return cfg.fixed_widths["relation"]


# ---------------- v1 定宽 ----------------

def v1_parse_line(line: str, widths: dict[str, int]) -> dict[str, str]:
    if not line.rstrip("\n"):
        raise EFixedWidthField("空行无法按定宽切分")
    fields: dict[str, str] = {}
    cursor = 0
    for name in FIXED_ORDER:
        width = widths[name]
        if cursor + width > len(line.rstrip("\n")):
            raise EFixedWidthField(
                f"定宽行字段缺失：需要字段 {name}（起始列 {cursor}，宽度 {width}）"
            )
        fields[name] = line[cursor:cursor + width].strip()
        cursor += width
    return fields


def v1_serialize_line(fields: dict[str, str], widths: dict[str, int]) -> str:
    return "".join(_pad(str(fields[name]), widths[name]) for name in FIXED_ORDER)


def v1_parse_document(
    text: str,
    cfg: AppConfig,
    v1_to_named: dict[str, str],
    *,
    strict_relation: bool = True,
) -> list[Sentence]:
    grouped: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        fields = v1_parse_line(raw_line, cfg.fixed_widths)
        sid = fields["sent_id"]
        if sid not in grouped:
            grouped[sid] = []
            order.append(sid)
        grouped[sid].append(fields)

    sentences: list[Sentence] = []
    for sid in order:
        rows = sorted(grouped[sid], key=lambda r: int(r["word_idx"]))
        words: list[Word] = []
        arcs: list[Arc] = []
        for row in rows:
            idx = int(row["word_idx"])
            words.append(Word(idx=idx, form=row["form"], pos=row["pos"]))
            head_token = row["head"]
            if head_token == cfg.head_filler:
                head = MISSING_HEAD
            else:
                head = int(head_token)
            permille = int(row["score"])
            if permille < 0 or permille > 1000:
                raise EScoreRange(f"句 {sid} 词 {idx} 千分比得分 {permille} 超出 0..1000")
            code = row["relation"]
            if strict_relation and code not in v1_to_named:
                raise ERelationUnknown(f"句 {sid} 词 {idx} 关系短码 {code} 不在映射表内")
            named = v1_to_named.get(code, code)
            score = permille / 1000.0
            precision = "permille" if permille % 10 else "permille-exact"
            arcs.append(
                Arc(
                    word_idx=idx,
                    head=head,
                    relation=named,
                    score=score,
                    precision=precision,
                    source_version="v1",
                    raw={
                        "head_field": head_token,
                        "relation_code": code,
                        "permille": permille,
                        "fixed_line": v1_serialize_line(row, cfg.fixed_widths),
                    },
                    referenced_words=[idx] + ([head] if head else []),
                )
            )
        sentences.append(
            Sentence(
                sent_id=sid,
                words=words,
                arcs=arcs,
                source_version="v1",
                raw_meta={"row_count": len(rows)},
            )
        )
    return sentences


def v1_serialize_document(
    sentences: list[Sentence],
    cfg: AppConfig,
    named_to_v1: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """返回定宽文本与降级损失清单（关系无法折回时保留原名）。"""
    lines: list[str] = []
    losses: list[dict[str, Any]] = []
    for sent in sorted(sentences, key=lambda s: s.sent_id):
        for arc in sorted(sent.arcs, key=lambda a: a.word_idx):
            head = arc.normalized_head()
            head_field = cfg.head_filler if head == MISSING_HEAD else str(head)
            code = named_to_v1.get(arc.relation)
            relation_field = code if code is not None else arc.relation
            rounded = round(arc.score * 1000.0)
            residual = arc.score - rounded / 1000.0
            if code is None:
                losses.append(
                    {
                        "sent_id": sent.sent_id,
                        "word_idx": arc.word_idx,
                        "type": "unmapped_relation",
                        "original_relation": arc.relation,
                        "fallback_field": relation_field,
                        "truncated": len(arc.relation) > self_widths_relation(cfg),
                    }
                )
            if abs(residual) > 1e-12:
                losses.append(
                    {
                        "sent_id": sent.sent_id,
                        "word_idx": arc.word_idx,
                        "type": "truncated_residual",
                        "residual": residual,
                        "rounded_permille": rounded,
                        "precision_label": arc.precision,
                    }
                )
            # 仅当 v2 起点非 1 起始时才真正发生编号折算，需记录降级信息。
            if cfg.v2_head_origin != cfg.unified_origin and head != MISSING_HEAD:
                losses.append(
                    {
                        "sent_id": sent.sent_id,
                        "word_idx": arc.word_idx,
                        "type": "origin_adjustment",
                        "detail": f"v2_origin={cfg.v2_head_origin} -> unified {cfg.unified_origin}-based",
                        "unified_head": head,
                    }
                )
            fields = {
                "sent_id": sent.sent_id,
                "word_idx": str(arc.word_idx),
                "form": sent.word_by_idx(arc.word_idx).form,
                "pos": sent.word_by_idx(arc.word_idx).pos,
                "head": head_field,
                "relation": relation_field,
                "score": str(rounded),
            }
            lines.append(v1_serialize_line(fields, cfg.fixed_widths))
    return "\n".join(lines) + ("\n" if lines else ""), losses


# ---------------- v2 JSON ----------------

def v2_parse_document(
    obj: dict[str, Any],
    cfg: AppConfig,
    named_relations: set[str],
    *,
    strict_relation: bool = True,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for sid in sorted(obj.keys()):
        payload = obj[sid]
        arc_items = payload["arcs"] if isinstance(payload, dict) else payload
        words_meta = payload.get("words") if isinstance(payload, dict) else None
        words: list[Word] = []
        arcs: list[Arc] = []
        for item in sorted(arc_items, key=lambda a: int(a["word_idx"])):
            idx = int(item["word_idx"])
            if words_meta:
                wm = words_meta[idx - 1]
                words.append(Word(idx=idx, form=wm["form"], pos=wm["pos"]))
            else:
                words.append(Word(idx=idx, form=item.get("form", f"w{idx}"), pos=item.get("pos", "X")))
            score = float(item["score"])
            if not math.isfinite(score):
                raise EScoreNonFinite(f"句 {sid} 词 {idx} v2 得分非有限数: {item['score']}")
            raw_head = item["head"]
            if raw_head is None:
                head = MISSING_HEAD
            else:
                float_head = float(raw_head)
                # 折算为统一 1 起始：0 起始需 +1；1 起始直接取整。
                head = int(round(float_head)) + (1 - cfg.v2_head_origin)
                if head < 0:
                    raise EHeadOutOfRange(f"句 {sid} 词 {idx} 折算后头词 {head} 非法")
            relation = item["relation"]
            if strict_relation and relation not in named_relations:
                raise ERelationUnknown(f"句 {sid} 词 {idx} 命名关系 {relation} 不在映射表内")
            arcs.append(
                Arc(
                    word_idx=idx,
                    head=head,
                    relation=relation,
                    score=score,
                    precision=str(item.get("precision", "float")),
                    source_version="v2",
                    raw={"head_offset": raw_head, "score_raw": item["score"]},
                    referenced_words=[idx] + ([head] if head else []),
                )
            )
        sentences.append(
            Sentence(
                sent_id=sid,
                words=words,
                arcs=arcs,
                source_version="v2",
                raw_meta={"arc_count": len(arcs)},
            )
        )
    return sentences


def v2_serialize_document(
    sentences: list[Sentence], cfg: AppConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回 v2 主文档与旁路元数据（缺失头词显式空值记录）。"""
    doc: dict[str, Any] = {}
    sidecar: dict[str, Any] = {"null_heads": [], "origin": cfg.v2_head_origin}
    for sent in sorted(sentences, key=lambda s: s.sent_id):
        arcs_out: list[dict[str, Any]] = []
        words_out = [{"form": w.form, "pos": w.pos} for w in sent.words]
        for arc in sorted(sent.arcs, key=lambda a: a.word_idx):
            head = arc.normalized_head()
            if head == MISSING_HEAD:
                head_out: float | None = None
                sidecar["null_heads"].append(
                    {"sent_id": sent.sent_id, "word_idx": arc.word_idx, "v1_filler": cfg.head_filler}
                )
            else:
                # 统一 1 起始 -> v2 配置起点。
                head_out = float(head - (1 - cfg.v2_head_origin))
            arcs_out.append(
                {
                    "word_idx": arc.word_idx,
                    "head": head_out,
                    "relation": arc.relation,
                    "score": round(float(arc.score), cfg.score_decimals),
                    "precision": arc.precision,
                }
            )
        doc[sent.sent_id] = {"words": words_out, "arcs": arcs_out}
    return doc, sidecar


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
