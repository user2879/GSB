"""编辑序列回放：从空白工作区按确定性序列重建统一表示。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .edits import mark_reparse, replace_relation, set_head, shift_heads
from .errors import DepEditError
from .models import Corpus, Sentence


def clone_corpus(corpus: Corpus) -> Corpus:
    import copy

    return Corpus(sentences={sid: copy.deepcopy(sent) for sid, sent in corpus.sentences.items()})


def apply_edit_sequence(corpus: Corpus, sequence: list[dict[str, Any]]) -> dict[str, Any]:
    """对统一表示应用编辑序列，返回回放摘要（含失败但不写入的操作）。"""
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for step in sequence:
        op = step["op"]
        sid = step["sent_id"]
        sent = corpus.sentences[sid]
        try:
            if op == "set_head":
                res = set_head(sent, int(step["word_idx"]), int(step["new_head"]))
            elif op == "replace_relation":
                res = replace_relation(
                    sent,
                    step["old_relation"],
                    step["new_relation"],
                    word_idx=step.get("word_idx"),
                )
            elif op == "shift_heads":
                res = shift_heads(sent, int(step["delta"]))
            elif op == "mark_reparse":
                res = mark_reparse(sent)
            else:
                raise ValueError(f"未知编辑操作 {op}")
            applied.append(
                {
                    "op": res.op,
                    "sent_id": res.sent_id,
                    "affected_words": res.affected_words,
                    "affected_arcs": res.affected_arcs,
                }
            )
        except DepEditError as exc:
            rejected.append(
                {
                    "op": op,
                    "sent_id": sid,
                    "business_code": exc.code,
                    "message": str(exc),
                }
            )
    return {"applied": applied, "rejected": rejected}


def load_sequence(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
