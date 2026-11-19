"""异常注入：对指定句、指定阶段施加真实输入故障。

注入项编号（inject_id）随异常写入失败清单，证明命中并映射业务码。
"""
from __future__ import annotations

import math
import threading
from typing import Any

from .config import AppConfig, INJECTION_BUSINESS_CODES, Injection
from .errors import CODE_TO_EXCEPTION, DepEditError
from .models import Sentence


_CURRENT_INJECTION = threading.local()


def current_inject_id() -> str | None:
    return getattr(_CURRENT_INJECTION, "inject_id", None)


def reset_injection_context() -> None:
    if hasattr(_CURRENT_INJECTION, "inject_id"):
        del _CURRENT_INJECTION.inject_id


class InjectionEngine:
    def __init__(self, injections: list[Injection]):
        self._items: dict[tuple[str, str], list[Injection]] = {}
        for inj in injections:
            self._items.setdefault((inj.sent_id, inj.stage), []).append(inj)

    def has(self, sent_id: str, stage: str) -> bool:
        return bool(self._items.get((sent_id, str(stage))))

    def raise_if_present(self, sent_id: str, stage: str) -> None:
        for inj in self._items.get((sent_id, stage), []):
            raise self._build_exception(inj, sent_id, stage)

    @staticmethod
    def _build_exception(inj: "Injection", sent_id: str, stage: str) -> DepEditError:
        code = INJECTION_BUSINESS_CODES[inj.fault]
        exc_cls = CODE_TO_EXCEPTION[code]
        exc = exc_cls(
            f"注入命中 {inj.inject_id}: 句 {sent_id} 阶段 {stage} 故障 {inj.fault}"
        )
        exc.inject_id = inj.inject_id  # type: ignore[attr-defined]
        return exc

    def matching(self, sent_id: str, stage: str) -> list["Injection"]:
        return list(self._items.get((sent_id, stage), []))

    def raise_if_present_faults(
        self, sent_id: str, stage: str, allowed_faults: set[str]
    ) -> None:
        for inj in self._items.get((sent_id, stage), []):
            if inj.fault in allowed_faults:
                raise self._build_exception(inj, sent_id, stage)

    # ---- 针对真实数据的定向破坏（解析前 / 序列化前） ----

    def corrupt_v1_lines(self, sent_id: str, lines: list[str], cfg: AppConfig) -> list[str]:
        reset_injection_context()
        out: list[str] = []
        for line in lines:
            out.append(line)
        v1_faults = {
            "fixed_width_missing", "score_over_1000",
            "relation_not_in_map", "head_out_of_range",
        }
        for inj in self._items.get((sent_id, "parse"), []):
            if inj.fault not in v1_faults:
                continue
            _CURRENT_INJECTION.inject_id = inj.inject_id
            if inj.fault == "fixed_width_missing":
                # 截断最后一行，造成定宽字段缺失，随后由真实解析器抛 E_FIXED_WIDTH_FIELD。
                if out:
                    out[-1] = out[-1][: max(1, len(out[-1]) - 10)]
            elif inj.fault == "score_over_1000":
                out = [self._rewrite_score(line, cfg, 9999) for line in out]
            elif inj.fault == "relation_not_in_map":
                out = [self._rewrite_relation(line, cfg, "ZZZ") for line in out]
            elif inj.fault == "head_out_of_range":
                out = [self._rewrite_head(line, cfg, 9999) for line in out]
        return out

    @staticmethod
    def _rewrite_head(line: str, cfg: AppConfig, value: int) -> str:
        from .codecs import v1_parse_line, v1_serialize_line

        fields = v1_parse_line(line, cfg.fixed_widths)
        fields["head"] = str(value)
        return v1_serialize_line(fields, cfg.fixed_widths)

    @staticmethod
    def _rewrite_score(line: str, cfg: AppConfig, value: int) -> str:
        from .codecs import v1_parse_line, v1_serialize_line

        fields = v1_parse_line(line, cfg.fixed_widths)
        fields["score"] = str(value)
        return v1_serialize_line(fields, cfg.fixed_widths)

    @staticmethod
    def _rewrite_relation(line: str, cfg: AppConfig, value: str) -> str:
        from .codecs import v1_parse_line, v1_serialize_line

        fields = v1_parse_line(line, cfg.fixed_widths)
        fields["relation"] = value
        return v1_serialize_line(fields, cfg.fixed_widths)

    def corrupt_v2_doc(self, sent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reset_injection_context()
        v2_faults = {"v2_non_finite_score", "head_out_of_range"}
        for inj in self._items.get((sent_id, "parse"), []):
            if inj.fault not in v2_faults:
                # v1 专用故障在 v2 解析路径不适用，直接跳过，避免跨版本误报。
                continue
            _CURRENT_INJECTION.inject_id = inj.inject_id
            arcs = payload["arcs"] if isinstance(payload, dict) else payload
            if inj.fault == "v2_non_finite_score":
                target = inj.target_word or 1
                for arc in arcs:
                    if int(arc["word_idx"]) == target:
                        arc["score"] = math.nan
                if not any(int(a["word_idx"]) == target for a in arcs) and arcs:
                    arcs[0]["score"] = math.inf
            elif inj.fault == "head_out_of_range":
                target = inj.target_word or 1
                for arc in arcs:
                    if int(arc["word_idx"]) == target:
                        arc["head"] = 9999.0
        return payload

    def inject_edit_cycle(self, sent: Sentence) -> Injection | None:
        """对句子制造真实环，交由 validation 抛出 E_CYCLE_DETECTED。"""
        for inj in self._items.get((sent.sent_id, "edit"), []):
            if inj.fault == "cycle_dependency" and len(sent.arcs) >= 2:
                first, second = sent.arcs[0], sent.arcs[1]
                first.head = second.word_idx
                second.head = first.word_idx
                return inj
        return None

    def convert_stage_guard(self, sent_id: str, version: str | None = None) -> None:
        v1_faults = {"score_over_1000", "relation_not_in_map", "fixed_width_missing", "head_out_of_range"}
        v2_faults = {"v2_non_finite_score", "head_out_of_range"}
        if version == "v1":
            allowed = v1_faults
        elif version == "v2":
            allowed = v2_faults
        else:
            self.raise_if_present(sent_id, "convert")
            return
        self.raise_if_present_faults(sent_id, "convert", allowed)


def inject_error_record(inj: Injection, exc: DepEditError) -> dict[str, Any]:
    return {
        "inject_id": inj.inject_id,
        "sent_id": inj.sent_id,
        "stage": inj.stage,
        "fault": inj.fault,
        "business_code": INJECTION_BUSINESS_CODES[inj.fault],
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
