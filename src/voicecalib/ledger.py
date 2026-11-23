"""校准账本：缓存键、精确失效与复用判定。

每个阶段的缓存键均由上游内容指纹、校准参数指纹与定点位宽档位派生。
账本持久化为 JSON，支持增量运行时的命中复用与下游失效传播。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import fingerprint, short_key


@dataclass
class LedgerEntry:
    stage: str
    round: int
    cache_key: str
    audio_file: str
    audio_hash: str
    upstream_keys: list[str]
    param_fingerprint: str
    frac_bits: int
    quant_gear: int
    status: str = "computed"          # computed / failed / pending
    error_code: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Ledger:
    entries: dict[str, LedgerEntry] = field(default_factory=dict)
    # 轮次 -> 各阶段缓存键，便于按轮次传播失效。
    round_index: dict[int, dict[str, str]] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)

    def key_for(self, stage: str, rnd: int) -> str | None:
        return self.round_index.get(rnd, {}).get(stage)

    def put(self, entry: LedgerEntry) -> None:
        self.entries[entry.cache_key] = entry
        self.round_index.setdefault(entry.round, {})[entry.stage] = entry.cache_key

    def mark_failed(self, stage: str, rnd: int, code: str) -> None:
        key = self.key_for(stage, rnd)
        if key and key in self.entries:
            self.entries[key].status = "failed"
            self.entries[key].error_code = code

    def mark_pending(self, stage: str, rnd: int, code: str) -> None:
        key = self.key_for(stage, rnd)
        if key and key in self.entries:
            self.entries[key].status = "pending"
            self.entries[key].error_code = code

    def record_decision(self, rnd: int, stage: str, action: str,
                        reason: str, cache_key: str | None) -> None:
        self.decisions.append({
            "round": rnd,
            "stage": stage,
            "action": action,                 # reuse / recompute
            "reason": reason,
            "cache_key": cache_key,
        })

    def plan_round_status(self, current_audio: dict[int, tuple[str, str]],
                          param_fingerprint: str, frac_bits: int,
                          quant_gear: int,
                          calib_param_fingerprint: str | None = None
                          ) -> dict[int, dict[str, str]]:
        """返回每轮每阶段动作：reuse 或 recompute（含判定依据）。

        evidence 阶段比对证据参数指纹，calibration 阶段比对校准参数指纹。
        """
        calib_fp = calib_param_fingerprint or param_fingerprint
        plan: dict[int, dict[str, str]] = {}
        for rnd in sorted(current_audio):
            audio_file, audio_hash = current_audio[rnd]
            round_plan: dict[str, str] = {}
            evidence_key = self.key_for("evidence", rnd)
            evidence_entry = self.entries.get(evidence_key) if evidence_key else None
            if evidence_entry is None:
                round_plan["evidence"] = "recompute:no_cache_entry"
            elif evidence_entry.audio_hash != audio_hash:
                round_plan["evidence"] = "recompute:audio_content_changed"
            elif (evidence_entry.param_fingerprint != param_fingerprint
                  or evidence_entry.frac_bits != frac_bits
                  or evidence_entry.quant_gear != quant_gear):  # 证据参数指纹
                round_plan["evidence"] = "recompute:param_or_fixedpoint_changed"
            elif evidence_entry.status == "failed":
                round_plan["evidence"] = "recompute:prior_failure"
            else:
                round_plan["evidence"] = "reuse:fingerprint_match"

            calib_key = self.key_for("calibration", rnd)
            calib_entry = self.entries.get(calib_key) if calib_key else None
            if not round_plan["evidence"].startswith("reuse"):
                round_plan["calibration"] = "recompute:evidence_invalidated"
            elif calib_entry is None:
                round_plan["calibration"] = "recompute:no_cache_entry"
            elif (calib_entry.param_fingerprint != calib_fp
                  or calib_entry.frac_bits != frac_bits
                  or calib_entry.quant_gear != quant_gear):
                round_plan["calibration"] = "recompute:param_or_fixedpoint_changed"
            elif calib_entry.status == "failed":
                round_plan["calibration"] = "recompute:prior_failure"
            else:
                round_plan["calibration"] = "reuse:fingerprint_match"

            attr_key = self.key_for("attribution", rnd)
            attr_entry = self.entries.get(attr_key) if attr_key else None
            if not round_plan["calibration"].startswith("reuse"):
                round_plan["attribution"] = "recompute:calibration_invalidated"
            elif attr_entry is None:
                round_plan["attribution"] = "recompute:no_cache_entry"
            elif attr_entry.status == "failed":
                round_plan["attribution"] = "recompute:prior_failure"
            else:
                round_plan["attribution"] = "reuse:fingerprint_match"

            plan[rnd] = round_plan

        # 校准链传播：校准按轮次串行依赖前驱，一旦某轮校准失效，
        # 其下游所有轮次的校准与归属级联重算（证据仍可命中复用）。
        ordered = sorted(plan)
        for idx, rnd in enumerate(ordered[1:], start=1):
            prev = ordered[idx - 1]
            if not plan[prev]["calibration"].startswith("reuse"):
                if plan[rnd]["calibration"].startswith("reuse"):
                    plan[rnd]["calibration"] =                         "recompute:upstream_calibration_invalidated"
                plan[rnd]["attribution"] =                     "recompute:upstream_calibration_invalidated"
        return plan

    def prune_missing_rounds(self, current_rounds: set[int]) -> list[int]:
        """音频被删除的轮次：连同下游全部移除，返回被移除轮次。"""
        removed = sorted(rnd for rnd in self.round_index if rnd not in current_rounds)
        if not removed:
            return []
        first_removed = removed[0]
        cascade = sorted(rnd for rnd in self.round_index if rnd >= first_removed)
        for rnd in cascade:
            for stage, key in list(self.round_index.pop(rnd, {}).items()):
                self.entries.pop(key, None)
        self.decisions = [
            d for d in self.decisions if d["round"] not in set(cascade)
        ]
        return cascade

    def save(self, path: str | Path) -> None:
        payload = {
            "ledger_version": 1,
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "round_index": {str(k): v for k, v in self.round_index.items()},
            "decisions": self.decisions,
        }
        Path(path).write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger = cls()
        for item in data.get("entries", []):
            ledger.entries[item["cache_key"]] = LedgerEntry(**item)
        for rnd_text, stage_map in data.get("round_index", {}).items():
            ledger.round_index[int(rnd_text)] = dict(stage_map)
        ledger.decisions = data.get("decisions", [])
        return ledger


def make_evidence_entry(rnd: int, artifact, param_fingerprint: str,
                        config) -> LedgerEntry:
    return LedgerEntry(
        stage="evidence",
        round=rnd,
        cache_key=artifact.cache_key,
        audio_file=artifact.file,
        audio_hash=artifact.audio_hash,
        upstream_keys=[artifact.audio_hash],
        param_fingerprint=param_fingerprint,
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
    )


def make_calib_entry(rnd: int, row, config) -> LedgerEntry:
    return LedgerEntry(
        stage="calibration",
        round=rnd,
        cache_key=row.cache_key,
        audio_file="",
        audio_hash="",
        upstream_keys=([row.evidence_key]
                       + ([f"calib:{row.prev_round}"]
                          if row.prev_round is not None else [])),
        param_fingerprint=config.param_fingerprint,
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
    )


def make_attr_entry(rnd: int, attribution, config) -> LedgerEntry:
    return LedgerEntry(
        stage="attribution",
        round=rnd,
        cache_key=attribution.cache_key,
        audio_file="",
        audio_hash="",
        upstream_keys=[f"calib:{attribution.calib_round}"],
        param_fingerprint=short_key("attr", config.adjacent_tolerance,
                                    dict(sorted(config.speaker_anchors_hz.items()))),
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
    )
