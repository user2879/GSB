"""Deterministic edit command stream applied to in-memory request state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DecodeConfig
from .errors import BizCode, BizError
from .models import Request, CandidateSequence, Step


@dataclass
class Command:
    command: str
    request_id: str
    seq_id: int | None = None
    step_index: int | None = None
    logprobs: dict[int, float] | None = None
    seed_token: int | None = None
    channel: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p_mille: int | None = None
    width: int | None = None
    tier: str | None = None
    exchange_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class CommandApplier:
    """Applies commands and reports which (seq, step) conclusions change."""

    def __init__(self, requests: list[Request], config: DecodeConfig,
                 exchange_version: str = "v2"):
        self.requests = {r.request_id: r for r in requests}
        self.config = config
        self.exchange_version = exchange_version
        self.config_changed = False
        # Deterministic list of affected (request_id, seq_id, step_index).
        self.affected: list[tuple[str, int, int]] = []

    def _req(self, command: Command) -> Request:
        req = self.requests.get(command.request_id)
        if req is None:
            raise _biz(BizCode.UNKNOWN_REQUEST,
                       f"request={command.request_id}",
                       "edit targets unknown request")
        return req

    def _seq(self, command: Command) -> CandidateSequence:
        req = self._req(command)
        seq = req.sequence_map().get(command.seq_id)
        if seq is None:
            raise _biz(BizCode.UNKNOWN_REQUEST,
                       f"request={command.request_id}/sequence={command.seq_id}",
                       "edit targets unknown sequence")
        return seq

    def apply_many(self, commands: list[Command]) -> None:
        for command in commands:
            self.apply(command)

    def apply(self, command: Command) -> None:
        handler = getattr(self, f"_on_{command.command}", None)
        if handler is None:
            raise _biz(BizCode.MISSING_FIELD, f"command={command.command}",
                       "unknown edit command")
        handler(command)

    def _on_add_sequence(self, c: Command) -> None:
        req = self._req(c)
        if c.seq_id in req.sequence_map():
            raise _biz(BizCode.DUPLICATE_SEQUENCE,
                       f"request={c.request_id}/sequence={c.seq_id}",
                       "sequence already exists")
        req.sequences.append(CandidateSequence(
            seq_id=c.seq_id, seed_token=c.seed_token or 0,
            channel=c.channel or "SAMPLE",
            temperature=c.temperature if c.temperature is not None else 1.0,
            steps=[]))

    def _on_append_step(self, c: Command) -> None:
        seq = self._seq(c)
        existing = seq.step_map()
        if c.step_index in existing:
            raise _biz(BizCode.DUPLICATE_SEQUENCE,
                       f"step={c.step_index}", "step already exists")
        expected = (max(existing) + 1) if existing else 1
        if c.step_index != expected:
            raise _biz(BizCode.STEP_HOLE, f"step={c.step_index}",
                       f"append must use next index {expected}")
        seq.steps.append(Step(index=c.step_index,
                              logprobs=_coerce_logprobs(c.logprobs)))
        self.affected.append((c.request_id, c.seq_id, c.step_index))

    def _on_replace_step(self, c: Command) -> None:
        seq = self._seq(c)
        if c.step_index not in seq.step_map():
            raise _biz(BizCode.UNKNOWN_REQUEST, f"step={c.step_index}",
                       "replace targets missing step")
        seq.step_map()[c.step_index].logprobs = _coerce_logprobs(c.logprobs)
        self.affected.append((c.request_id, c.seq_id, c.step_index))

    def _on_set_temperature(self, c: Command) -> None:
        self._seq(c).temperature = float(c.temperature)  # type: ignore[arg-type]
        self._mark_seq_steps(c)

    def _on_set_top_p(self, c: Command) -> None:
        self.config = self.config.with_replacements(top_p_mille=c.top_p_mille)
        self.config_changed = True

    def _on_set_top_k(self, c: Command) -> None:
        self.config = self.config.with_replacements(top_k=c.top_k)
        self.config_changed = True

    def _on_set_quantization(self, c: Command) -> None:
        changes = {}
        if c.width is not None:
            changes["width"] = c.width
        if c.tier is not None:
            changes["tier"] = c.tier
        self.config = self.config.with_replacements(**changes)
        self.config_changed = True

    def _on_set_exchange_version(self, c: Command) -> None:
        if c.exchange_version not in ("v1", "v2"):
            raise _biz(BizCode.V2_UNKNOWN_VERSION,
                       f"exchange={c.exchange_version}", "bad exchange version")
        self.exchange_version = c.exchange_version

    def _mark_seq_steps(self, c: Command) -> None:
        seq = self._seq(c)
        for step in seq.steps:
            self.affected.append((c.request_id, c.seq_id, step.index))


def _biz(code: int, location: str, message: str) -> BizError:
    return BizError(code, location, message)


def _coerce_logprobs(logprobs: dict | None) -> dict[int, float]:
    if not logprobs:
        return {}
    return {int(token): float(value) for token, value in logprobs.items()}
