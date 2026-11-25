"""Mutable pipeline state and deterministic edit-command application.

Each edit returns the exact sets of changed tokens and changed operator node
ids, which drive precise ledger invalidation. Token indices always refer to
the position in the current token sequence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .config import Chain, OpSpec, PipelineModel, QuantSpec, validate_chain
from .errors import Code, IssueLog
from .exchange import QuantRecord


@dataclass
class EditOutcome:
    changed_tokens: set[int] = field(default_factory=set)
    changed_ops: set[str] = field(default_factory=set)
    config_changed: bool = False
    version_changed: bool = False
    applied: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


class StateEditor:
    def __init__(self, model: PipelineModel, exchange_version: str, log: IssueLog):
        self.model = model
        self.exchange_version = exchange_version
        self.log = log

    def apply(self, edits: list[dict]) -> EditOutcome:
        outcome = EditOutcome()
        for index, edit in enumerate(edits):
            try:
                self._apply_one(edit, outcome)
                outcome.applied.append(edit.get("cmd", f"edit-{index}"))
            except KeyError as exc:  # pragma: no cover - defensive
                outcome.skipped.append({"index": index, "reason": str(exc), "edit": edit})
                self.log.add(Code.E_SCHEMA, f"edits[{index}]", f"missing key {exc}")
        return outcome

    # ------------------------------------------------------------------ #
    def _chain(self, chain_id: int) -> Chain:
        if chain_id not in self.model.chains:
            raise KeyError(f"chain {chain_id} missing")
        return self.model.chains[chain_id]

    def _find_op(self, chain_id: int, seq: int) -> OpSpec:
        for op in self._chain(chain_id).ops:
            if op.seq == seq:
                return op
        raise KeyError(f"c{chain_id}op{seq}")

    def _apply_one(self, edit: dict, outcome: EditOutcome) -> None:
        cmd = edit["cmd"]
        if cmd == "update_op_param":
            op = self._find_op(edit["chain_id"], edit["seq"])
            op.params.update(edit.get("params", {}))
            outcome.changed_ops.add(op.node_id())

        elif cmd == "set_quant":
            op = self._find_op(edit["chain_id"], edit["seq"])
            payload = edit["quant"]
            op.quant = QuantSpec(
                permille=payload.get("permille"),
                width=payload.get("width"),
                scheme=payload.get("scheme"),
                precision_label=payload.get("precision_label", "manual"),
                completeness=payload.get("completeness", "complete"),
                residual=payload.get("residual", 0.0),
                missing=tuple(payload.get("missing", [])),
            )
            outcome.changed_ops.add(op.node_id())

        elif cmd == "add_op":
            chain = self._chain(edit["chain_id"])
            seq = edit["seq"]
            if any(op.seq == seq for op in chain.ops):
                raise KeyError(f"seq {seq} already exists")
            quant_payload = edit.get("quant", {"permille": 1000, "width": 32,
                                               "scheme": "qint32", "precision_label": "manual",
                                               "completeness": "complete"})
            op = OpSpec(
                chain_id=edit["chain_id"],
                seq=seq,
                op_type=edit["op_type"],
                inputs=tuple(sorted(edit.get("inputs", []))),
                params=edit.get("params", {}),
                quant=QuantSpec(
                    permille=quant_payload.get("permille"),
                    width=quant_payload.get("width"),
                    scheme=quant_payload.get("scheme"),
                    precision_label=quant_payload.get("precision_label", "manual"),
                    completeness=quant_payload.get("completeness", "complete"),
                    residual=quant_payload.get("residual", 0.0),
                    missing=tuple(quant_payload.get("missing", [])),
                ),
                source_version=self.exchange_version,
            )
            chain.ops.append(op)
            chain.ops.sort(key=lambda item: item.seq)
            if not validate_chain(chain.chain_id, chain.ops, self.log):
                raise KeyError("chain invalid after add")
            outcome.changed_ops.add(op.node_id())

        elif cmd == "remove_op":
            chain = self._chain(edit["chain_id"])
            target = self._find_op(edit["chain_id"], edit["seq"])
            chain.ops = [op for op in chain.ops if op.seq != edit["seq"]]
            # Re-number remaining seqs so they stay contiguous.
            for new_seq, op in enumerate(sorted(chain.ops, key=lambda item: item.seq)):
                op.seq = new_seq
                op.inputs = tuple(ref for ref in op.inputs if ref != edit["seq"])
            outcome.changed_ops.add(target.node_id())
            outcome.changed_tokens.update(range(len(self.model.tokens)))

        elif cmd == "replace_token":
            index = edit["index"]
            self.model.tokens[index] = edit["token"]
            outcome.changed_tokens.add(index)

        elif cmd == "remove_token":
            index = edit["index"]
            del self.model.tokens[index]
            self._rebuild_segments_after_token_change()
            outcome.changed_tokens.update(range(len(self.model.tokens) + 1))

        elif cmd == "set_segments":
            self.model.segments = [tuple(pair) for pair in edit["segments"]]
            outcome.changed_tokens.update(range(len(self.model.tokens)))

        elif cmd == "set_gear":
            self.model.config.gear = edit["gear"]
            outcome.config_changed = True

        elif cmd == "set_frac_bits":
            self.model.config.frac_bits = int(edit["frac_bits"])
            outcome.config_changed = True

        elif cmd == "set_version":
            self.exchange_version = edit["version"]
            outcome.version_changed = True

    def _rebuild_segments_after_token_change(self) -> None:
        count = len(self.model.tokens)
        if not self.model.segments:
            return
        width = max(1, count // max(1, len(self.model.segments)))
        segments: list[tuple[int, int]] = []
        cursor = 0
        while cursor < count:
            end = min(count, cursor + width)
            segments.append((cursor, end))
            cursor = end
        if not segments:
            segments = [(0, 0)]
        self.model.segments = segments

    def quant_records(self) -> list[QuantRecord]:
        records: list[QuantRecord] = []
        for op in self.model.all_ops():
            records.append(QuantRecord(
                chain_id=op.chain_id,
                seq=op.seq,
                quant=copy.deepcopy(op.quant),
                source_version=op.source_version,
            ))
        return records
