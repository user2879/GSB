"""Blocked embedding table: loading, validation, fingerprints, edits."""

import json
import math

from .errors import E_EMBED_INVALID
from .fingerprints import (block_fingerprint, layout_fingerprint,
                           row_fingerprint)
from .fixedpoint import quantize_vector


class Row:
    __slots__ = ("word_id", "block_index", "slot", "vector", "qvec",
                 "row_fp")

    def __init__(self, word_id, block_index, slot, vector):
        self.word_id = word_id
        self.block_index = block_index
        self.slot = slot
        self.vector = tuple(vector)
        self.qvec = ()
        self.row_fp = ""


class EmbeddingTable:
    """Rows organised in blocks of ``rows_per_block`` rows each."""

    def __init__(self, config):
        self.config = config
        self.rows = {}            # word_id -> Row
        self.blocks = {}          # block_index -> [word_id] in slot order
        self.block_fps = {}       # block_index -> fingerprint
        self.layout_fp = ""
        self.failure_list = []    # validation failures (E_EMBED_INVALID)
        self.failed_word_ids = set()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, config, injector=None):
        table = cls(config)
        seen_ids = set()
        slots = {}
        with open(config.table_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    table._fail(line_no, None, f"bad json: {exc}")
                    continue
                word_id = rec.get("word_id")
                block_index = rec.get("block_index")
                vector = rec.get("vector")
                if injector is not None:
                    vector = injector.corrupt_vector(word_id, vector)
                if not isinstance(word_id, int) or isinstance(word_id, bool):
                    table._fail(line_no, word_id, "word_id must be int")
                    continue
                if not isinstance(block_index, int) or block_index < 0:
                    table._fail(line_no, word_id,
                                "block_index must be int >= 0")
                    continue
                if word_id in seen_ids:
                    table._fail(line_no, word_id, "duplicate word_id")
                    continue
                if (not isinstance(vector, list)
                        or len(vector) != config.dim):
                    table._fail(line_no, word_id,
                                f"vector length != dim {config.dim}")
                    continue
                if not all(isinstance(v, (int, float))
                           and not isinstance(v, bool)
                           and math.isfinite(v) for v in vector):
                    table._fail(line_no, word_id,
                                "vector has non-finite element")
                    continue
                seen_ids.add(word_id)
                slot = slots.get(block_index, 0)
                slots[block_index] = slot + 1
                row = Row(word_id, block_index, slot,
                          [float(v) for v in vector])
                table.rows[word_id] = row
                table.blocks.setdefault(block_index, []).append(word_id)
        table._check_layout()
        table.finalize()
        return table

    def _fail(self, line_no, word_id, reason):
        self.failure_list.append({
            "code": E_EMBED_INVALID,
            "line": line_no,
            "word_id": word_id,
            "reason": reason,
        })
        if isinstance(word_id, int):
            self.failed_word_ids.add(word_id)

    def _check_layout(self):
        expected_rows = self.config.rows_per_block
        indices = sorted(self.blocks)
        if indices != list(range(len(indices))):
            self.failure_list.append({
                "code": E_EMBED_INVALID,
                "line": None,
                "word_id": None,
                "reason": f"block indices not continuous: {indices}",
            })
        for idx in indices:
            count = len(self.blocks[idx])
            if count != expected_rows:
                self.failure_list.append({
                    "code": E_EMBED_INVALID,
                    "line": None,
                    "word_id": None,
                    "reason": (f"block {idx} has {count} rows, "
                               f"expected {expected_rows}"),
                })

    # -------------------------------------------------------- fingerprints
    def finalize(self):
        """(Re)compute layout fingerprint and quantize all rows."""
        self.layout_fp = layout_fingerprint(
            [[idx, len(self.blocks[idx])] for idx in sorted(self.blocks)])
        self.requantize(self.config.quant_level, self.config.frac_bits)

    def requantize(self, quant_level, frac_bits):
        for row in self.rows.values():
            self._quantize_row(row, quant_level, frac_bits)
        self._recompute_block_fps()

    def _quantize_row(self, row, quant_level, frac_bits):
        row.qvec = quantize_vector(row.vector, frac_bits, quant_level)
        row.row_fp = row_fingerprint(self.config.dim, quant_level,
                                     frac_bits, row.qvec)

    def _recompute_block_fps(self):
        self.block_fps = {}
        for idx in sorted(self.blocks):
            fps = [self.rows[wid].row_fp for wid in self.blocks[idx]]
            self.block_fps[idx] = block_fingerprint(
                idx, self.config.rows_per_block, fps)

    def block_layout_pairs(self):
        return [[idx, len(self.blocks[idx])] for idx in sorted(self.blocks)]

    # ----------------------------------------------------------------- edits
    def _validate_new_vector(self, word_id, vector):
        if (not isinstance(vector, list)
                or len(vector) != self.config.dim):
            raise ValueError(f"vector length != dim {self.config.dim}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in vector):
            raise ValueError("vector has non-finite element")
        return [float(v) for v in vector]

    def replace_row(self, word_id, vector):
        if word_id not in self.rows:
            raise KeyError(f"word_id {word_id} not in table")
        vector = self._validate_new_vector(word_id, vector)
        row = self.rows[word_id]
        row.vector = tuple(vector)
        self._quantize_row(row, self.config.quant_level,
                           self.config.frac_bits)
        self._recompute_block_fps()
        return {word_id}

    def replace_block(self, block_index, new_rows):
        if block_index not in self.blocks:
            raise KeyError(f"block {block_index} not in table")
        if len(new_rows) != self.config.rows_per_block:
            raise ValueError("replacement block row count mismatch")
        old_ids = list(self.blocks[block_index])
        new_ids = []
        for slot, item in enumerate(new_rows):
            wid = item["word_id"]
            if wid in self.rows and wid not in old_ids:
                raise ValueError(f"word_id {wid} already exists")
            vector = self._validate_new_vector(wid, item["vector"])
            new_ids.append(wid)
        for wid in old_ids:
            del self.rows[wid]
        self.blocks[block_index] = []
        for slot, item in enumerate(new_rows):
            wid = item["word_id"]
            row = Row(wid, block_index, slot,
                      self._validate_new_vector(wid, item["vector"]))
            self._quantize_row(row, self.config.quant_level,
                               self.config.frac_bits)
            self.rows[wid] = row
            self.blocks[block_index].append(wid)
        self._recompute_block_fps()
        return set(old_ids) | set(new_ids)

    def append_block(self, new_rows):
        if len(new_rows) != self.config.rows_per_block:
            raise ValueError("appended block row count mismatch")
        block_index = (max(self.blocks) + 1) if self.blocks else 0
        added = []
        for item in new_rows:
            wid = item["word_id"]
            if wid in self.rows:
                raise ValueError(f"word_id {wid} already exists")
            added.append(wid)
        self.blocks[block_index] = []
        for slot, item in enumerate(new_rows):
            wid = item["word_id"]
            row = Row(wid, block_index, slot,
                      self._validate_new_vector(wid, item["vector"]))
            self._quantize_row(row, self.config.quant_level,
                               self.config.frac_bits)
            self.rows[wid] = row
            self.blocks[block_index].append(wid)
        self.layout_fp = layout_fingerprint(self.block_layout_pairs())
        self._recompute_block_fps()
        return block_index, set(added)

    # -------------------------------------------------------------- serialize
    def to_jsonl_records(self):
        records = []
        for idx in sorted(self.blocks):
            for wid in self.blocks[idx]:
                row = self.rows[wid]
                records.append({
                    "word_id": row.word_id,
                    "block_index": row.block_index,
                    "vector": list(row.vector),
                })
        return records

    def validate_stage_artifact(self, version):
        return {
            "stage": "validate",
            "version": version,
            "table_layout_fp": self.layout_fp,
            "row_count": len(self.rows),
            "block_count": len(self.blocks),
            "failure_list": list(self.failure_list),
        }

    def fingerprint_stage_artifact(self, version):
        return {
            "stage": "fingerprint",
            "version": version,
            "quant_level": self.config.quant_level,
            "frac_bits": self.config.frac_bits,
            "dim": self.config.dim,
            "layout_fp": self.layout_fp,
            "rows": {str(wid): {
                "row_fp": row.row_fp,
                "block_index": row.block_index,
                "slot": row.slot,
            } for wid, row in sorted(self.rows.items())},
            "blocks": {str(idx): self.block_fps[idx]
                       for idx in sorted(self.block_fps)},
        }
