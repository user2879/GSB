from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .chunking import build_blocks
from .config import ConfigBundle, load_config_file, validate_config
from .corpus import load_command_stream, load_corpus_file, load_vocabulary, parse_corpus_line
from .errors import E_COMMAND_INVALID, DiagnosticError, Failure


class CommandState:
    def __init__(
        self,
        config: ConfigBundle,
        records: List[Dict[str, Any]],
        vocabulary: List[str],
        overrides: Optional[Dict[str, List[str]]] = None,
        deleted_blocks: Optional[Set[str]] = None,
        receipt_version: str = "v3",
    ) -> None:
        self.config = config
        self.records = records
        self.vocabulary = vocabulary
        self.overrides = overrides or {}
        self.deleted_blocks = deleted_blocks or set()
        self.receipt_version = receipt_version

    @classmethod
    def from_sample_dir(cls, root: str | Path) -> "CommandState":
        root = Path(root)
        vocabulary = load_vocabulary(root / "config" / "vocabulary.json")
        config = load_config_file(root / "config" / "build_config.yaml")
        corpus = load_corpus_file(root / "data" / "paragraphs.jsonl", vocabulary)
        return cls(config=config, records=corpus.records, vocabulary=vocabulary)

    @property
    def blocks(self) -> List[Dict[str, Any]]:
        return build_blocks(self.records, self.config, self.overrides, self.deleted_blocks)

    def _validate_terms(self, terms: Any, where: str) -> List[str]:
        if not isinstance(terms, list) or not all(isinstance(term, str) and term in set(self.vocabulary) for term in terms):
            raise DiagnosticError(
                E_COMMAND_INVALID,
                f"{where} terms must be a list of controlled vocabulary terms",
                [Failure(E_COMMAND_INVALID, "invalid terms", field=where)],
            )
        return list(terms)

    def apply(self, command: Dict[str, Any]) -> None:
        op = command["op"]
        if op == "append_batch":
            seen = {record["paragraph_id"] for record in self.records}
            new_records: List[Dict[str, Any]] = []
            failures: List[Failure] = []
            for index, paragraph in enumerate(command["paragraphs"]):
                record, line_failures = parse_corpus_line(
                    json.dumps(paragraph, ensure_ascii=False),
                    index + 1,
                    f"command:{command.get('id')}",
                    set(self.vocabulary),
                    seen,
                )
                failures.extend(line_failures)
                if record:
                    new_records.append(record)
            if failures:
                raise DiagnosticError(E_COMMAND_INVALID, "invalid append_batch", failures)
            self.records.extend(new_records)
        elif op == "replace_paragraph":
            paragraph = dict(command["paragraph"])
            target_id = command.get("target_id") or paragraph.get("paragraph_id")
            paragraph["paragraph_id"] = target_id
            seen = {record["paragraph_id"] for record in self.records if record["paragraph_id"] != target_id}
            record, failures = parse_corpus_line(
                json.dumps(paragraph, ensure_ascii=False),
                1,
                f"command:{command.get('id')}",
                set(self.vocabulary),
                seen,
            )
            if failures or record is None:
                raise DiagnosticError(E_COMMAND_INVALID, "invalid replace_paragraph", failures)
            for index, existing in enumerate(self.records):
                if existing["paragraph_id"] == target_id:
                    self.records[index] = record
                    break
            else:
                self.records.append(record)
        elif op == "delete_paragraph":
            target_id = command["target_id"]
            before = len(self.records)
            self.records = [record for record in self.records if record["paragraph_id"] != target_id]
            if len(self.records) == before:
                raise DiagnosticError(E_COMMAND_INVALID, f"paragraph {target_id!r} not found")
        elif op == "replace_block":
            target_id = command["target_id"]
            terms = self._validate_terms(command.get("terms"), target_id)
            if not any(block["block_id"] == target_id for block in self.blocks):
                raise DiagnosticError(E_COMMAND_INVALID, f"block {target_id!r} not found in current chunk layout")
            self.overrides[target_id] = terms
            self.deleted_blocks.discard(target_id)
        elif op == "delete_block":
            target_id = command["target_id"]
            if not any(block["block_id"] == target_id for block in self.blocks):
                raise DiagnosticError(E_COMMAND_INVALID, f"block {target_id!r} not found in current chunk layout")
            self.deleted_blocks.add(target_id)
            self.overrides.pop(target_id, None)
        elif op == "set_config":
            values = dict(command["values"])
            merged = dict(self.config.data)
            merged.update(values)
            self.config = validate_config(merged, path=f"command:{command.get('id')}")
        elif op == "set_threshold":
            merged = dict(self.config.data)
            merged["similarity_threshold"] = command.get("threshold", command.get("value"))
            self.config = validate_config(merged, path=f"command:{command.get('id')}")
        elif op == "set_point_budget":
            merged = dict(self.config.data)
            merged["projection_point_budget"] = int(command.get("budget", command.get("value")))
            self.config = validate_config(merged, path=f"command:{command.get('id')}")
        elif op == "switch_receipt_version":
            version = command.get("target_version", command.get("version"))
            if version not in {"v1", "v2", "v3"}:
                raise DiagnosticError(E_COMMAND_INVALID, "receipt version must be v1, v2, or v3")
            self.receipt_version = version
        else:
            raise DiagnosticError(E_COMMAND_INVALID, f"unsupported command {op!r}")


def load_all_receipts(root: str | Path) -> List[Dict[str, Any]]:
    from .receipts import load_receipt_file

    root = Path(root)
    receipts: List[Dict[str, Any]] = []
    receipt_dir = root / "receipts"
    for version in ("v1", "v2", "v3"):
        path = receipt_dir / f"receipts_{version}.json"
        if path.exists():
            receipts.extend(load_receipt_file(path, forced_version=version))
    return receipts


def load_queries(root: str | Path) -> List[Dict[str, Any]]:
    path = Path(root) / "data" / "queries.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("queries", raw if isinstance(raw, list) else []))


def diff_objects(expected: Any, actual: Any, path: str = "$", limit: int = 20) -> List[Dict[str, Any]]:
    differences: List[Dict[str, Any]] = []

    def walk(left: Any, right: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if float(left) == float(right):
                    return
            differences.append({"path": current, "expected_type": type(left).__name__, "actual_type": type(right).__name__})
            return
        if isinstance(left, dict):
            keys = sorted(set(left) | set(right))
            for key in keys:
                if key not in left or key not in right:
                    differences.append({"path": f"{current}.{key}", "issue": "missing_key"})
                else:
                    walk(left[key], right[key], f"{current}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                differences.append({"path": current, "issue": "length_mismatch", "expected": len(left), "actual": len(right)})
            for index, (item_left, item_right) in enumerate(zip(left, right)):
                walk(item_left, item_right, f"{current}[{index}]")
        else:
            if left != right:
                differences.append({"path": current, "expected": left, "actual": right})

    walk(expected, actual, path)
    return differences


def load_commands(root: str | Path) -> List[Dict[str, Any]]:
    return load_command_stream(Path(root) / "commands" / "command_stream.json")
