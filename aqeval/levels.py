"""Quality-level code/name mapping and threshold resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LEVEL_CODE_PATTERN, LEVEL_NAME_PATTERN
from .errors import FormatError, LevelMapError
from .util import read_json, sha256_json


@dataclass(frozen=True)
class LevelEntry:
    code: str
    name: str
    min_total_permille: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "min_total_permille": self.min_total_permille,
        }


class LevelMap:
    def __init__(self, entries: list[LevelEntry]) -> None:
        self.entries = sorted(entries, key=lambda entry: (entry.min_total_permille, entry.code))
        self._validate()
        self._by_code = {entry.code: entry for entry in self.entries}
        self._by_name = {entry.name: entry for entry in self.entries}

    def _validate(self) -> None:
        if not self.entries:
            raise LevelMapError("level map must contain at least one level")
        codes: set[str] = set()
        names: set[str] = set()
        thresholds: set[int] = set()
        for entry in self.entries:
            if not LEVEL_CODE_PATTERN.fullmatch(entry.code):
                raise LevelMapError(f"invalid level code: {entry.code!r}")
            if not LEVEL_NAME_PATTERN.fullmatch(entry.name):
                raise LevelMapError(f"invalid level name: {entry.name!r}")
            if not 0 <= entry.min_total_permille <= 1000:
                raise LevelMapError("level threshold must be in 0..1000")
            if entry.code in codes:
                raise LevelMapError(f"duplicate level code: {entry.code}")
            if entry.name in names:
                raise LevelMapError(f"duplicate level name: {entry.name}")
            if entry.min_total_permille in thresholds:
                raise LevelMapError(f"duplicate level threshold: {entry.min_total_permille}")
            codes.add(entry.code)
            names.add(entry.name)
            thresholds.add(entry.min_total_permille)
        if min(thresholds) != 0:
            raise LevelMapError("level map must include a level starting at 0")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LevelMap":
        if not isinstance(raw, dict) or not isinstance(raw.get("levels"), list):
            raise FormatError("level map must contain a levels array")
        entries: list[LevelEntry] = []
        for item in raw["levels"]:
            if not isinstance(item, dict):
                raise FormatError("level entry must be an object")
            try:
                entries.append(
                    LevelEntry(
                        code=str(item["code"]),
                        name=str(item["name"]),
                        min_total_permille=int(item["min_total_permille"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FormatError(f"invalid level entry: {exc}") from exc
        return cls(entries)

    @classmethod
    def load(cls, path: str | Path) -> "LevelMap":
        try:
            return cls.from_dict(read_json(path))
        except (OSError, ValueError) as exc:
            raise FormatError(f"cannot read level map: {exc}") from exc

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "aqeval.levels.v1", "levels": [entry.to_dict() for entry in self.entries]}

    def by_code(self, code: str) -> LevelEntry:
        entry = self._by_code.get(code)
        if entry is None:
            raise LevelMapError(f"level code {code!r} is not present in the level map")
        return entry

    def by_name(self, name: str) -> LevelEntry:
        entry = self._by_name.get(name)
        if entry is None:
            raise LevelMapError(f"level name {name!r} is not present in the level map")
        return entry

    def optional_by_code(self, code: str) -> LevelEntry | None:
        return self._by_code.get(code)

    def optional_by_name(self, name: str) -> LevelEntry | None:
        return self._by_name.get(name)

    def level_for_permille(self, permille: int) -> LevelEntry:
        current = self.entries[0]
        for entry in self.entries:
            if entry.min_total_permille <= permille:
                current = entry
            else:
                break
        return current

    def level_for_micro(self, micro: int) -> LevelEntry:
        current = self.entries[0]
        for entry in self.entries:
            if entry.min_total_permille * 1000 <= micro:
                current = entry
            else:
                break
        return current

    def entry_fingerprint(self, code: str) -> str:
        entry = self.optional_by_code(code)
        if entry is None:
            return "missing"
        return sha256_json(entry.to_dict())

    def add(self, code: str, name: str, min_total_permille: int) -> "LevelMap":
        return LevelMap(self.entries + [LevelEntry(code, name, min_total_permille)])

    def remove(self, *, code: str | None = None, name: str | None = None) -> "LevelMap":
        if (code is None) == (name is None):
            raise LevelMapError("remove_level requires exactly one of code or name")
        entries = [
            entry
            for entry in self.entries
            if not ((code is not None and entry.code == code) or (name is not None and entry.name == name))
        ]
        if len(entries) == len(self.entries):
            raise LevelMapError("level to remove was not found")
        return LevelMap(entries)

    def set_threshold(self, *, code: str | None, name: str | None, value: int) -> "LevelMap":
        if (code is None) == (name is None):
            raise LevelMapError("set_level_threshold requires exactly one of code or name")
        changed: list[LevelEntry] = []
        found = False
        for entry in self.entries:
            if (code is not None and entry.code == code) or (
                name is not None and entry.name == name
            ):
                changed.append(LevelEntry(entry.code, entry.name, value))
                found = True
            else:
                changed.append(entry)
        if not found:
            raise LevelMapError("level to update was not found")
        return LevelMap(changed)
