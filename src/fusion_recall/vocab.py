from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import canonical_hash


class Vocabulary:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.seed = int(raw.get("seed", 0))
        self.terms: list[dict[str, Any]] = list(raw["terms"])
        self.by_concept: dict[str, dict[str, Any]] = {}
        self._resolver: dict[str, str] = {}
        for term in self.terms:
            concept_id = term["concept_id"]
            if concept_id in self.by_concept:
                raise ValueError(f"duplicate concept id: {concept_id}")
            self.by_concept[concept_id] = term
            self._add_alias(concept_id, concept_id)
            forms = term.get("forms", {})
            for view_name, form in forms.items():
                self._add_alias(concept_id, form)
                for alias in term.get("aliases", {}).get(view_name, []):
                    self._add_alias(concept_id, alias)

    def _add_alias(self, concept_id: str, alias: Any) -> None:
        if not isinstance(alias, str) or not alias:
            return
        keys = {alias, alias.casefold()}
        for key in keys:
            existing = self._resolver.get(key)
            if existing is not None and existing != concept_id:
                raise ValueError(f"ambiguous controlled term alias: {alias}")
            self._resolver[key] = concept_id

    @property
    def fingerprint(self) -> str:
        return canonical_hash({"seed": self.seed, "terms": self.terms})

    def resolve(self, term: Any) -> str | None:
        if not isinstance(term, str) or not term:
            return None
        return self._resolver.get(term, self._resolver.get(term.casefold()))

    def surface_form(self, concept_id: str, view_name: str) -> str:
        term = self.by_concept[concept_id]
        return term.get("forms", {}).get(view_name, concept_id)

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def load_vocabulary(path: str | Path) -> Vocabulary:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("terms"), list):
        raise ValueError("vocabulary must be an object with a terms list")
    return Vocabulary(raw)
