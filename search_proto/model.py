from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .util import sha256_json, sorted_unique


FIELD_ORDER = ("title", "body")
FIELD_RANK = {name: index for index, name in enumerate(FIELD_ORDER)}


def _ordered_fields(fields: dict[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(fields.get(name, ()))
        for name in FIELD_ORDER
        if name in fields
    }


def _ordered_marks(
    marks: dict[str, Iterable[int]] | None,
) -> dict[str, tuple[int, ...]]:
    if not marks:
        return {}
    return {
        name: tuple(sorted(int(position) for position in marks.get(name, ())))
        for name in FIELD_ORDER
        if name in marks and marks.get(name)
    }


@dataclass(frozen=True)
class Document:
    doc_id: str
    version: int
    fields: dict[str, tuple[str, ...]]
    stopword_marks: dict[str, tuple[int, ...]]
    content_fingerprint: str

    @classmethod
    def create(
        cls,
        doc_id: str,
        version: int,
        fields: dict[str, Iterable[str]],
        stopword_marks: dict[str, Iterable[int]] | None = None,
    ) -> "Document":
        ordered_fields = _ordered_fields(fields)
        ordered_marks = _ordered_marks(stopword_marks)
        payload = {
            "doc_id": doc_id,
            "version": version,
            "fields": {key: list(value) for key, value in ordered_fields.items()},
            "field_stopword_marks": {
                key: list(value) for key, value in ordered_marks.items()
            },
        }
        return cls(
            doc_id=doc_id,
            version=version,
            fields=ordered_fields,
            stopword_marks=ordered_marks,
            content_fingerprint=sha256_json(payload),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Document":
        fields_raw = value.get("fields")
        if fields_raw is None:
            fields_raw = {
                name: value.get(name)
                for name in FIELD_ORDER
                if value.get(name) is not None
            }
        marks_raw = value.get("field_stopword_marks", value.get("stopword_marks", {}))
        return cls.create(
            doc_id=value["doc_id"],
            version=value["version"],
            fields=fields_raw,
            stopword_marks=marks_raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version": self.version,
            "fields": {key: list(value) for key, value in self.fields.items()},
            "field_stopword_marks": {
                key: list(value) for key, value in self.stopword_marks.items()
            },
            "content_fingerprint": self.content_fingerprint,
        }

    def field_order(self) -> tuple[str, ...]:
        return tuple(name for name in FIELD_ORDER if name in self.fields)


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    terms: tuple[str, ...]
    field_preferences: tuple[str, ...]
    content_fingerprint: str

    @classmethod
    def create(
        cls,
        query_id: str,
        terms: Iterable[str],
        field_preferences: Iterable[str],
    ) -> "QuerySpec":
        raw_terms = tuple(terms)
        ordered_preferences = tuple(
            name for name in FIELD_ORDER if name in set(field_preferences)
        )
        payload = {
            "query_id": query_id,
            "terms": list(raw_terms),
            "field_preferences": list(ordered_preferences),
        }
        return cls(
            query_id=query_id,
            terms=raw_terms,
            field_preferences=ordered_preferences,
            content_fingerprint=sha256_json(payload),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QuerySpec":
        preferences = value.get("field_preferences", FIELD_ORDER)
        if isinstance(preferences, dict):
            preferences = [name for name, enabled in preferences.items() if enabled]
        return cls.create(value["query_id"], value.get("terms", ()), preferences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "terms": list(self.terms),
            "field_preferences": list(self.field_preferences),
            "content_fingerprint": self.content_fingerprint,
        }

    def sorted_unique_terms(self) -> tuple[str, ...]:
        return tuple(sorted_unique(self.terms))


@dataclass(frozen=True)
class StopwordTable:
    version: str
    global_terms: frozenset[str]
    by_field: dict[str, frozenset[str]]
    content_fingerprint: str
    effective_version: str

    @classmethod
    def create(
        cls,
        version: str,
        global_terms: Iterable[str] = (),
        by_field: dict[str, Iterable[str]] | None = None,
    ) -> "StopwordTable":
        normalised_global = frozenset(global_terms)
        normalised_by_field = {
            name: frozenset((by_field or {}).get(name, ()))
            for name in FIELD_ORDER
            if (by_field or {}).get(name)
        }
        payload = {
            "version": version,
            "global": sorted(normalised_global),
            "by_field": {
                name: sorted(normalised_by_field.get(name, ()))
                for name in FIELD_ORDER
                if normalised_by_field.get(name)
            },
        }
        fingerprint = sha256_json(payload)
        return cls(
            version=version,
            global_terms=normalised_global,
            by_field=normalised_by_field,
            content_fingerprint=fingerprint,
            effective_version=f"{version}@{fingerprint[:16]}",
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StopwordTable":
        global_terms = value.get("global", value.get("global_terms", ()))
        return cls.create(
            version=value.get("version", "unversioned"),
            global_terms=global_terms,
            by_field=value.get("by_field", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "global": sorted(self.global_terms),
            "by_field": {
                name: sorted(self.by_field.get(name, ()))
                for name in FIELD_ORDER
                if self.by_field.get(name)
            },
            "content_fingerprint": self.content_fingerprint,
            "effective_version": self.effective_version,
        }

    def is_stopword(self, field: str, term: str) -> bool:
        return term in self.global_terms or term in self.by_field.get(field, ())

    def add(self, field: str, term: str) -> "StopwordTable":
        if field == "*":
            return StopwordTable.create(
                self.version,
                self.global_terms | {term},
                {name: self.by_field.get(name, frozenset()) for name in FIELD_ORDER},
            )
        by_field = {
            name: set(self.by_field.get(name, frozenset())) for name in FIELD_ORDER
        }
        if field not in by_field:
            raise ValueError(f"unknown stopword field: {field}")
        by_field[field].add(term)
        return StopwordTable.create(self.version, self.global_terms, by_field)

    def remove(self, field: str, term: str) -> "StopwordTable":
        if field == "*":
            return StopwordTable.create(
                self.version,
                self.global_terms - {term},
                {name: self.by_field.get(name, frozenset()) for name in FIELD_ORDER},
            )
        by_field = {
            name: set(self.by_field.get(name, frozenset())) for name in FIELD_ORDER
        }
        if field not in by_field:
            raise ValueError(f"unknown stopword field: {field}")
        by_field[field].discard(term)
        return StopwordTable.create(self.version, self.global_terms, by_field)


@dataclass(frozen=True)
class ControlledVocabulary:
    version: str
    terms: frozenset[str]
    content_fingerprint: str

    @classmethod
    def create(cls, version: str, terms: Iterable[str]) -> "ControlledVocabulary":
        normalised = frozenset(terms)
        payload = {"version": version, "terms": sorted(normalised)}
        return cls(
            version=version,
            terms=normalised,
            content_fingerprint=sha256_json(payload),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any] | list[str]) -> "ControlledVocabulary":
        if isinstance(value, list):
            return cls.create("unversioned", value)
        return cls.create(value.get("version", "unversioned"), value.get("terms", ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "terms": sorted(self.terms),
            "content_fingerprint": self.content_fingerprint,
        }

