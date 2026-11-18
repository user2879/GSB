"""Domain editing rule table: synonym normalization, category expansion,
and answer rewrite templates."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .fingerprint import fingerprint


class RuleTable:
    def __init__(self, data: dict[str, Any]):
        self.version: str = str(data["version"])
        # Insertion order matters for deterministic first-match rewriting.
        self.synonyms: OrderedDict[str, str] = OrderedDict(
            (str(k), str(v)) for k, v in data.get("synonyms", {}).items()
        )
        self.category_expand: dict[str, list[str]] = {
            str(k): [str(v) for v in values]
            for k, values in data.get("category_expand", {}).items()
        }
        self.templates: OrderedDict[str, str] = OrderedDict(
            (str(k), str(v)) for k, v in data.get("templates", {}).items()
        )
        if not self.templates:
            raise ValueError("rule table must define at least one template")

    @property
    def template_names(self) -> list[str]:
        return list(self.templates.keys())

    def template(self, name: str) -> str:
        return self.templates[name]

    def apply_synonyms(self, tokens: list[str]) -> list[str]:
        return [self.synonyms.get(token, token) for token in tokens]

    def expand_categories(self, tokens: list[str]) -> list[str]:
        expanded: list[str] = []
        for token in tokens:
            if token not in expanded:
                expanded.append(token)
            for extra in self.category_expand.get(token, ()):
                if extra not in expanded:
                    expanded.append(extra)
        return expanded

    def render_answer(
        self,
        template_name: str,
        *,
        query: str,
        top_snippet: str,
        citation_count: int,
        top_doc_id: str,
        sources: list[str],
    ) -> str:
        template = self.template(template_name)
        return template.format(
            query=query,
            top_snippet=top_snippet,
            citation_count=citation_count,
            top_doc_id=top_doc_id,
            sources=";".join(sources),
        )

    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "synonyms": list(self.synonyms.items()),
            "category_expand": [
                [key, values]
                for key, values in sorted(self.category_expand.items())
            ],
            "templates": list(self.templates.items()),
        }
        return fingerprint(payload)

    def with_template(self, name: str, body: str) -> "RuleTable":
        data = {
            "version": self.version,
            "synonyms": dict(self.synonyms),
            "category_expand": dict(self.category_expand),
            "templates": dict(self.templates),
        }
        data["templates"][name] = body
        return RuleTable(data)


def load_rule_table(data: dict[str, Any]) -> RuleTable:
    return RuleTable(data)

