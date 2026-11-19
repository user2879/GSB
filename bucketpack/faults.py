"""Deterministic command-line fault injection helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .spec import RuleSpec, Spec

INJECTION_CHOICES = ("width", "bucket", "cycle", "ledger")


def apply_injections(
    spec: Spec,
    injections: Iterable[str],
) -> tuple[Spec, dict[str, dict[tuple[int, str], Any]], bool]:
    selected = set(injections)
    for injection in sorted(selected):
        if injection not in INJECTION_CHOICES:
            raise ValueError(f"unknown injection: {injection}")
    new_spec = spec
    overrides: dict[str, dict[tuple[int, str], Any]] = {}
    ledger_failure = "ledger" in selected

    if "cycle" in selected:
        source = spec.sources[0]
        cycle_rule = RuleSpec(
            id="injected.cycle",
            source=source.id,
            input_kind="bucket",
            input_name="injected.cycle",
            boundaries=(0,),
            scale=0,
        )
        new_spec = new_spec.replace(
            rules=tuple(sorted(new_spec.rules + (cycle_rule,), key=lambda item: item.id))
        )
    if "width" in selected:
        new_spec = new_spec.replace(
            packing=replace(new_spec.packing, pack_width_budget=1)
        )
    if "bucket" in selected:
        source = new_spec.sources[0]
        column = source.columns[0]
        overrides.setdefault(source.id, {})[(0, column.name)] = "injected-type-conflict"
    return new_spec, overrides, ledger_failure
