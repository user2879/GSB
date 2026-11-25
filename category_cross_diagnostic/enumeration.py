"""Deterministic combination enumeration and fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .config import Strategy
from .config import width_gear_fingerprint
from .hashing import sha256_hex, stable_task_key
from .sample_data import TableData


@dataclass(frozen=True)
class Combo:
    combo_id: str
    columns: tuple[str, ...]
    order: int
    base_fingerprint: str
    support_pairs: tuple[tuple[str, str], ...]
    enumeration_params_fingerprint: str

    @property
    def stable_sort_key(self) -> str:
        return stable_task_key(self.base_fingerprint, "combination")

    def width_fingerprint(self, width_gear_fp: str) -> str:
        return sha256_hex("combo-width-scoped-v1", self.base_fingerprint, width_gear_fp)

    def cache_key(self, stage: str, stage_params_fp: str, width_gear_fp: str) -> str:
        active_fp = self.width_fingerprint(width_gear_fp)
        return sha256_hex(
            "cache-key-v2",
            stage,
            active_fp,
            self.base_fingerprint,
            stage_params_fp,
            width_gear_fp,
        )

    def to_identity_dict(self, width_gear_fp: str) -> dict[str, Any]:
        return {
            "combo_id": self.combo_id,
            "columns": list(self.columns),
            "order": self.order,
            "base_fingerprint": self.base_fingerprint,
            "combo_fingerprint": self.width_fingerprint(width_gear_fp),
            "stable_sort_key": self.stable_sort_key,
            "support_pairs": [list(pair) for pair in self.support_pairs],
            "enumeration_params_fingerprint": self.enumeration_params_fingerprint,
        }


def enumerate_combinations(table: TableData, strategy: Strategy) -> list[Combo]:
    edges = set(strategy.candidate_pairs)
    eligible = sorted(
        {column for pair in edges for column in pair}
        & {name for name, spec in table.columns.items() if spec.dtype == "categorical"}
    )
    max_order = min(int(strategy.order_limit), len(eligible))
    combos: list[Combo] = []

    for order in range(1, max_order + 1):
        for columns in combinations(eligible, order):
            column_tuple = tuple(columns)
            pair_support = tuple(
                pair for pair in combinations(column_tuple, 2) if pair in edges
            )
            if strategy.expansion_mode == "clique" and order >= 2:
                required_pair_count = order * (order - 1) // 2
                if len(pair_support) != required_pair_count:
                    continue

            base_fp = _combination_fingerprint(table, column_tuple, order)
            enum_params_fp = sha256_hex(
                "enumeration-params-v1",
                strategy.expansion_mode,
                order,
                [list(pair) for pair in pair_support],
            )
            combo_id = "cross__" + "__".join(column_tuple)
            combos.append(
                Combo(
                    combo_id=combo_id,
                    columns=column_tuple,
                    order=order,
                    base_fingerprint=base_fp,
                    support_pairs=pair_support,
                    enumeration_params_fingerprint=enum_params_fp,
                )
            )
    return combos


def _combination_fingerprint(
    table: TableData, columns: tuple[str, ...], order: int
) -> str:
    return sha256_hex(
        "combination-content-order-v1",
        order,
        [
            {
                "column": name,
                "content_fingerprint": table.column_fingerprints[name],
            }
            for name in columns
        ],
    )


def gear_fingerprint(strategy: Strategy) -> str:
    return width_gear_fingerprint(
        strategy.encoding_width,
        strategy.fixed_point_width,
        strategy.quantization_levels,
    )
