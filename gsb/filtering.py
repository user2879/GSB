from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .model import Predicate, TableSpec
from .util import INT64_NULL


def null_mask(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    if np.issubdtype(array.dtype, np.integer):
        return array == INT64_NULL
    return np.zeros(len(array), dtype=bool)


def apply_predicate(array: np.ndarray, predicate: Predicate) -> np.ndarray:
    nulls = null_mask(array)
    valid = ~nulls
    if predicate.op == "is_null":
        return nulls
    if predicate.op == "not_null":
        return valid
    result = np.zeros(len(array), dtype=bool)
    if predicate.op == "eq":
        result = valid & (array == predicate.value)
    elif predicate.op == "ne":
        result = valid & (array != predicate.value)
    elif predicate.op == "lt":
        result = valid & (array < predicate.value)
    elif predicate.op == "le":
        result = valid & (array <= predicate.value)
    elif predicate.op == "gt":
        result = valid & (array > predicate.value)
    elif predicate.op == "ge":
        result = valid & (array >= predicate.value)
    elif predicate.op == "between":
        low, high = predicate.value
        result = valid & (array >= low) & (array <= high)
    elif predicate.op == "in":
        result = valid & np.isin(array, np.asarray(predicate.value))
    elif predicate.op == "not_in":
        result = valid & ~np.isin(array, np.asarray(predicate.value))
    return result.astype(bool, copy=False)


def filtered_arrays(
    table: TableSpec,
    arrays: Mapping[str, np.ndarray],
    predicates: Iterable[Predicate],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = int(len(next(iter(arrays.values())))) if arrays else 0
    mask = np.ones(rows, dtype=bool)
    for predicate in predicates:
        mask &= apply_predicate(arrays[predicate.column], predicate)
    filtered = {name: np.ascontiguousarray(value[mask]) for name, value in arrays.items()}
    return filtered, mask


def predicate_digest(predicates: Iterable[Predicate]) -> str:
    from .util import sha256_json

    return sha256_json([predicate.to_dict() for predicate in predicates])
