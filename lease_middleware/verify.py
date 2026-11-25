"""Independent read-only ledger replay verification."""

from __future__ import annotations

from typing import Any

from .canonical import canonical_text
from .engine import replay_bundle, result_digest
from .errors import VerifyError
from .ledger import Ledger
from .spec import SpecBundle, load_bundle


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        if isinstance(expected, int) and isinstance(actual, int) and not isinstance(
            expected, bool
        ) and not isinstance(actual, bool):
            pass
        else:
            return f"{path}: type expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            return f"{path}.{missing[0]}: missing ledger field"
        if extra:
            return f"{path}.{extra[0]}: unexpected ledger field"
        for key in sorted(expected_keys):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length expected {len(expected)}, got {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def verify_ledger(bundle: SpecBundle, ledger: Ledger) -> dict[str, Any]:
    derived = replay_bundle(bundle)
    expected_records = derived["records"]
    actual_records = ledger.records()

    common_count = min(len(expected_records), len(actual_records))
    for index in range(common_count):
        expected = expected_records[index]
        actual = actual_records[index]
        difference = first_difference(expected, actual, f"entries[{index}]")
        if difference:
            location, _, message = difference.partition(": ")
            raise VerifyError(
                "ledger item differs from replay: " + message,
                field=location,
                details={
                    "index": index,
                    "expected": expected,
                    "actual": actual,
                    "difference": difference,
                },
            )

    if len(expected_records) != len(actual_records):
        index = common_count
        raise VerifyError(
            "ledger record count differs from independent replay",
            field=f"entries[{index}]",
            details={
                "expected_count": len(expected_records),
                "actual_count": len(actual_records),
            },
        )

    ledger_digest = ledger.verify_digest()
    return {
        "ok": True,
        "code": "E_OK",
        "verified_records": len(actual_records),
        "ledger_digest": ledger_digest,
        "conclusion_digest": result_digest(derived),
        "ownership": derived["ownership"],
        "capacity": derived["capacity"],
        "summary": derived["summary"],
        "canonical_conclusion": canonical_text(
            {
                "versions": derived["versions"],
                "leases": derived["leases"],
                "ownership": derived["ownership"],
                "capacity": derived["capacity"],
            }
        ),
    }


def verify_paths(
    nodes_path: str,
    requests_path: str,
    events_path: str,
    ledger_path: str,
) -> dict[str, Any]:
    bundle = load_bundle(nodes_path, requests_path, events_path)
    ledger = Ledger.open_readonly(ledger_path)
    try:
        return verify_ledger(bundle, ledger)
    finally:
        ledger.close()
