from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import Analyzer
from .errors import ReplayError, SnapshotError
from .ledger import Ledger
from .persistence_model import batch_content_digest, split_service_analysis


def first_difference(expected: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            key = missing[0] if missing else extra[0]
            return (
                f"{path}.{key}",
                expected.get(key, "<missing>"),
                actual.get(key, "<missing>"),
            )
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}.length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return path, expected, actual
    return None


def _match_service_rows(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row.get("policy_id"),
            row.get("scope", "bucket"),
            row.get("bucket_index", -1),
        ): row
        for row in actual
    }
    ordered = []
    for row in expected:
        key = (
            row.get("policy_id"),
            row.get("scope", "bucket"),
            row.get("bucket_index", -1),
        )
        ordered.append(by_key.get(key, {}))
    return ordered


def _raise_mismatch(service_id: str, field_path: str, detail: tuple[str, Any, Any]) -> None:
    _, expected, actual = detail
    raise ReplayError(
        f"$.services[{service_id}].{field_path}",
        "stored ledger content cannot be reproduced from the recorded specification",
        service_id=service_id,
        field=f"$.services[{service_id}].{field_path}",
        context={"expected": expected, "actual": actual},
    )


def verify_ledger_run(
    db_path: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        if run_id is None:
            current = ledger.get_current_run_row()
            if current is None:
                raise SnapshotError("$.runs", "there is no active complete ledger run")
            run_id = current["run_id"]
        view = ledger.get_run_view(run_id, verify=False)
        if view is None:
            raise SnapshotError(f"$.runs[{run_id}]", "ledger run is missing")
        services_doc, policies_doc = ledger.run_documents(run_id)
        analyzer = Analyzer(services_doc, policies_doc)
        bundle = analyzer.build_bundle(include_sensitivity=True)
        stored_services = {row["service_id"]: row for row in view["services"]}

        service_payload_checks = 0
        bucket_checks = 0
        attribution_checks = 0
        for analysis in bundle.service_analyses:
            service_id = analysis["service_id"]
            stored = stored_services.get(service_id)
            if stored is None or stored["status"] != "committed":
                raise ReplayError(
                    f"$.services[{service_id}]",
                    "service batch is not committed",
                    service_id=service_id,
                    field=f"$.services[{service_id}].status",
                )
            expected_split = split_service_analysis(analysis)
            comparisons = (
                ("summary", expected_split["summary"], stored["summary"]),
                ("error_bounds", expected_split["error_bounds"], stored["error_bounds"]),
                ("buckets", expected_split["buckets"], stored["buckets"]),
                (
                    "attribution",
                    expected_split["attribution"],
                    _match_service_rows(expected_split["attribution"], stored["attribution"]),
                ),
            )
            for field_name, expected, actual in comparisons:
                difference = first_difference(expected, actual, f"$.services[{service_id}].{field_name}")
                if difference is not None:
                    _raise_mismatch(service_id, field_name + difference[0][len(f"$.services[{service_id}].{field_name}"):], difference)
                if field_name == "buckets":
                    bucket_checks += len(expected)
                elif field_name == "attribution":
                    attribution_checks += len(expected)
                else:
                    service_payload_checks += 1
            owner_flips = [
                flip for flip in bundle.ledger_flips if flip["owner_service"] == service_id
            ]
            expected_digest = batch_content_digest(
                run_id,
                int(stored["ordinal"]),
                service_id,
                expected_split,
                owner_flips,
            )
            if expected_digest != stored["content_digest"]:
                raise ReplayError(
                    f"$.services[{service_id}].content_digest",
                    "batch content digest is not reproducible",
                    service_id=service_id,
                    field=f"$.services[{service_id}].content_digest",
                    context={"expected": expected_digest, "actual": stored["content_digest"]},
                )

        expected_flips = sorted(
            [
                {
                    "owner_service": flip["owner_service"],
                    "flip_key": flip["flip_key"],
                    "kind": flip["kind"],
                    "payload": flip["payload"],
                }
                for flip in bundle.ledger_flips
            ],
            key=lambda item: (item["owner_service"], item["flip_key"]),
        )
        actual_flips = sorted(
            view["flips"],
            key=lambda item: (item["owner_service"], item["flip_key"]),
        )
        difference = first_difference(expected_flips, actual_flips, "$.flips")
        if difference is not None:
            owner = "<ledger>"
            field = difference[0]
            if expected_flips and actual_flips:
                owner = actual_flips[0].get("owner_service", owner)
            raise ReplayError(
                field,
                "stored flip records cannot be reproduced",
                service_id=owner,
                field=field,
            )
        actual_snapshot = ledger.verify_run(run_id)
        return {
            "ok": True,
            "code": "OK",
            "run_id": run_id,
            "strategy_version": policies_doc.version,
            "service_version": services_doc.version,
            "snapshot_digest": actual_snapshot,
            "services_checked": len(bundle.service_analyses),
            "policies_per_service": {
                analysis["service_id"]: sorted(analysis["policies"])
                for analysis in bundle.service_analyses
            },
            "scale_powers": {
                analysis["service_id"]: {
                    policy_id: result["policy"]["scale_power"]
                    for policy_id, result in sorted(analysis["policies"].items())
                }
                for analysis in bundle.service_analyses
            },
            "service_payload_checks": service_payload_checks,
            "bucket_checks": bucket_checks,
            "attribution_checks": attribution_checks,
        }
