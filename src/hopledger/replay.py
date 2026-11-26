from __future__ import annotations

from typing import Any, Mapping

from .canonical import first_deep_diff
from .errors import ReplayError


RECEIPT_COMPARED_FIELDS = (
    "schema",
    "type",
    "kind",
    "hop_id",
    "node_id",
    "request_id",
    "config_version",
    "scale_factor",
    "queue_ms",
    "process_ms",
    "forward_ms",
    "status",
    "failure_injected",
    "body_bytes",
    "body_digest",
)


def comparable_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: receipt[key] for key in RECEIPT_COMPARED_FIELDS if key in receipt}


def _infer_hop_id(diff_path: str, left: Any) -> str | None:
    tokens = diff_path.replace("[", ".").replace("]", "").split(".")
    if "nodes" in tokens:
        index = tokens.index("nodes") + 1
        try:
            node_index = int(tokens[index])
            nodes = left.get("nodes", []) if isinstance(left, dict) else []
            if 0 <= node_index < len(nodes):
                return nodes[node_index].get("node_id")
        except (ValueError, IndexError, AttributeError):
            return None
    return None


def _raise_diff(
    request_id: str,
    section: str,
    diff: tuple[str, Any, Any],
    expected_root: Any,
) -> None:
    path, expected, actual = diff
    hop_id = _infer_hop_id(path, expected_root)
    field_name = path.split(".")[-1]
    raise ReplayError(
        "replay mismatch at first inconsistent item",
        request_id=request_id,
        hop_id=hop_id,
        section=section,
        field=field_name,
        path=path,
        expected=expected,
        actual=actual,
    )


def compare_result_maps(
    full: Mapping[str, Mapping[str, Any]],
    recovered: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = sorted(full)
    actual_ids = sorted(recovered)
    if expected_ids != actual_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        request_id = (missing or extra or ["<unknown>"])[0]
        raise ReplayError(
            "recovered request set does not match full run",
            request_id=request_id,
            hop_id=None,
            field="request_id",
            missing=missing,
            extra=extra,
        )

    checked_receipts = 0
    for request_id in expected_ids:
        expected = full[request_id]
        actual = recovered[request_id]

        expected_receipts = {
            receipt["node_id"]: comparable_receipt(receipt)
            for receipt in expected.get("receipts", [])
            if "node_id" in receipt
        }
        actual_receipts = {
            receipt["node_id"]: comparable_receipt(receipt)
            for receipt in actual.get("receipts", [])
            if "node_id" in receipt
        }
        for hop_id in sorted(set(expected_receipts) | set(actual_receipts)):
            if hop_id not in expected_receipts:
                raise ReplayError(
                    "unexpected recovered receipt",
                    request_id=request_id,
                    hop_id=hop_id,
                    field="receipt",
                )
            if hop_id not in actual_receipts:
                raise ReplayError(
                    "missing recovered receipt",
                    request_id=request_id,
                    hop_id=hop_id,
                    field="receipt",
                )
            diff = first_deep_diff(expected_receipts[hop_id], actual_receipts[hop_id])
            if diff is not None:
                path, expected_value, actual_value = diff
                raise ReplayError(
                    "replay receipt field mismatch",
                    request_id=request_id,
                    hop_id=hop_id,
                    field=path.split(".")[-1],
                    path=path,
                    expected=expected_value,
                    actual=actual_value,
                )
            checked_receipts += 1

        for section in ("conclusion", "attribution"):
            expected_value = expected.get(section)
            actual_value = actual.get(section)
            diff = first_deep_diff(expected_value, actual_value)
            if diff is not None:
                _raise_diff(request_id, section, diff, expected_value)

    return {
        "ok": True,
        "request_count": len(expected_ids),
        "receipt_count": checked_receipts,
        "first_mismatch": None,
    }
