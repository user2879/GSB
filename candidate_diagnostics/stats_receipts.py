from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NORMALIZED_FIELDS = [
    "request_id",
    "channel",
    "population",
    "quota",
    "selected",
    "unselected",
    "ratio_permille",
    "scheme",
    "receipt_version",
    "receipt_time",
    "source_format",
    "source_file",
]


def _receipt(
    *,
    request_id: str,
    channel: str,
    population: int,
    selected: int,
    quota: int | None = None,
    unselected: int | None = None,
    ratio_permille: int | None = None,
    scheme: str | None = None,
    version: int,
    receipt_time: str,
    source_format: str,
    source_file: str,
) -> dict[str, Any]:
    population = int(population)
    selected = int(selected)
    quota = int(quota if quota is not None else selected)
    unselected = int(unselected if unselected is not None else population - selected)
    return {
        "request_id": request_id,
        "channel": channel or "__ALL__",
        "population": population,
        "quota": quota,
        "selected": selected,
        "unselected": unselected,
        "ratio_permille": ratio_permille,
        "scheme": scheme,
        "receipt_version": int(version),
        "receipt_time": receipt_time,
        "source_format": source_format,
        "source_file": source_file,
        "consistency_ok": selected + unselected == population and quota <= population,
    }


def parse_legacy_pipe(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    receipts: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|")
        if parts[0] != "STATS1" or len(parts) != 9:
            continue
        _, rid, channel, population, quota, selected, ratio, ts, version = parts
        receipts.append(
            _receipt(
                request_id=rid,
                channel=channel,
                population=int(population),
                quota=int(quota),
                selected=int(selected),
                ratio_permille=int(ratio),
                receipt_time=ts,
                version=int(version),
                scheme="legacy_pipe_layer_quota_v1",
                source_format="legacy_pipe_v1",
                source_file=str(path),
            )
        )
    return receipts


def parse_nested_json(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    envelope_version = int(payload.get("receipt_version", 2))
    issued_at = payload.get("issued_at", "")
    receipts: list[dict[str, Any]] = []
    for item in payload.get("receipts", []):
        counts = item.get("counts", {})
        sampling = item.get("sampling", {})
        receipts.append(
            _receipt(
                request_id=item["request_id"],
                channel=item.get("layer", item.get("channel", "__ALL__")),
                population=counts.get("population", counts.get("total")),
                quota=counts.get("quota"),
                selected=counts.get("selected", counts.get("picked")),
                unselected=counts.get("unselected"),
                ratio_permille=sampling.get("ratio_permille"),
                scheme=sampling.get("scheme"),
                version=item.get("receipt_version", envelope_version),
                receipt_time=item.get("receipt_time", issued_at),
                source_format="nested_json_v2",
                source_file=str(path),
            )
        )
    return receipts


def parse_csv_v3(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    receipts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ratio = row.get("ratio_permille")
            if ratio in (None, ""):
                decimal_ratio = float(row["ratio"])
                ratio_permille = int(round(decimal_ratio * 1000))
            else:
                ratio_permille = int(ratio)
            receipts.append(
                _receipt(
                    request_id=row["request_id"],
                    channel=row.get("channel", "__ALL__"),
                    population=int(row["population"]),
                    quota=int(row.get("quota") or row["selected"]),
                    selected=int(row["selected"]),
                    unselected=int(row["unselected"]) if row.get("unselected") else None,
                    ratio_permille=ratio_permille,
                    scheme=row.get("scheme"),
                    version=int(row.get("receipt_version", 3)),
                    receipt_time=row["receipt_time"],
                    source_format="columnar_csv_v3",
                    source_file=str(path),
                )
            )
    return receipts


def load_three_receipt_formats(
    v1_path: str | Path, v2_path: str | Path, v3_path: str | Path
) -> list[dict[str, Any]]:
    return (
        parse_legacy_pipe(v1_path)
        + parse_nested_json(v2_path)
        + parse_csv_v3(v3_path)
    )


def adjudicate_receipts(receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    receipts = list(receipts)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        groups[(receipt["request_id"], receipt["channel"])].append(dict(receipt))

        authoritative: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda item: (int(item["receipt_version"]), str(item["receipt_time"])),
        )
        winner = candidates[-1]
        ranked = list(reversed(candidates))
        for receipt in candidates:
            receipt["authoritative"] = receipt is winner
        authoritative.append(winner)
        same_version_latest: list[dict[str, Any]] = []
        if len(candidates) == 1:
            basis = "only receipt for request/channel"
        else:
            top = ranked[0]
            second = ranked[1]
            if top["receipt_version"] > second["receipt_version"]:
                basis = (
                    f"receipt_version {top['receipt_version']} takes precedence over "
                    f"{second['receipt_version']}; winner time {top['receipt_time']}"
                )
            else:
                same_version_latest = [
                    item for item in ranked if item["receipt_version"] == top["receipt_version"]
                ]
                basis = (
                    f"same receipt_version {top['receipt_version']}; latest receipt_time "
                    f"{top['receipt_time']} wins over {second['receipt_time']}"
                )
                if (
                    len(same_version_latest) > 1
                    and same_version_latest[0]["receipt_time"]
                    == same_version_latest[1]["receipt_time"]
                ):
                    basis += "; tied version/time resolved by stable source_file/source_format order"
                    winner = sorted(
                        same_version_latest,
                        key=lambda item: (item["source_file"], item["source_format"]),
                    )[0]
                    for receipt in candidates:
                        receipt["authoritative"] = receipt is winner
        decisions.append(
            {
                "request_id": key[0],
                "channel": key[1],
                "authoritative_source": winner["source_format"],
                "authoritative_file": winner["source_file"],
                "authoritative_version": winner["receipt_version"],
                "authoritative_time": winner["receipt_time"],
                "decision_basis": basis,
                "candidate_order_new_to_old": [
                    {
                        "version": item["receipt_version"],
                        "time": item["receipt_time"],
                        "source_format": item["source_format"],
                    }
                    for item in ranked
                ],
            }
        )

    view = {
        "schema": "normalized-sampling-stats-view-v1",
        "receipt_count": len(receipts),
        "authoritative_count": len(authoritative),
        "authoritative": sorted(authoritative, key=lambda r: (r["request_id"], r["channel"])),
        "decisions": decisions,
        "all_receipts": sorted(
            receipts,
            key=lambda r: (r["request_id"], r["channel"], r["receipt_version"], r["receipt_time"]),
        ),
    }
    return view
