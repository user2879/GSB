from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .graph import descendant_set, ordered_affected, shards_for_nodes
from .samples import generate_rows, sample_rows_digest
from .spec import load_spec_bundle, validate_quota_chain, validate_shards


class Workspace:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.declarations = self.root / "declarations"
        self.samples_dir = self.root / "samples"
        self.ledger_dir = self.root / "ledger"
        for path in (self.declarations, self.samples_dir, self.ledger_dir):
            path.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def load(self):
        return load_spec_bundle(self.root)

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    def read_json(self, relative: str | Path) -> Any:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def append_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.declarations / "events.json"
        doc = self.read_json(path)
        next_seq = max((int(event["seq"]) for event in doc["events"]), default=0) + 1
        event = {
            "seq": next_seq,
            "type": event_type,
            "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "payload": payload,
        }
        doc["events"].append(event)
        self.write_json("declarations/events.json", doc)
        return event

    def regenerate_sample_catalogs(self, quota_versions: list[int] | None = None) -> dict[str, Any]:
        bundle = self.load()
        versions = quota_versions or [item["version"] for item in bundle.quota_versions]
        overrides = json.loads((self.samples_dir / "overrides.json").read_text(encoding="utf-8"))["overrides"]
        written: list[str] = []
        for shard in bundle.shards:
            node = bundle.node_by_id[shard["node_id"]]
            by_version: dict[str, Any] = {}
            for version in versions:
                rows = generate_rows(node, shard, version, overrides)
                by_version[str(version)] = {
                    "row_count": len(rows),
                    "sample_digest": sample_rows_digest(rows),
                    "rows": rows,
                }
            doc = {
                "format_version": 1,
                "shard_id": shard["shard_id"],
                "node_id": shard["node_id"],
                "quota_versions": by_version,
            }
            relative = f"samples/{shard['shard_id']}.json"
            self.write_json(relative, doc)
            written.append(relative)
        index = {
            "format_version": 1,
            "generated_for_quota_versions": versions,
            "files": [f"samples/{shard['shard_id']}.json" for shard in bundle.shards],
        }
        self.write_json("samples/catalog-index.json", index)
        return {"written": written, "index": "samples/catalog-index.json"}

    def replace_sample(self, shard_id: str, row_index: int, sample: dict[str, Any]) -> dict[str, Any]:
        bundle = self.load()
        if shard_id not in bundle.shard_by_id:
            from .errors import SpecError

            raise SpecError("shard_id 不存在", entry=shard_id, field="shard_id")
        path = self.samples_dir / "overrides.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("overrides", {}).setdefault(shard_id, {})[str(row_index)] = sample
        self.write_json("samples/overrides.json", doc)
        self.append_event("sample_replaced", {"shard_id": shard_id, "row_index": row_index})
        self.regenerate_sample_catalogs()
        return {"shard_id": shard_id, "row_index": row_index, "sample": sample}

    def remove_shard(self, shard_id: str) -> dict[str, Any]:
        doc = self.read_json("declarations/shards.json")
        before = len(doc["shards"])
        doc["shards"] = [shard for shard in doc["shards"] if shard.get("shard_id") != shard_id]
        if len(doc["shards"]) == before:
            from .errors import SpecError

            raise SpecError("shard_id 不存在", entry=shard_id, field="shard_id")
        bundle = self.load()
        validate_shards(doc, bundle.nodes)
        self.write_json("declarations/shards.json", doc)
        sample_file = self.samples_dir / f"{shard_id}.json"
        if sample_file.exists():
            sample_file.unlink()
        self.append_event("shard_removed", {"shard_id": shard_id})
        return {"removed": shard_id}

    def add_shard(self, shard: dict[str, Any]) -> dict[str, Any]:
        doc = self.read_json("declarations/shards.json")
        nodes_doc = self.read_json("declarations/nodes.json")
        candidate = {"format_version": 1, "shards": doc["shards"] + [shard]}
        validate_shards(candidate, nodes_doc["nodes"])
        doc["shards"].append(shard)
        self.write_json("declarations/shards.json", doc)
        self.append_event("shard_added", {"shard_id": shard["shard_id"]})
        self.regenerate_sample_catalogs()
        return shard

    def add_quota_version(self, version: dict[str, Any]) -> dict[str, Any]:
        doc = self.read_json("declarations/quota_versions.json")
        doc["versions"].append(version)
        validate_quota_chain(doc)
        self.write_json("declarations/quota_versions.json", doc)
        self.append_event("quota_version_added", {"version": version["version"], "parent_version": version.get("parent_version")})
        return version


def propagation_report(
    bundle: Any,
    *,
    changed_shard_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    changed_node_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    quota_change: dict[str, Any] | None = None,
    active_quota_version_change: bool = False,
) -> dict[str, Any]:
    direct: set[str] = set(changed_shard_ids or [])
    direct |= shards_for_nodes(changed_node_ids or [], bundle.shards)
    if quota_change:
        direct |= quota_change.get("batch_shard_ids", set()) & set(bundle.shard_by_id)
        direct |= shards_for_nodes(quota_change.get("node_ids", set()), bundle.shards)

    affected_reasons = descendant_set(direct, bundle.shards)
    if quota_change and (quota_change.get("global_budget_changed") or quota_change.get("scale_changed")):
        for shard in bundle.shards:
            affected_reasons.setdefault(shard["shard_id"], [])
            reason = "quota_global_or_scale_change"
            if reason not in affected_reasons[shard["shard_id"]]:
                affected_reasons[shard["shard_id"]].append(reason)

    if active_quota_version_change:
        for shard in bundle.shards:
            reasons = affected_reasons.setdefault(shard["shard_id"], [])
            reason = "activated_quota_version_binding"
            if reason not in reasons:
                reasons.append(reason)

    ordered = ordered_affected(affected_reasons, bundle.shards)
    steps = [
        {
            "order_index": index,
            "shard_id": shard_id,
            "basis": affected_reasons.get(shard_id, []),
            "tie_break": "stable_topology_rank_then_shard_id",
        }
        for index, shard_id in enumerate(ordered)
    ]
    return {
        "affected_shards": ordered,
        "count": len(ordered),
        "steps": steps,
        "unaffected_shards": sorted(set(bundle.shard_by_id) - set(ordered)),
        "order_rule": "stable_topological_order_then_shard_id",
    }


def canonical_file_digest(path: Path) -> str:
    from .canonical import sha256_hex

    return sha256_hex(path.read_bytes())
