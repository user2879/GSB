"""Gray snapshot rebuild, authoritative recompute and recovery replay.

The fast Tkinter view reads snapshot rows directly. The authoritative path
re-derives routing decisions and NumPy output digests from the declaration and
the persisted parameter-version table, then re-merges everything and compares
the two results field by field.
"""

from collections import OrderedDict
from datetime import datetime, timezone

from . import determinism as det
from .errors import ReplayError, SnapshotError
from .hashing import canonical_bytes, route_decision, sha256_hex
from .storage import LedgerStore

DESTINATIONS = ("gray", "stable")


def _empty_group():
    return {
        "hit_count": 0,
        "sample_ids": [],
        "batch_ids": [],
        "sub_batches": [],
        "elapsed_ms_total": 0,
        "elapsed_ms_max": 0,
        "param_versions": [],
        "output_digests": [],
        "splits": [],
    }


def merge_hits(hits, seq_from, seq_to):
    """Merge hits grouped by (rule_id, destination), version chain ascending."""
    groups = OrderedDict()
    for hit in sorted(hits, key=lambda h: (h["rule_id"], h["destination"],
                                           h["backend_seq"], h["ledger_seq"])):
        key = (hit["rule_id"], hit["destination"])
        group = groups.setdefault(key, _empty_group())
        group["hit_count"] += 1
        group["sample_ids"].append(hit["sample_id"])
        if hit["batch_id"] not in group["batch_ids"]:
            group["batch_ids"].append(hit["batch_id"])
        sub_key = [hit["batch_id"], hit["sub_batch"]]
        if sub_key not in group["sub_batches"]:
            group["sub_batches"].append(sub_key)
        group["elapsed_ms_total"] += hit["elapsed_ms"]
        group["elapsed_ms_max"] = max(group["elapsed_ms_max"],
                                      hit["elapsed_ms"])
        version_entry = {"backend_seq": hit["backend_seq"],
                         "param_digest": hit["param_digest"]}
        if version_entry not in group["param_versions"]:
            group["param_versions"].append(version_entry)
        group["output_digests"].append(hit["output_digest"])
        if hit.get("split_reason"):
            reason = {"batch_id": hit["batch_id"],
                      "sub_batch": hit["sub_batch"],
                      "reason": hit["split_reason"]}
            if reason not in group["splits"]:
                group["splits"].append(reason)

    body = OrderedDict()
    for (rule_id, destination), group in groups.items():
        body.setdefault(rule_id, {})[destination] = _finalize_group(group)
    for rule in body:
        total = sum(body[rule][d]["hit_count"]
                    for d in DESTINATIONS if d in body[rule])
        gray = body[rule].get("gray", {}).get("hit_count", 0)
        body[rule]["gray_ratio_bp"] = (
            round(gray * 10000 / total) if total else 0
        )
    return {
        "ledger_seq_from": seq_from,
        "ledger_seq_to": seq_to,
        "groups": body,
    }


def _finalize_group(group):
    finalized = dict(group)
    finalized["sample_ids"] = sorted(group["sample_ids"])
    finalized["output_digest_chain"] = sha256_hex(
        canonical_bytes(group["output_digests"])
    )
    del finalized["output_digests"]
    finalized["elapsed_ms_avg"] = (
        round(group["elapsed_ms_total"] / group["hit_count"], 3)
        if group["hit_count"] else 0
    )
    finalized["param_versions"] = sorted(
        group["param_versions"], key=lambda v: v["backend_seq"]
    )
    return finalized


def snapshot_digest(snapshot_body, param_versions, seq_from, seq_to):
    envelope = {
        "ledger_seq_from": seq_from,
        "ledger_seq_to": seq_to,
        "param_versions": sorted(
            param_versions, key=lambda v: (v["backend_id"], v["backend_seq"])
        ),
        "body": snapshot_body,
    }
    return sha256_hex(canonical_bytes(envelope))


def build_snapshot(store: LedgerStore, rebuild_id, upper_bound):
    hits = store.hits_in_range(1, upper_bound)
    seq_from = hits[0]["ledger_seq"] if hits else 0
    seq_to = hits[-1]["ledger_seq"] if hits else 0
    merged = merge_hits(hits, seq_from, seq_to)
    param_versions = [
        {"backend_id": r["backend_id"], "backend_seq": r["backend_seq"],
         "param_digest": r["param_digest"]}
        for r in store.param_versions()
    ]
    digest = snapshot_digest(merged["groups"], param_versions, seq_from, seq_to)
    snapshot = {
        "snapshot_id": f"snap-{rebuild_id}",
        "rebuild_id": rebuild_id,
        "ledger_seq_from": seq_from,
        "ledger_seq_to": seq_to,
        "param_versions": param_versions,
        "body": merged["groups"],
        "digest": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incoming_during_rebuild": None,
    }
    return snapshot


def _rebuild_params_for(version_row):
    params = det.build_parameters(
        version_row["seed"], version_row["model_dim"],
        version_row["hidden_dim"],
    )
    digest = det.parameter_digest(
        version_row["backend_id"], version_row["model_version"],
        version_row["quant_version"], version_row["model_dim"],
        version_row["hidden_dim"], version_row["seed"], params,
    )
    if digest != version_row["param_digest"]:
        raise ReplayError(
            f"parameter version {version_row['backend_id']}#"
            f"{version_row['backend_seq']} is not reproducible",
            field=f"param_versions.{version_row['backend_id']}."
                  f"{version_row['backend_seq']}",
        )
    return params


def authoritative_recompute(declaration, store: LedgerStore,
                            seq_from=None, seq_to=None):
    """Replay every hit: routing, forward pass, digest and re-merge.

    Row content-digest verification is deferred to the end so an injected
    reply corruption surfaces as E_SNAPSHOT before the ledger-level E_REPLAY.
    """
    if seq_to is None:
        seq_to = store.max_ledger_seq()
    if seq_from is None:
        seq_from = 1
    hits = store.hits_in_range(seq_from, seq_to)

    version_rows = {(r["backend_id"], r["backend_seq"]): r
                    for r in store.param_versions()}
    params_cache = {}
    rules = {r["rule_id"]: r for r in declaration["rules"]}
    backends = {b["backend_id"]: b for b in declaration["backends"]}

    replayed = []
    for hit in hits:
        rule = rules[hit["rule_id"]]
        decision = route_decision(hit["rule_id"], hit["client_key"],
                                  hit["weight_bp"])
        if decision != hit["destination"]:
            raise SnapshotError(
                f"routing replay mismatch on {hit['sample_id']}",
                field=f"{hit['rule_id']}.{hit['sample_id']}.destination",
            )
        backend_id = (rule["stable_backend"] if decision == "stable"
                      else rule["gray_backend"])
        if backend_id != hit["backend_id"]:
            raise SnapshotError(
                f"backend replay mismatch on {hit['sample_id']}",
                field=f"{hit['rule_id']}.{hit['sample_id']}.backend_id",
            )
        version_key = (backend_id, hit["backend_seq"])
        if version_key not in version_rows:
            raise ReplayError(
                f"parameter digest version missing: {backend_id}#"
                f"{hit['backend_seq']} (sample {hit['sample_id']})",
                field=f"param_versions.{backend_id}.{hit['backend_seq']}",
            )
        if version_key not in params_cache:
            params_cache[version_key] = _rebuild_params_for(
                version_rows[version_key]
            )
        vector = det.build_input_vector(
            hit["rule_id"], hit["sample_id"], backends[backend_id]["model_dim"]
        )
        in_digest = det.input_digest(
            hit["rule_id"], hit["sample_id"],
            backends[backend_id]["model_dim"], vector,
        )
        if in_digest != hit["input_digest"]:
            raise SnapshotError(
                f"input digest replay mismatch on {hit['sample_id']}",
                field=f"{hit['rule_id']}.{hit['sample_id']}.input_digest",
            )
        import numpy as np

        output = det.forward(params_cache[version_key], vector.reshape(1, -1))
        out_digest = det.output_digest(
            hit["rule_id"], [hit["sample_id"]], hit["param_digest"], output
        )
        if out_digest != hit["output_digest"]:
            raise SnapshotError(
                f"corrupted reply digest detected on {hit['sample_id']}",
                field=f"{hit['rule_id']}.{hit['sample_id']}.output_digest",
            )
        replayed.append(hit)
    store.verify_hit_digests()
    return merge_hits(replayed, seq_from, seq_to)


def compare_snapshots(expected_body, actual_body, path=""):
    """Raise E_SNAPSHOT at the first differing rule/field (byte comparison)."""
    if isinstance(expected_body, dict) and isinstance(actual_body, dict):
        for key in sorted(set(expected_body) | set(actual_body)):
            sub_path = f"{path}.{key}" if path else key
            if key not in actual_body:
                raise SnapshotError(f"missing field {sub_path} in snapshot",
                                    field=sub_path)
            if key not in expected_body:
                raise SnapshotError(f"unexpected field {sub_path} in snapshot",
                                    field=sub_path)
            compare_snapshots(expected_body[key], actual_body[key], sub_path)
        return
    if expected_body != actual_body:
        raise SnapshotError(
            f"snapshot mismatch at {path}: {expected_body!r} != {actual_body!r}",
            field=path,
        )


def verify_snapshot(declaration, store: LedgerStore):
    snapshot = store.load_latest_snapshot()
    if snapshot is None:
        raise ReplayError("no persisted snapshot to verify", field="snapshots")
    recomputed = authoritative_recompute(
        declaration, store, snapshot["ledger_seq_from"],
        snapshot["ledger_seq_to"],
    )
    compare_snapshots(snapshot["body"], recomputed["groups"])
    digest = snapshot_digest(
        recomputed["groups"], snapshot["param_versions"],
        snapshot["ledger_seq_from"], snapshot["ledger_seq_to"],
    )
    if digest != snapshot["digest"]:
        raise ReplayError(
            "snapshot envelope digest not reproducible",
            field=f"snapshots.{snapshot['snapshot_id']}.digest",
        )
    return snapshot
