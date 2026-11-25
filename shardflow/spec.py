"""Local JSON/NDJSON specification loading and strong validation.

A declaration is a JSON document containing nodes, shards, revisions and
global limits. Events may be embedded as ``events`` in the same document or
supplied in a separate NDJSON file (one event object per line).

All validation failures raise :class:`SpecError` (``E_SPEC``) and identify the
first offending object and field.
"""

import json
import os

from .errors import SpecError
from .util import is_non_negative_int, is_positive_int


class Node:
    def __init__(self, data):
        self.id = data.get("node_id", data.get("id"))
        self.host = data["host"]
        self.port = data["port"]
        self.concurrency = data["concurrency"]
        self.lease_limit = data["lease_limit"]
        self.max_renewals = data["max_renewals"]
        self.seed = data["seed"]
        self.capacity_version = data["capacity_version"]
        self.fixed_scale = data["fixed_scale"]
        self.compute_iters = data["compute_iters"]


class Shard:
    def __init__(self, data):
        self.id = data["shard_id"]
        self.node_id = data["node_id"]
        self.depends_on = list(data["depends_on"])
        self.resource_units = data["resource_units"]
        self.sample_count = data["sample_count"]
        self.config_keys = list(data["config_keys"])


class Revision:
    def __init__(self, data):
        self.version = data["version"]
        self.parent = data["parent"]
        self.values = dict(data["values"])


class Event:
    def __init__(self, data, index):
        self.index = index
        self.type = data["type"]
        self.shard_id = data.get("shard_id")
        self.revision = data.get("revision")
        self.node_id = data.get("node_id")
        self.concurrency = data.get("concurrency")
        self.raw = dict(data)


class Spec:
    def __init__(self, doc, source_path=None):
        self.source_path = source_path
        self.global_budget = doc["global_budget"]
        self.lease_ttl_ms = doc["lease_ttl_ms"]
        self.handshake_timeout_ms = doc["handshake_timeout_ms"]
        self.io_timeout_ms = doc["io_timeout_ms"]
        self.nodes = {n["node_id"]: Node(n) for n in doc["nodes"]}
        self.shards = {s["shard_id"]: Shard(s) for s in doc["shards"]}
        self.revisions = {r["version"]: Revision(r) for r in doc["revisions"]}
        self.events = [Event(e, i) for i, e in enumerate(doc.get("events", []))]
        self.topo_order = topological_order(self.shards)
        self.revision_order = sorted(self.revisions)
        self.node_order = sorted(self.nodes)
        self.shard_order = sorted(self.shards)
        self.dependents = build_dependents(self.shards)

    def revision_values(self, version):
        """Fold the parent chain into the effective integer key mapping."""
        values = {}
        seen = set()
        current = version
        while current is not None:
            if current in seen:
                raise SpecError(
                    "revision chain contains a cycle",
                    object_id=current,
                    field="parent",
                )
            seen.add(current)
            if current not in self.revisions:
                raise SpecError(
                    "unknown parent revision", object_id=current, field="parent"
                )
            revision = self.revisions[current]
            for key, value in revision.values.items():
                values.setdefault(key, value)
            current = revision.parent
        return values

    def node_for(self, shard_id):
        return self.nodes[self.shards[shard_id].node_id]


def _fail(message, object_id=None, field=None):
    raise SpecError(message, object_id=object_id, field=field)


def _require_object(value, object_id, field):
    if not isinstance(value, dict):
        _fail("expected object", object_id, field)


def _validate_node(data, index, seen_ids, seen_ports):
    if not isinstance(data, dict):
        _fail("node declaration must be an object", "nodes[%d]" % index, None)
    node_id = data.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        _fail("node_id must be a non-empty string", "nodes[%d]" % index, "node_id")
    if node_id in seen_ids:
        _fail("duplicate node_id", node_id, "node_id")
    seen_ids.add(node_id)
    if data.get("host") != "127.0.0.1":
        _fail("host must be the literal 127.0.0.1", node_id, "host")
    port = data.get("port")
    if not is_positive_int(port) or port > 65535:
        _fail("port must be a positive integer <= 65535", node_id, "port")
    if port in seen_ports:
        _fail("duplicate loopback port", node_id, "port")
    seen_ports.add(port)
    for field in ("concurrency", "lease_limit"):
        if not is_positive_int(data.get(field)):
            _fail("%s must be a positive integer" % field, node_id, field)
    renewals = data.get("max_renewals")
    if not is_non_negative_int(renewals):
        _fail("max_renewals must be a non-negative integer", node_id, "max_renewals")
    if not is_positive_int(data.get("seed")):
        _fail("seed must be a positive integer", node_id, "seed")
    if not is_positive_int(data.get("capacity_version")):
        _fail("capacity_version must be a positive integer", node_id,
              "capacity_version")
    if not is_positive_int(data.get("fixed_scale")):
        _fail("fixed_scale must be a positive integer", node_id, "fixed_scale")
    if not is_non_negative_int(data.get("compute_iters")):
        _fail("compute_iters must be a non-negative integer", node_id,
              "compute_iters")


def _validate_shard(data, index, node_ids, seen_ids):
    if not isinstance(data, dict):
        _fail("shard declaration must be an object", "shards[%d]" % index, None)
    shard_id = data.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id:
        _fail("shard_id must be a non-empty string", "shards[%d]" % index,
              "shard_id")
    if shard_id in seen_ids:
        _fail("duplicate shard_id", shard_id, "shard_id")
    seen_ids.add(shard_id)
    if data.get("node_id") not in node_ids:
        _fail("shard references unknown node", shard_id, "node_id")
    deps = data.get("depends_on")
    if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
        _fail("depends_on must be a list of shard ids", shard_id, "depends_on")
    if len(deps) != len(set(deps)):
        _fail("duplicate dependency entry", shard_id, "depends_on")
    for field in ("resource_units", "sample_count"):
        if not is_positive_int(data.get(field)):
            _fail("%s must be a positive integer" % field, shard_id, field)
    keys = data.get("config_keys")
    if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
        _fail("config_keys must be a list of strings", shard_id, "config_keys")


def _validate_revision(data, index, seen_versions):
    if not isinstance(data, dict):
        _fail("revision must be an object", "revisions[%d]" % index, None)
    version = data.get("version")
    if not is_positive_int(version):
        _fail("revision version must be a positive integer",
              "revisions[%d]" % index, "version")
    if version in seen_versions:
        _fail("duplicate revision version", version, "version")
    seen_versions.add(version)
    parent = data.get("parent")
    if parent is not None and not is_positive_int(parent):
        _fail("parent must be null or a positive integer", version, "parent")
    values = data.get("values")
    if not isinstance(values, dict):
        _fail("revision values must be an object", version, "values")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            _fail("config key must be a non-empty string", version, "values")
        if not is_positive_int(value):
            _fail("config value must be a positive integer", version, "values.%s" % key)


def _validate_event(data, index, shard_ids, node_ids, revision_versions,
                    dispatched):
    if not isinstance(data, dict):
        _fail("event must be an object", "events[%d]" % index, None)
    event_id = "events[%d]" % index
    event_type = data.get("type")
    if event_type not in {"dispatch", "cancel", "quota_change", "config_apply"}:
        _fail("unknown event type", event_id, "type")
    if event_type in {"dispatch", "cancel"}:
        shard_id = data.get("shard_id")
        if shard_id not in shard_ids:
            _fail("event references unknown shard", event_id, "shard_id")
        if event_type == "dispatch":
            dispatched.add(shard_id)
        elif shard_id not in dispatched:
            _fail("cancel must target a previously dispatched shard", event_id,
                  "shard_id")
    if event_type == "quota_change":
        if data.get("node_id") not in node_ids:
            _fail("quota_change references unknown node", event_id, "node_id")
        if not is_positive_int(data.get("concurrency")):
            _fail("quota_change concurrency must be a positive integer",
                  event_id, "concurrency")
    if event_type == "config_apply":
        if data.get("revision") not in revision_versions:
            _fail("config_apply references unknown revision", event_id,
                  "revision")


def _validate_doc(doc):
    if not isinstance(doc, dict):
        _fail("top-level declaration must be an object", None, None)
    for field in ("nodes", "shards", "revisions"):
        if not isinstance(doc.get(field), list):
            _fail("%s must be a list" % field, None, field)
    if not is_positive_int(doc.get("global_budget")):
        _fail("global_budget must be a positive integer", None, "global_budget")
    if not is_positive_int(doc.get("lease_ttl_ms")):
        _fail("lease_ttl_ms must be a positive integer", None, "lease_ttl_ms")
    for field in ("handshake_timeout_ms", "io_timeout_ms"):
        if not is_positive_int(doc.get(field)):
            _fail("%s must be a positive integer" % field, None, field)

    node_ids, node_ports = set(), set()
    for index, node in enumerate(doc["nodes"]):
        _validate_node(node, index, node_ids, node_ports)

    shard_ids = set()
    for index, shard in enumerate(doc["shards"]):
        _validate_shard(shard, index, node_ids, shard_ids)
    for shard in doc["shards"]:
        for dep in shard["depends_on"]:
            if dep not in shard_ids:
                _fail("dependency references unknown shard", shard["shard_id"],
                      "depends_on")
        if shard["shard_id"] in shard["depends_on"]:
            _fail("shard cannot depend on itself", shard["shard_id"],
                  "depends_on")
    topological_order({s["shard_id"]: s for s in doc["shards"]})

    revision_versions = set()
    for index, revision in enumerate(doc["revisions"]):
        _validate_revision(revision, index, revision_versions)
    for revision in doc["revisions"]:
        if revision["parent"] is not None:
            if revision["parent"] not in revision_versions:
                _fail("parent revision does not exist", revision["version"],
                      "parent")
            if revision["parent"] >= revision["version"]:
                _fail("parent revision must be older than the child",
                      revision["version"], "parent")
    # Explicit parent-chain cycle detection (folding also detects, but give the
    # first offending object here).
    for revision in doc["revisions"]:
        seen = set()
        current = revision["version"]
        while current is not None:
            if current in seen:
                _fail("revision chain contains a cycle", current, "parent")
            seen.add(current)
            current = next(
                (r["parent"] for r in doc["revisions"] if r["version"] == current),
                None,
            )

    events = doc.get("events", [])
    if not isinstance(events, list):
        _fail("events must be a list", None, "events")
    dispatched = set()
    for index, event in enumerate(events):
        _validate_event(event, index, shard_ids, node_ids, revision_versions,
                        dispatched)


def topological_order(shards):
    """Kahn ordering with id tie breaks; raises E_SPEC on a cycle."""
    def _deps(sid):
        shard = shards[sid]
        return shard["depends_on"] if isinstance(shard, dict) else shard.depends_on

    indegree = {sid: len(_deps(sid)) for sid in sorted(shards)}
    dependents = {sid: [] for sid in indegree}
    for sid in sorted(shards):
        deps = _deps(sid)
        for dep in deps:
            dependents[dep].append(sid)
    ready = sorted(sid for sid, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        for child in sorted(dependents[sid]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(indegree):
        remaining = sorted(sid for sid, degree in indegree.items() if degree > 0)
        _fail("dependency graph contains a cycle", remaining[0], "depends_on")
    return order


def build_dependents(shards):
    dependents = {sid: [] for sid in shards}
    for sid in sorted(shards):
        for dep in shards[sid].depends_on:
            dependents[dep].append(sid)
    return {sid: sorted(children) for sid, children in dependents.items()}


def load_ndjson_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail("invalid NDJSON line: %s" % exc, "events[%d]" % line_number,
                      None)
            events.append(event)
    return events


def load_spec(spec_path, events_path=None):
    with open(spec_path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if events_path is None:
        default = os.path.splitext(spec_path)[0] + ".events.ndjson"
        if os.path.exists(default):
            events_path = default
    if events_path is not None:
        doc = dict(doc)
        doc["events"] = load_ndjson_events(events_path)
    _validate_doc(doc)
    return Spec(doc, source_path=os.path.abspath(spec_path))
