"""Load and strongly validate services / requests / keys JSON files.

Any violation raises DiagError(E_SPEC) naming the first illegal field and
its 1-based line number, and the whole specification is rejected.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import E_SPEC
from .errors import DiagError
from .jsonparse import parse_file

LOOPBACK_LITERAL = "127.0.0.1"
MIN_PORT = 1024
MAX_PORT = 65535


@dataclass(frozen=True)
class KeyVersion:
    version: int
    secret: str
    retire_ms: int  # -1 means "current version, never retired"


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    host: str
    port: int
    capacity_tokens: int
    refill_interval_ms: int
    cost_tokens: int
    signature_window_ms: int
    bucket_keys: tuple


@dataclass(frozen=True)
class RequestSpec:
    seq: int
    client_key: str
    service_id: str
    arrival_ms: int
    bucket_key: str
    cred_version: int
    signature: str


@dataclass(frozen=True)
class Spec:
    services: dict
    keys: dict
    requests: tuple
    services_digest: str = ""
    keys_digest: str = ""
    requests_digest: str = ""

    def current_version(self, service_id):
        return self.keys[service_id][-1].version


def _fail(message, field=None, line=None):
    raise DiagError(E_SPEC, message, field=field, line=line)


def _require_obj(node, ctx):
    if not isinstance(node.value, dict):
        _fail(f"{ctx}: expected JSON object", line=node.line)
    return node.value


def _require_field(obj_node, key, ctx):
    obj = _require_obj(obj_node, ctx)
    if key not in obj:
        _fail(f"{ctx}: missing required field {key!r}", field=key,
              line=obj_node.line)
    return obj[key]


def _as_str(node, key, ctx, non_empty=True):
    if not isinstance(node.value, str) or (non_empty and not node.value):
        _fail(f"{ctx}: field {key!r} must be a non-empty string",
              field=key, line=node.line)
    return node.value


def _as_int(node, key, ctx, minimum=None):
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{ctx}: field {key!r} must be an integer",
              field=key, line=node.line)
    if minimum is not None and value < minimum:
        _fail(f"{ctx}: field {key!r} must be >= {minimum}",
              field=key, line=node.line)
    return value


def _load_services(path):
    root = parse_file(path)
    services_node = _require_field(root, "services", "services.json")
    if not isinstance(services_node.value, list):
        _fail("services.json: 'services' must be an array",
              field="services", line=services_node.line)
    services = {}
    used_ports = {}
    for item in services_node.value:
        ctx = f"services.json entry (line {item.line})"
        sid = _as_str(_require_field(item, "service_id", ctx),
                      "service_id", ctx)
        if sid in services:
            _fail(f"duplicate service_id {sid!r}", field="service_id",
                  line=item.line)
        host = _as_str(_require_field(item, "host", ctx), "host", ctx)
        if host != LOOPBACK_LITERAL:
            _fail(f"service {sid!r}: host must be the loopback literal "
                  f"{LOOPBACK_LITERAL!r}", field="host", line=item.line)
        port = _as_int(_require_field(item, "port", ctx), "port", ctx)
        if not (MIN_PORT <= port <= MAX_PORT):
            _fail(f"service {sid!r}: port {port} out of range "
                  f"[{MIN_PORT},{MAX_PORT}]", field="port", line=item.line)
        if port in used_ports:
            _fail(f"service {sid!r}: port {port} already used by "
                  f"{used_ports[port]!r}", field="port", line=item.line)
        used_ports[port] = sid
        capacity = _as_int(_require_field(item, "capacity_tokens", ctx),
                           "capacity_tokens", ctx, minimum=1)
        interval = _as_int(_require_field(item, "refill_interval_ms", ctx),
                           "refill_interval_ms", ctx, minimum=1)
        cost = _as_int(_require_field(item, "cost_tokens", ctx),
                       "cost_tokens", ctx, minimum=1)
        window = _as_int(_require_field(item, "signature_window_ms", ctx),
                         "signature_window_ms", ctx, minimum=1)
        bk_node = _require_field(item, "bucket_keys", ctx)
        if (not isinstance(bk_node.value, list) or not bk_node.value):
            _fail(f"service {sid!r}: 'bucket_keys' must be a non-empty "
                  f"array", field="bucket_keys", line=bk_node.line)
        bucket_keys = []
        for bk in bk_node.value:
            name = _as_str(bk, "bucket_keys", ctx)
            if name in bucket_keys:
                _fail(f"service {sid!r}: duplicate bucket_key {name!r}",
                      field="bucket_keys", line=bk.line)
            bucket_keys.append(name)
        services[sid] = ServiceSpec(
            service_id=sid, host=host, port=port,
            capacity_tokens=capacity, refill_interval_ms=interval,
            cost_tokens=cost, signature_window_ms=window,
            bucket_keys=tuple(bucket_keys))
    if not services:
        _fail("services.json: at least one service is required",
              field="services", line=services_node.line)
    return services


def _load_keys(path, services):
    root = parse_file(path)
    keys_node = _require_field(root, "keys", "keys.json")
    keys_obj = _require_obj(keys_node, "keys.json: 'keys'")
    keys = {}
    for sid in sorted(keys_obj):  # stable iteration; validation order below
        svc_node = keys_obj[sid]
        if sid not in services:
            _fail(f"keys.json: key set references unknown service {sid!r}",
                  field=sid, line=svc_node.line)
        if not isinstance(svc_node.value, list) or not svc_node.value:
            _fail(f"keys.json: service {sid!r} must have a non-empty "
                  f"version array", field=sid, line=svc_node.line)
        versions = []
        prev_version = None
        count = len(svc_node.value)
        for index, entry in enumerate(svc_node.value):
            ctx = f"keys.json service {sid!r} entry (line {entry.line})"
            version = _as_int(_require_field(entry, "version", ctx),
                              "version", ctx, minimum=1)
            if prev_version is not None and version <= prev_version:
                _fail(f"keys.json service {sid!r}: versions must be "
                      f"strictly increasing ({version} after "
                      f"{prev_version})", field="version", line=entry.line)
            prev_version = version
            secret = _as_str(_require_field(entry, "secret", ctx),
                             "secret", ctx)
            retire_node = _require_field(entry, "retire_ms", ctx)
            if retire_node.value is None:
                if index != count - 1:
                    _fail(f"keys.json service {sid!r}: only the current "
                          f"(last) version may have retire_ms=null",
                          field="retire_ms", line=retire_node.line)
                retire_ms = -1
            else:
                retire_ms = _as_int(retire_node, "retire_ms", ctx,
                                    minimum=0)
            versions.append(KeyVersion(version=version, secret=secret,
                                       retire_ms=retire_ms))
        keys[sid] = tuple(versions)
    for sid in services:
        if sid not in keys:
            _fail(f"keys.json: missing key versions for service {sid!r}",
                  field=sid, line=keys_node.line)
    return keys


def _load_requests(path, services):
    root = parse_file(path)
    reqs_node = _require_field(root, "requests", "requests.json")
    if not isinstance(reqs_node.value, list):
        _fail("requests.json: 'requests' must be an array",
              field="requests", line=reqs_node.line)
    requests = []
    prev_arrival = None
    for index, item in enumerate(reqs_node.value):
        ctx = f"requests.json entry (line {item.line})"
        seq = _as_int(_require_field(item, "seq", ctx), "seq", ctx,
                      minimum=1)
        expected_seq = index + 1
        if seq != expected_seq:
            _fail(f"requests.json: seq must be contiguous; expected "
                  f"{expected_seq}, got {seq}", field="seq",
                  line=item.line)
        client_key = _as_str(_require_field(item, "client_key", ctx),
                             "client_key", ctx)
        service_id = _as_str(_require_field(item, "service_id", ctx),
                             "service_id", ctx)
        if service_id not in services:
            _fail(f"request seq={seq}: unknown target service "
                  f"{service_id!r}", field="service_id", line=item.line)
        arrival = _as_int(_require_field(item, "arrival_ms", ctx),
                          "arrival_ms", ctx, minimum=0)
        if prev_arrival is not None and arrival < prev_arrival:
            _fail(f"request seq={seq}: arrival_ms must be monotonically "
                  f"non-decreasing ({arrival} after {prev_arrival})",
                  field="arrival_ms", line=item.line)
        prev_arrival = arrival
        bucket_key = _as_str(_require_field(item, "bucket_key", ctx),
                             "bucket_key", ctx)
        if bucket_key not in services[service_id].bucket_keys:
            _fail(f"request seq={seq}: bucket_key {bucket_key!r} not "
                  f"declared by service {service_id!r}",
                  field="bucket_key", line=item.line)
        cred_version = _as_int(_require_field(item, "cred_version", ctx),
                               "cred_version", ctx, minimum=1)
        signature = _as_str(_require_field(item, "signature", ctx),
                            "signature", ctx)
        requests.append(RequestSpec(
            seq=seq, client_key=client_key, service_id=service_id,
            arrival_ms=arrival, bucket_key=bucket_key,
            cred_version=cred_version, signature=signature))
    if not requests:
        _fail("requests.json: at least one request is required",
              field="requests", line=reqs_node.line)
    return tuple(requests)


def _file_digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_spec(data_dir):
    """Load services.json, keys.json, requests.json from data_dir."""
    services_path = os.path.join(data_dir, "services.json")
    keys_path = os.path.join(data_dir, "keys.json")
    requests_path = os.path.join(data_dir, "requests.json")
    for path in (services_path, keys_path, requests_path):
        if not os.path.isfile(path):
            _fail(f"required spec file not found: {path}")
    services = _load_services(services_path)
    keys = _load_keys(keys_path, services)
    requests = _load_requests(requests_path, services)
    return Spec(
        services=services, keys=keys, requests=requests,
        services_digest=_file_digest(services_path),
        keys_digest=_file_digest(keys_path),
        requests_digest=_file_digest(requests_path))


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
