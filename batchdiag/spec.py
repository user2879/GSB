"""Local JSON loading and strict validation of the merge specification."""

from dataclasses import dataclass, field

from .errors import SpecError
from .jsonlocate import LineMap, load


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival: int
    vector: tuple


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    model_dim: int
    hidden_dim: int
    seed: int
    param_version: int
    host: str
    port: int
    batch_size: int
    quota: int


@dataclass(frozen=True)
class MergeRules:
    global_concurrency: int
    socket_timeout_ms: int


@dataclass
class Spec:
    requests: list
    workers: list
    merge_rules: MergeRules
    model_dim: int
    sources: dict = field(default_factory=dict)
    line_maps: dict = field(default_factory=dict)

    def worker(self, worker_id):
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        return None

    def _fail(self, source_key, key_path, reason):
        line_map = self.line_maps.get(source_key)
        line = line_map.line_of(*key_path) if isinstance(line_map, LineMap) else None
        raise SpecError(reason, field=".".join(str(p) for p in key_path),
                        line=line, source=self.sources.get(source_key, source_key))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool))


def load_spec(requests_path, workers_path, rules_path) -> Spec:
    """Load and strictly validate the three local JSON documents.

    Validation stops at the first illegal field and raises :class:`SpecError`
    (``E_SPEC``) carrying the dotted field path and source line.
    """
    documents = {}
    line_maps = {}
    sources = {}
    for key, path in (("requests", requests_path), ("workers", workers_path),
                      ("rules", rules_path)):
        value, line_map = load(str(path))
        documents[key] = value
        line_maps[key] = line_map
        sources[key] = str(path)

    spec = Spec(requests=[], workers=[], merge_rules=None, model_dim=0,
                sources=sources, line_maps=line_maps)

    workers_doc = documents["workers"]
    if not isinstance(workers_doc, list) or not workers_doc:
        spec._fail("workers", (), "workers must be a non-empty array")

    seen_ports = set()
    seen_worker_ids = set()
    model_dim = None
    for index, raw in enumerate(workers_doc):
        base = (index,)
        if not isinstance(raw, dict):
            spec._fail("workers", base, "worker declaration must be an object")
        worker_id = raw.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            spec._fail("workers", base + ("worker_id",),
                       "worker_id must be a non-empty string")
        if worker_id in seen_worker_ids:
            spec._fail("workers", base + ("worker_id",),
                       f"duplicate worker_id {worker_id!r}")
        seen_worker_ids.add(worker_id)

        if not _is_int(raw.get("model_dim")) or raw["model_dim"] <= 0:
            spec._fail("workers", base + ("model_dim",),
                       "model_dim must be a positive integer")
        if not _is_int(raw.get("hidden_dim")) or raw["hidden_dim"] <= 0:
            spec._fail("workers", base + ("hidden_dim",),
                       "hidden_dim must be a positive integer")
        if not _is_int(raw.get("seed")):
            spec._fail("workers", base + ("seed",),
                       "seed must be an integer")
        if not _is_int(raw.get("param_version")) or raw["param_version"] < 1:
            spec._fail("workers", base + ("param_version",),
                       "param_version must be a positive integer")
        if raw.get("host") != "127.0.0.1":
            spec._fail("workers", base + ("host",),
                       "host must be the loopback address 127.0.0.1")
        port = raw.get("port")
        if not _is_int(port) or not (1024 <= port <= 65535):
            spec._fail("workers", base + ("port",),
                       "port must be a high port in [1024, 65535]")
        if port in seen_ports:
            spec._fail("workers", base + ("port",),
                       f"loopback port {port} is used by another worker")
        seen_ports.add(port)
        if not _is_int(raw.get("batch_size")) or raw["batch_size"] <= 0:
            spec._fail("workers", base + ("batch_size",),
                       "batch_size must be a positive integer")
        if not _is_int(raw.get("quota")) or raw["quota"] <= 0:
            spec._fail("workers", base + ("quota",),
                       "quota must be a positive integer")
        if model_dim is None:
            model_dim = raw["model_dim"]
        elif model_dim != raw["model_dim"]:
            spec._fail("workers", base + ("model_dim",),
                       f"model_dim {raw['model_dim']} disagrees with the "
                       f"platform model_dim {model_dim}")

    rules_doc = documents["rules"]
    if not isinstance(rules_doc, dict):
        spec._fail("rules", (), "merge rules must be an object")
    if not _is_int(rules_doc.get("global_concurrency")) \
            or rules_doc["global_concurrency"] <= 0:
        spec._fail("rules", ("global_concurrency",),
                   "global_concurrency must be a positive integer")
    if not _is_int(rules_doc.get("socket_timeout_ms")) \
            or rules_doc["socket_timeout_ms"] <= 0:
        spec._fail("rules", ("socket_timeout_ms",),
                   "socket_timeout_ms must be a positive integer")

    requests_doc = documents["requests"]
    if not isinstance(requests_doc, list) or not requests_doc:
        spec._fail("requests", (), "requests must be a non-empty array")
    seen_ids = set()
    seen_arrivals = set()
    for index, raw in enumerate(requests_doc):
        base = (index,)
        if not isinstance(raw, dict):
            spec._fail("requests", base, "request entry must be an object")
        request_id = raw.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            spec._fail("requests", base + ("request_id",),
                       "request_id must be a non-empty string")
        if request_id in seen_ids:
            spec._fail("requests", base + ("request_id",),
                       f"duplicate request_id {request_id!r}")
        seen_ids.add(request_id)
        arrival = raw.get("arrival")
        if not _is_int(arrival) or arrival < 0:
            spec._fail("requests", base + ("arrival",),
                       "arrival must be a non-negative integer")
        if arrival in seen_arrivals:
            spec._fail("requests", base + ("arrival",),
                       f"duplicate arrival sequence {arrival}")
        seen_arrivals.add(arrival)
        vector = raw.get("vector")
        if not isinstance(vector, list) or not vector:
            spec._fail("requests", base + ("vector",),
                       "vector must be a non-empty array")
        if len(vector) != model_dim:
            spec._fail("requests", base + ("vector",),
                       f"vector dim {len(vector)} != worker model_dim "
                       f"{model_dim}")
        for value_index, value in enumerate(vector):
            if not _is_number(value):
                spec._fail("requests",
                           base + ("vector", value_index),
                           "vector elements must be numbers")

    spec.model_dim = model_dim
    spec.workers = [
        WorkerSpec(
            worker_id=raw["worker_id"], model_dim=raw["model_dim"],
            hidden_dim=raw["hidden_dim"], seed=raw["seed"],
            param_version=raw["param_version"], host=raw["host"],
            port=raw["port"], batch_size=raw["batch_size"],
            quota=raw["quota"],
        )
        for raw in workers_doc
    ]
    spec.requests = [
        RequestSpec(request_id=raw["request_id"], arrival=raw["arrival"],
                    vector=tuple(raw["vector"]))
        for raw in requests_doc
    ]
    spec.merge_rules = MergeRules(
        global_concurrency=rules_doc["global_concurrency"],
        socket_timeout_ms=rules_doc["socket_timeout_ms"],
    )
    return spec
