"""本地 JSON 声明加载与强校验（需求 1）。

校验范围：工作进程声明、请求流（投递事件）、配额事件与归并规则。
任一不合法都抛出 :class:`SpecError`（错误码 E_SPEC），携带首个非法
位置（JSON Pointer 风格）与字段名。
"""

import json

from .errors import SpecError

HOST_LITERAL = "127.0.0.1"
ALLOWED_MERGE_POLICIES = ("exclude_superseded",)


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class WorkerSpec:
    def __init__(self, worker_id, port, concurrency, param_version):
        self.worker_id = worker_id
        self.host = HOST_LITERAL
        self.port = port
        self.concurrency = concurrency
        self.param_version = param_version

    def to_dict(self):
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "concurrency": self.concurrency,
            "param_version": self.param_version,
        }


class Delivery:
    """一次请求投递事件；``replaces`` 指向被覆盖的历史请求。"""

    def __init__(self, request_id, arrival_seq, value, replaces):
        self.request_id = request_id
        self.arrival_seq = arrival_seq
        self.value = value
        self.replaces = replaces

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "arrival_seq": self.arrival_seq,
            "value": self.value,
            "replaces": self.replaces,
        }


class QuotaEvent:
    def __init__(self, event_id, worker_id, concurrency):
        self.event_id = event_id
        self.worker_id = worker_id
        self.concurrency = concurrency


class Spec:
    """校验通过后的不可变声明视图。"""

    def __init__(
        self,
        workers,
        deliveries,
        quota_events,
        input_dim,
        hidden_dim,
        seed,
        batch_size,
        lease_ms,
        global_max_active,
        supersede_field,
        merge_policy,
        connect_timeout_ms,
        read_timeout_ms,
    ):
        self.workers = workers
        self.deliveries = deliveries
        self.quota_events = quota_events
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.batch_size = batch_size
        self.lease_ms = lease_ms
        self.global_max_active = global_max_active
        self.supersede_field = supersede_field
        self.merge_policy = merge_policy
        self.connect_timeout_ms = connect_timeout_ms
        self.read_timeout_ms = read_timeout_ms

        self._worker_by_id = {w.worker_id: w for w in workers}
        self._delivery_by_id = {d.request_id: d for d in deliveries}
        self._effective_quota = {w.worker_id: w.concurrency for w in workers}
        for event in quota_events:
            self._effective_quota[event.worker_id] = event.concurrency

    @property
    def worker_ids_sorted(self):
        return [w.worker_id for w in self.workers]

    def worker(self, worker_id):
        return self._worker_by_id[worker_id]

    def all_request_ids(self):
        """包含已被覆盖请求在内的全部请求标识（供回执 E_REF 校验）。"""
        return set(self._delivery_by_id)

    def effective_quota(self, worker_id):
        return self._effective_quota[worker_id]

    def effective_deliveries(self):
        """应用归并规则后的有效请求，按到达序号稳定排序。"""
        superseded = set()
        for delivery in self.deliveries:
            if delivery.replaces is not None:
                superseded.add(delivery.replaces)
        active = [d for d in self.deliveries if d.request_id not in superseded]
        return sorted(active, key=lambda d: d.arrival_seq)

    def supersession_map(self):
        """request_id -> 被它覆盖的请求标识（没有则为 None）。"""
        return {d.request_id: d.replaces for d in self.deliveries}


def load_spec_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"JSON 解析失败: {exc.msg}", location=f"#{exc.pos}", field="<json>")
    return parse_spec(raw)


def parse_spec(raw):
    if not isinstance(raw, dict):
        raise SpecError("根节点必须是对象", location="/", field="<root>")

    model = _require_object(raw, "model", "/")
    input_dim = _require_pos_int(model, "input_dim", "/model")
    hidden_dim = _require_pos_int(model, "hidden_dim", "/model")
    seed = _require_int(model, "seed", "/model")
    batch_size = _require_pos_int(model, "batch_size", "/model")

    limits = _require_object(raw, "limits", "/")
    lease_ms = _require_pos_int(limits, "lease_ms", "/limits")
    global_max_active = _require_pos_int(limits, "global_max_active_leases", "/limits")
    connect_timeout_ms = _require_pos_int(limits, "connect_timeout_ms", "/limits")
    read_timeout_ms = _require_pos_int(limits, "read_timeout_ms", "/limits")

    merge_rule = _require_object(raw, "merge_rule", "/")
    supersede_field = merge_rule.get("supersede_field", "replaces")
    if not isinstance(supersede_field, str) or not supersede_field:
        raise SpecError("归并覆盖字段名必须是非空字符串", location="/merge_rule", field="supersede_field")
    merge_policy = merge_rule.get("policy", "exclude_superseded")
    if merge_policy not in ALLOWED_MERGE_POLICIES:
        raise SpecError(
            f"不支持的归并策略 {merge_policy!r}，允许值 {ALLOWED_MERGE_POLICIES}",
            location="/merge_rule",
            field="policy",
        )

    workers = _parse_workers(raw)
    worker_ids = {w.worker_id for w in workers}
    deliveries = _parse_deliveries(raw, supersede_field)
    quota_events = _parse_quota_events(raw, worker_ids)

    return Spec(
        workers=workers,
        deliveries=deliveries,
        quota_events=quota_events,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seed=seed,
        batch_size=batch_size,
        lease_ms=lease_ms,
        global_max_active=global_max_active,
        supersede_field=supersede_field,
        merge_policy=merge_policy,
        connect_timeout_ms=connect_timeout_ms,
        read_timeout_ms=read_timeout_ms,
    )


def _require_object(raw, key, base):
    value = raw.get(key)
    if not isinstance(value, dict):
        raise SpecError(f"{key} 必须是对象", location=base, field=key)
    return value


def _require_int(obj, key, base):
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecError(f"{key} 必须是整数", location=base, field=key)
    return value


def _require_pos_int(obj, key, base):
    value = obj.get(key)
    if not _is_pos_int(value):
        raise SpecError(f"{key} 必须是正整数", location=base, field=key)
    return value


def _require_identifier(obj, key, base):
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise SpecError(f"{key} 必须是非空字符串", location=base, field=key)
    return value


def _parse_workers(raw):
    value = raw.get("workers")
    if not isinstance(value, list) or not value:
        raise SpecError("workers 必须是非空数组", location="/", field="workers")
    workers = []
    seen_ids = set()
    seen_ports = set()
    for index, item in enumerate(value):
        base = f"/workers/{index}"
        if not isinstance(item, dict):
            raise SpecError("工作进程声明必须是对象", location=base, field="<item>")
        worker_id = _require_identifier(item, "worker_id", base)
        if worker_id in seen_ids:
            raise SpecError(f"工作进程标识 {worker_id!r} 重复", location=base, field="worker_id")
        seen_ids.add(worker_id)

        host = item.get("host")
        if host != HOST_LITERAL:
            raise SpecError(
                f"host 必须是回环字面量 {HOST_LITERAL!r}", location=base, field="host"
            )
        port = item.get("port")
        if not _is_pos_int(port) or not (1024 <= port <= 65535):
            raise SpecError("port 必须是 1024..65535 的正整数", location=base, field="port")
        if port in seen_ports:
            raise SpecError(f"回环端口 {port} 被多个工作进程重复声明", location=base, field="port")
        seen_ports.add(port)

        concurrency = _require_pos_int(item, "concurrency", base)
        param_version = _require_pos_int(item, "param_version", base)
        workers.append(WorkerSpec(worker_id, port, concurrency, param_version))
    return workers


def _parse_deliveries(raw, supersede_field):
    value = raw.get("deliveries")
    if not isinstance(value, list) or not value:
        raise SpecError("deliveries 必须是非空数组", location="/", field="deliveries")
    deliveries = []
    seen_ids = set()
    seen_seqs = set()
    for index, item in enumerate(value):
        base = f"/deliveries/{index}"
        if not isinstance(item, dict):
            raise SpecError("投递事件必须是对象", location=base, field="<item>")
        request_id = _require_identifier(item, "request_id", base)
        if request_id in seen_ids:
            raise SpecError(f"请求标识 {request_id!r} 重复", location=base, field="request_id")
        seen_ids.add(request_id)

        arrival_seq = item.get("arrival_seq")
        if not _is_pos_int(arrival_seq):
            raise SpecError("arrival_seq 必须是正整数", location=base, field="arrival_seq")
        if arrival_seq in seen_seqs:
            raise SpecError(f"到达序号 {arrival_seq} 重复", location=base, field="arrival_seq")
        seen_seqs.add(arrival_seq)

        payload_value = item.get("value")
        if not _is_nonneg_int(payload_value):
            raise SpecError("value 必须是非负整数（定点输入）", location=base, field="value")

        replaces = item.get(supersede_field)
        if replaces is not None and (not isinstance(replaces, str) or not replaces):
            raise SpecError(
                f"{supersede_field} 必须为 null 或非空字符串",
                location=base,
                field=supersede_field,
            )
        deliveries.append(Delivery(request_id, arrival_seq, payload_value, replaces))

    for delivery in deliveries:
        if delivery.replaces is not None and delivery.replaces not in seen_ids:
            raise SpecError(
                f"覆盖引用 {delivery.replaces!r} 指向不存在的请求",
                location="/deliveries",
                field="replaces",
            )
        if delivery.replaces == delivery.request_id:
            raise SpecError("请求不能覆盖自身", location="/deliveries", field="replaces")
    return deliveries


def _parse_quota_events(raw, worker_ids):
    value = raw.get("quota_events", [])
    if not isinstance(value, list):
        raise SpecError("quota_events 必须是数组", location="/", field="quota_events")
    events = []
    seen_event_ids = set()
    for index, item in enumerate(value):
        base = f"/quota_events/{index}"
        if not isinstance(item, dict):
            raise SpecError("配额事件必须是对象", location=base, field="<item>")
        event_id = _require_identifier(item, "event_id", base)
        if event_id in seen_event_ids:
            raise SpecError(f"配额事件标识 {event_id!r} 重复", location=base, field="event_id")
        seen_event_ids.add(event_id)
        worker_id = _require_identifier(item, "worker_id", base)
        if worker_id not in worker_ids:
            raise SpecError(
                f"配额事件引用了未声明的工作进程 {worker_id!r}",
                location=base,
                field="worker_id",
            )
        concurrency = _require_pos_int(item, "concurrency", base)
        events.append(QuotaEvent(event_id, worker_id, concurrency))
    return events
