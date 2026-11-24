"""声明文件与 NDJSON 事件流的加载及强校验。

任何不合法输入都抛出 :class:`SpecError`（``E_SPEC``），并指明首个违规字段与行号。
校验顺序固定，因此“首个违规点”可重复复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecError
from .protocol import ACTIONS

MIN_PORT = 10000
MAX_PORT = 65535
VALID_ACTIONS = frozenset(ACTIONS)
DOMAIN_FIELDS = ("domain_id", "host", "port", "slots", "lease_cap", "lease_ms", "renew_ms", "seed")
INSTANCE_FIELDS = ("instance_id", "domain_id", "priority", "demand")
EVENT_FIELDS = ("seq", "arrival_ms", "instance_id", "action")


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    host: str
    port: int
    slots: int
    lease_cap: int
    lease_ms: int
    renew_ms: int
    seed: int


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    domain_id: str
    priority: int
    demand: int


@dataclass(frozen=True)
class Event:
    seq: int
    arrival_ms: int
    instance_id: str
    action: str


@dataclass(frozen=True)
class Session:
    """会话级参数：全局并发租约上限、定点缩放幂次、实时节拍等。"""

    global_lease_cap: int
    scale_power: int
    pacing: float
    batch_size: int

    @property
    def scale_factor(self) -> int:
        return 10 ** self.scale_power


@dataclass
class SpecBundle:
    domains: list[DomainSpec]
    instances: list[InstanceSpec]
    events: list[Event]
    session: Session
    spec_path: Path
    events_path: Path
    session_path: Path

    def domain(self, domain_id: str) -> DomainSpec:
        return self._domain_map[domain_id]

    def instance(self, instance_id: str) -> InstanceSpec:
        return self._instance_map[instance_id]

    def domain_instances(self, domain_id: str) -> list[InstanceSpec]:
        return self._by_domain[domain_id]

    def __post_init__(self) -> None:
        self._domain_map = {d.domain_id: d for d in self.domains}
        self._instance_map = {i.instance_id: i for i in self.instances}
        by_domain: dict[str, list[InstanceSpec]] = {d.domain_id: [] for d in self.domains}
        for inst in self.instances:
            by_domain[inst.domain_id].append(inst)
        self._by_domain = by_domain


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _locate_line(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def _fail(message: str, field: str, text: str | None = None, needle: str | None = None,
          line: int | None = None) -> None:
    if line is None and text is not None and needle is not None:
        line = _locate_line(text, needle)
    raise SpecError(message, field=field, line=line if line is not None else 1)


def _require_keys(obj: dict, required: tuple[str, ...], text: str, needle: str, where: str) -> None:
    for key in required:
        if key not in obj:
            _fail(f"{where}缺少必填字段 {key}", field=key, text=text, needle=needle)


def load_spec(spec_path: str | Path) -> tuple[list[DomainSpec], list[InstanceSpec], str, Path]:
    path = Path(spec_path)
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError("声明 JSON 语法错误", field="<json>", line=exc.lineno) from exc
    if not isinstance(raw, dict):
        _fail("声明文件根节点必须为 JSON 对象", field="<root>", line=1)

    domains_raw = raw.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        _fail("domains 必须为非空数组", field="domains", line=1)
    instances_raw = raw.get("instances")
    if not isinstance(instances_raw, list) or not instances_raw:
        _fail("instances 必须为非空数组", field="instances", line=1)
    events_file = raw.get("events_file")
    if not isinstance(events_file, str) or not events_file:
        _fail("events_file 必须为非空字符串（相对声明文件目录解析）", field="events_file", line=1)

    domains: list[DomainSpec] = []
    seen_domain_ids: set[str] = set()
    seen_ports: set[int] = set()
    for item in domains_raw:
        if not isinstance(item, dict):
            _fail("domain 条目必须为对象", field="domains", text=text, needle='"domains"')
        _require_keys(item, DOMAIN_FIELDS, text, str(item.get("domain_id", '"domains"')), "domain 条目")
        did = item["domain_id"]
        needle = f'"{did}"'
        if not isinstance(did, str) or not did:
            _fail("domain_id 必须为非空字符串", field="domain_id", text=text, needle='"domains"')
        if did in seen_domain_ids:
            _fail(f"域标识 {did} 重复", field="domain_id", text=text, needle=needle)
        seen_domain_ids.add(did)
        if item["host"] != "127.0.0.1":
            _fail(f"域 {did} 的 host 必须是回环字面量 127.0.0.1", field="host", text=text, needle=needle)
        for key in ("port", "slots", "lease_cap", "lease_ms", "renew_ms", "seed"):
            if not _is_plain_int(item[key]):
                _fail(f"域 {did} 的 {key} 必须为整数", field=key, text=text, needle=needle)
        port = item["port"]
        if not (MIN_PORT <= port <= MAX_PORT):
            _fail(f"域 {did} 端口 {port} 不在高位回环段 {MIN_PORT}-{MAX_PORT}", field="port",
                  text=text, needle=needle)
        if port in seen_ports:
            _fail(f"域 {did} 端口 {port} 全局重复", field="port", text=text, needle=needle)
        seen_ports.add(port)
        for key in ("slots", "lease_cap", "lease_ms", "renew_ms"):
            if item[key] <= 0:
                _fail(f"域 {did} 的 {key} 必须为正整数", field=key, text=text, needle=needle)
        if item["renew_ms"] >= item["lease_ms"]:
            _fail(f"域 {did} 的 renew_ms 必须严格小于 lease_ms", field="renew_ms", text=text, needle=needle)
        domains.append(DomainSpec(
            domain_id=did, host=item["host"], port=port, slots=item["slots"],
            lease_cap=item["lease_cap"], lease_ms=item["lease_ms"],
            renew_ms=item["renew_ms"], seed=item["seed"],
        ))

    instances: list[InstanceSpec] = []
    seen_instance_ids: set[str] = set()
    domain_slots = {d.domain_id: d.slots for d in domains}
    for item in instances_raw:
        if not isinstance(item, dict):
            _fail("instance 条目必须为对象", field="instances", text=text, needle='"instances"')
        _require_keys(item, INSTANCE_FIELDS, text, str(item.get("instance_id", '"instances"')), "instance 条目")
        iid = item["instance_id"]
        needle = f'"{iid}"'
        if not isinstance(iid, str) or not iid:
            _fail("instance_id 必须为非空字符串", field="instance_id", text=text, needle='"instances"')
        if iid in seen_instance_ids:
            _fail(f"实例标识 {iid} 重复", field="instance_id", text=text, needle=needle)
        seen_instance_ids.add(iid)
        if item["domain_id"] not in seen_domain_ids:
            _fail(f"实例 {iid} 所属域 {item['domain_id']} 不存在", field="domain_id", text=text, needle=needle)
        if not _is_plain_int(item["priority"]):
            _fail(f"实例 {iid} 的 priority 必须为整数", field="priority", text=text, needle=needle)
        if not _is_plain_int(item["demand"]) or item["demand"] <= 0:
            _fail(f"实例 {iid} 的 demand 必须为正整数", field="demand", text=text, needle=needle)
        if item["demand"] > domain_slots[item["domain_id"]]:
            _fail(f"实例 {iid} 的 demand 超过所属域槽位总量", field="demand", text=text, needle=needle)
        instances.append(InstanceSpec(
            instance_id=iid, domain_id=item["domain_id"],
            priority=item["priority"], demand=item["demand"],
        ))

    events_path = (path.parent / events_file).resolve()
    return domains, instances, events_file, events_path


def load_events(events_path: str | Path, known_instances: set[str]) -> list[Event]:
    path = Path(events_path)
    events: list[Event] = []
    expected_seq = 1
    last_arrival = -1
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise SpecError("NDJSON 不允许空行", field="<line>", line=lineno)
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError("事件 NDJSON 语法错误", field="<json>", line=lineno) from exc
            if not isinstance(item, dict):
                raise SpecError("事件必须为 JSON 对象", field="<root>", line=lineno)
            for key in EVENT_FIELDS:
                if key not in item:
                    raise SpecError(f"事件缺少必填字段 {key}", field=key, line=lineno)
            if not _is_plain_int(item["seq"]):
                raise SpecError("事件 seq 必须为整数", field="seq", line=lineno)
            if item["seq"] != expected_seq:
                raise SpecError(
                    f"事件序号不连续：期望 {expected_seq} 实际 {item['seq']}", field="seq", line=lineno)
            if not _is_plain_int(item["arrival_ms"]) or item["arrival_ms"] < 0:
                raise SpecError("arrival_ms 必须为非负整数", field="arrival_ms", line=lineno)
            if item["arrival_ms"] < last_arrival:
                raise SpecError(
                    f"到达时刻 {item['arrival_ms']} 早于上一事件 {last_arrival}，必须单调不减",
                    field="arrival_ms", line=lineno)
            last_arrival = item["arrival_ms"]
            if not isinstance(item["instance_id"], str) or item["instance_id"] not in known_instances:
                raise SpecError(
                    f"事件实例 {item['instance_id']} 未在声明中注册", field="instance_id", line=lineno)
            if item["action"] not in VALID_ACTIONS:
                raise SpecError(
                    f"动作 {item['action']} 非法，可选 {sorted(VALID_ACTIONS)}", field="action", line=lineno)
            events.append(Event(
                seq=item["seq"], arrival_ms=item["arrival_ms"],
                instance_id=item["instance_id"], action=item["action"],
            ))
            expected_seq += 1
    if not events:
        raise SpecError("事件流不能为空", field="<file>", line=1)
    return events


def load_session(session_path: str | Path) -> Session:
    path = Path(session_path)
    if not path.exists():
        return Session(global_lease_cap=4, scale_power=2, pacing=0.05, batch_size=32)
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError("会话 JSON 语法错误", field="<json>", line=exc.lineno) from exc
    if not isinstance(raw, dict):
        raise SpecError("会话文件根节点必须为对象", field="<root>", line=1)
    cap = raw.get("global_lease_cap", 4)
    if not _is_plain_int(cap) or cap <= 0:
        raise SpecError("global_lease_cap 必须为正整数", field="global_lease_cap", line=1)
    power = raw.get("scale_power", 2)
    if not _is_plain_int(power) or power not in (0, 1, 2, 3):
        raise SpecError("scale_power 必须是 0/1/2/3（对应 10 的 0 至 3 次幂）",
                        field="scale_power", line=1)
    pacing = raw.get("pacing", 0.05)
    if not isinstance(pacing, (int, float)) or isinstance(pacing, bool) or pacing < 0:
        raise SpecError("pacing 必须为非负数值", field="pacing", line=1)
    batch_size = raw.get("batch_size", 32)
    if not _is_plain_int(batch_size) or batch_size <= 0:
        raise SpecError("batch_size 必须为正整数", field="batch_size", line=1)
    return Session(global_lease_cap=cap, scale_power=power, pacing=float(pacing), batch_size=batch_size)


def load_bundle(spec_path: str | Path, session_path: str | Path | None = None) -> SpecBundle:
    spec_p = Path(spec_path)
    domains, instances, _events_file, events_p = load_spec(spec_p)
    session_p = Path(session_path) if session_path else spec_p.parent / "session.json"
    session = load_session(session_p)
    events = load_events(events_p, {i.instance_id for i in instances})
    bundle = SpecBundle(
        domains=domains, instances=instances, events=events, session=session,
        spec_path=spec_p.resolve(), events_path=events_p, session_path=session_p.resolve(),
    )
    return bundle
