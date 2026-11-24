"""工作器 / 请求流 / 归并规则的本地 JSON 加载与强校验。

校验在第一个违规项处抛出 ``SpecError``（E_SPEC），定位信息包含
``section``、``index``（或标识）与 ``field``，满足“首个违规项与字段”。
"""

from dataclasses import dataclass, field
import json
import os

from .errors import SpecError

SUPPORTED_QUANTS = ("fp32", "fp16", "int8")
SUPPORTED_PARAM_FORMATS = ("pf1", "pf2")
SUPPORTED_RECEIPT_FORMATS = ("r1", "r2", "r3")
SUPPORTED_RETRY_STRATEGIES = ("immediate", "bounded")


def _is_int(value) -> bool:
    # bool 是 int 的子类，业务上明确拒绝布尔充当整数。
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quant: str
    param_format: str


@dataclass(frozen=True)
class RequestSpec:
    seq: int
    client_key: str
    vector: tuple
    arrival_ms: int


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    worker_ids: tuple
    batch_size: int
    param_version: int
    receipt_formats: tuple
    retry_policy: str
    max_retries: int
    socket_timeout_ms: int


@dataclass(frozen=True)
class RuntimeSpec:
    spec_version: int
    global_concurrency: int
    per_worker_concurrency: int
    workers: tuple
    requests: tuple
    rules: tuple
    source_paths: dict = field(default_factory=dict)

    def worker(self, worker_id: str) -> WorkerSpec:
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        raise SpecError(
            f"未知工作器标识 {worker_id}", section="rule", field="worker_ids"
        )


def _fail(section, field, message, *, index=None, ident=None):
    loc = {"section": section, "field": field}
    if index is not None:
        loc["index"] = index
    if ident is not None:
        loc["ident"] = ident
    raise SpecError(message, **loc)


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SpecError(f"声明文件不存在: {path}", section="file", field="path")
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"JSON 解析失败: {exc.msg}",
            section="file",
            field="json",
            index=exc.lineno,
        )


def _validate_workers(raw):
    if not isinstance(raw, list) or not raw:
        _fail("workers", "workers", "工作器声明必须是非空数组")
    seen_ids, seen_ports = set(), set()
    workers = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail("workers", "<item>", "工作器声明项必须是对象", index=idx)
        wid = item.get("worker_id")
        if not isinstance(wid, str) or not wid:
            _fail("workers", "worker_id", "工作器标识必须是非空字符串", index=idx)
        if wid in seen_ids:
            _fail("workers", "worker_id", f"工作器标识重复: {wid}", index=idx, ident=wid)
        seen_ids.add(wid)
        host = item.get("host")
        if host != "127.0.0.1":
            _fail("workers", "host", "端口必须绑定 127.0.0.1 回环字面量",
                  index=idx, ident=wid)
        port = item.get("port")
        if not _is_int(port) or not (1024 <= port <= 65535):
            _fail("workers", "port", "端口必须是 1024..65535 的整数",
                  index=idx, ident=wid)
        if port in seen_ports:
            _fail("workers", "port", f"回环端口全局重复: {port}",
                  index=idx, ident=wid)
        seen_ports.add(port)
        for fname in ("model_dim", "hidden_dim"):
            if not _is_int(item.get(fname)) or item[fname] <= 0:
                _fail("workers", fname, "模型/隐藏层维度必须是正整数",
                      index=idx, ident=wid)
        if not _is_int(item.get("seed")):
            _fail("workers", "seed", "随机种子必须是整数", index=idx, ident=wid)
        quant = item.get("quant")
        if quant not in SUPPORTED_QUANTS:
            _fail("workers", "quant",
                  f"参数量化口径不在受支持集合 {SUPPORTED_QUANTS}",
                  index=idx, ident=wid)
        param_format = item.get("param_format")
        if param_format not in SUPPORTED_PARAM_FORMATS:
            _fail("workers", "param_format",
                  f"参数格式版本不在受支持集合 {SUPPORTED_PARAM_FORMATS}",
                  index=idx, ident=wid)
        workers.append(
            WorkerSpec(
                worker_id=wid, host=host, port=port,
                model_dim=item["model_dim"], hidden_dim=item["hidden_dim"],
                seed=item["seed"], quant=quant, param_format=param_format,
            )
        )
    return tuple(workers)


def _validate_requests(raw, workers: tuple):
    if not isinstance(raw, list) or not raw:
        _fail("requests", "requests", "请求流必须是非空数组")
    dim_by_worker = {w.worker_id: w.model_dim for w in workers}
    default_dim = workers[0].model_dim
    seen_seq = set()
    requests = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail("requests", "<item>", "请求声明项必须是对象", index=idx)
        seq = item.get("seq")
        if not _is_int(seq):
            _fail("requests", "seq", "请求序号必须是整数", index=idx)
        if seq in seen_seq:
            _fail("requests", "seq", f"请求序号重复: {seq}", index=idx)
        seen_seq.add(seq)
        key = item.get("client_key")
        if not isinstance(key, str) or not key:
            _fail("requests", "client_key", "客户端键必须是非空字符串", index=idx)
        vector = item.get("vector")
        if not isinstance(vector, list) or not vector:
            _fail("requests", "vector", "输入向量必须是非空数组", index=idx)
        if not all(_is_int(v) or isinstance(v, float) for v in vector):
            _fail("requests", "vector", "输入向量元素必须是数值", index=idx)
        target_dim = dim_by_worker.get(item.get("worker_id", ""), default_dim)
        if len(vector) != target_dim:
            _fail("requests", "vector",
                  f"输入向量长度 {len(vector)} 与声明维度 {target_dim} 不一致",
                  index=idx)
        arrival = item.get("arrival_ms")
        if not _is_int(arrival) or arrival < 0:
            _fail("requests", "arrival_ms", "到达毫秒必须是非负整数", index=idx)
        requests.append(
            RequestSpec(seq=seq, client_key=key, vector=tuple(vector),
                        arrival_ms=arrival)
        )
    return tuple(requests)


def _validate_rules(raw, workers: tuple) -> tuple:
    if not isinstance(raw, list) or not raw:
        _fail("rules", "rules", "归并规则必须是非空数组")
    worker_ids = {w.worker_id for w in workers}
    seen_rule = set()
    rules = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail("rules", "<item>", "规则声明项必须是对象", index=idx)
        rid = item.get("rule_id")
        if not isinstance(rid, str) or not rid:
            _fail("rules", "rule_id", "规则标识必须是非空字符串", index=idx)
        if rid in seen_rule:
            _fail("rules", "rule_id", f"规则标识重复: {rid}", index=idx, ident=rid)
        seen_rule.add(rid)
        group = item.get("worker_ids")
        if not isinstance(group, list) or not group:
            _fail("rules", "worker_ids", "工作器集合必须是非空数组",
                  index=idx, ident=rid)
        if any(w not in worker_ids for w in group):
            _fail("rules", "worker_ids", "工作器集合包含未声明工作器",
                  index=idx, ident=rid)
        if len(set(group)) != len(group):
            _fail("rules", "worker_ids", "工作器集合内不得重复", index=idx, ident=rid)
        batch_size = item.get("batch_size")
        if not _is_int(batch_size) or batch_size <= 0:
            _fail("rules", "batch_size", "批大小必须是正整数", index=idx, ident=rid)
        param_version = item.get("param_version")
        if not _is_int(param_version) or param_version <= 0:
            _fail("rules", "param_version", "参数版本序号必须是正整数",
                  index=idx, ident=rid)
        formats = item.get("receipt_formats", [item.get("receipt_format", "r1")])
        if not isinstance(formats, list) or not formats:
            _fail("rules", "receipt_formats", "回执格式版本必须是非空数组",
                  index=idx, ident=rid)
        if any(f not in SUPPORTED_RECEIPT_FORMATS for f in formats):
            _fail("rules", "receipt_formats",
                  f"回执格式版本须属于 {SUPPORTED_RECEIPT_FORMATS}",
                  index=idx, ident=rid)
        policy = item.get("retry_policy", "immediate")
        if policy not in SUPPORTED_RETRY_STRATEGIES:
            _fail("rules", "retry_policy",
                  f"重试策略须属于 {SUPPORTED_RETRY_STRATEGIES}",
                  index=idx, ident=rid)
        max_retries = item.get("max_retries", 2)
        if not _is_int(max_retries) or max_retries < 0:
            _fail("rules", "max_retries", "最大重试次数必须是非负整数",
                  index=idx, ident=rid)
        timeout = item.get("socket_timeout_ms", 2000)
        if not _is_int(timeout) or timeout <= 0:
            _fail("rules", "socket_timeout_ms", "套接字超时须是正整数毫秒",
                  index=idx, ident=rid)
        rules.append(
            RuleSpec(
                rule_id=rid, worker_ids=tuple(group), batch_size=batch_size,
                param_version=param_version, receipt_formats=tuple(formats),
                retry_policy=policy, max_retries=max_retries,
                socket_timeout_ms=timeout,
            )
        )
    return tuple(rules)


def load_spec(workers_path: str, requests_path: str, rules_path: str,
              *, spec_version: int = 1, global_concurrency: int = 2,
              per_worker_concurrency: int = 1) -> RuntimeSpec:
    """加载并强校验三份本地 JSON 声明。"""
    workers_raw = _load_json(workers_path)
    requests_raw = _load_json(requests_path)
    rules_raw = _load_json(rules_path)
    workers = _validate_workers(workers_raw)
    requests = _validate_requests(requests_raw, workers)
    rules = _validate_rules(rules_raw, workers)
    if not _is_int(spec_version) or spec_version <= 0:
        _fail("runtime", "spec_version", "声明版本必须是正整数")
    if not _is_int(global_concurrency) or global_concurrency <= 0:
        _fail("runtime", "global_concurrency", "全局并发上限必须是正整数")
    if not _is_int(per_worker_concurrency) or per_worker_concurrency <= 0:
        _fail("runtime", "per_worker_concurrency", "每工作器并发配额必须是正整数")
    return RuntimeSpec(
        spec_version=spec_version,
        global_concurrency=global_concurrency,
        per_worker_concurrency=per_worker_concurrency,
        workers=workers, requests=requests, rules=rules,
        source_paths={
            "workers": os.path.abspath(workers_path),
            "requests": os.path.abspath(requests_path),
            "rules": os.path.abspath(rules_path),
        },
    )


def spec_to_jsonable(spec: RuntimeSpec) -> dict:
    """把运行时规格转为完全可 JSON 序列化的稳定结构（键排序由摘要层负责）。"""
    return {
        "spec_version": spec.spec_version,
        "global_concurrency": spec.global_concurrency,
        "per_worker_concurrency": spec.per_worker_concurrency,
        "workers": [w.__dict__ for w in spec.workers],
        "requests": [
            {**r.__dict__, "vector": list(r.vector)} for r in spec.requests
        ],
        "rules": [
            {**r.__dict__,
             "worker_ids": list(r.worker_ids),
             "receipt_formats": list(r.receipt_formats)}
            for r in spec.rules
        ],
        "source_paths": dict(spec.source_paths),
    }
