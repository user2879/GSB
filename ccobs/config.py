"""会话配置与对端应答脚本的本地 JSON 加载及强校验。

任何非法输入均以 E_CONFIG 整体拒绝, 并指明首个非法条目的位置。
"""
import hashlib
import json
from dataclasses import dataclass, field

from .errors import E_CONFIG, BizError

SCRIPT_ACTIONS = ("ack", "dup_ack", "loss", "delay", "reorder", "gap")


@dataclass(frozen=True)
class SessionConfig:
    initial_seq: int
    cwnd_max: int          # 窗口上限(字节)
    mss: int               # 最大报文段(字节)
    scale_factor: int      # 会话缩放因子 0..6, 定点单位 = 10 ** scale_factor
    session_budget: int    # 会话预算(最大事件数)
    batch_size: int        # 每批事件数
    window_batches: int    # 每个观测窗口包含的批数
    dup_ack_limit: int     # 单窗口内重复确认声明上限
    peer_host: str = "127.0.0.1"
    peer_port: int = 39211

    @property
    def unit(self):
        return 10 ** self.scale_factor

    @property
    def window_events(self):
        return self.batch_size * self.window_batches


@dataclass(frozen=True)
class ScriptEntry:
    index: int             # 脚本内序号(0 起)
    event_seq: int
    action: str
    delay_ms: int = 0
    skip: int = 0          # 仅 gap 动作使用: 应答序号偏移量


@dataclass(frozen=True)
class PeerScript:
    entries: tuple
    digest: str

    def by_seq(self):
        return {e.event_seq: e for e in self.entries}


def _fail(message, **details):
    raise BizError(E_CONFIG, message, **details)


def _require_int(obj, key, where, minimum=None, maximum=None):
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("field must be an integer", entry=where, field=key)
    if minimum is not None and value < minimum:
        _fail("field below minimum", entry=where, field=key, value=value, minimum=minimum)
    if maximum is not None and value > maximum:
        _fail("field above maximum", entry=where, field=key, value=value, maximum=maximum)
    return value


def _load_json(path, where):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        _fail("file not found", entry=where, path=str(path))
    except json.JSONDecodeError as exc:
        _fail("invalid json", entry=where, line=exc.lineno, reason=exc.msg)


def load_session_config(path):
    raw = _load_json(path, "session_config")
    if not isinstance(raw, dict):
        _fail("config root must be an object", entry="session_config")
    cfg = SessionConfig(
        initial_seq=_require_int(raw, "initial_seq", "session_config", minimum=0),
        cwnd_max=_require_int(raw, "cwnd_max", "session_config", minimum=1),
        mss=_require_int(raw, "mss", "session_config", minimum=1),
        scale_factor=_require_int(raw, "scale_factor", "session_config", minimum=0, maximum=6),
        session_budget=_require_int(raw, "session_budget", "session_config", minimum=1),
        batch_size=_require_int(raw, "batch_size", "session_config", minimum=1),
        window_batches=_require_int(raw, "window_batches", "session_config", minimum=1),
        dup_ack_limit=_require_int(raw, "dup_ack_limit", "session_config", minimum=1),
        peer_host=str(raw.get("peer_host", "127.0.0.1")),
        peer_port=_require_int(raw, "peer_port", "session_config", minimum=1024, maximum=65535),
    )
    if cfg.cwnd_max < cfg.mss:
        _fail("cwnd_max must be >= mss", entry="session_config",
              field="cwnd_max", value=cfg.cwnd_max, mss=cfg.mss)
    return cfg


def load_peer_script(path, config):
    raw = _load_json(path, "peer_script")
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        _fail("script root must be an object with an entries array", entry="peer_script")
    entries = []
    expected_seq = config.initial_seq
    for index, item in enumerate(raw["entries"]):
        where = "entries[%d]" % index
        if not isinstance(item, dict):
            _fail("script entry must be an object", entry=where)
        seq = _require_int(item, "event_seq", where, minimum=0)
        if seq != expected_seq:
            _fail("script event_seq must be continuous from initial_seq",
                  entry=where, field="event_seq", value=seq, expected=expected_seq)
        action = item.get("action")
        if action not in SCRIPT_ACTIONS:
            _fail("unknown script action", entry=where, field="action", value=action)
        delay_ms = _require_int(item, "delay_ms", where, minimum=0) \
            if "delay_ms" in item else 0
        skip = 0
        if action == "gap":
            skip = _require_int(item, "skip", where, minimum=1)
        entries.append(ScriptEntry(index=index, event_seq=seq, action=action,
                                   delay_ms=delay_ms, skip=skip))
        expected_seq += 1
    if not entries:
        _fail("script must contain at least one entry", entry="peer_script")
    if len(entries) > config.session_budget:
        _fail("script exceeds session budget", entry="peer_script",
              value=len(entries), budget=config.session_budget)
    digest = hashlib.sha256(
        json.dumps([e.__dict__ for e in entries], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return PeerScript(entries=tuple(entries), digest=digest)


def config_digest(config):
    payload = {
        "initial_seq": config.initial_seq,
        "cwnd_max": config.cwnd_max,
        "mss": config.mss,
        "scale_factor": config.scale_factor,
        "session_budget": config.session_budget,
        "batch_size": config.batch_size,
        "window_batches": config.window_batches,
        "dup_ack_limit": config.dup_ack_limit,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
