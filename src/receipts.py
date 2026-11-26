"""回执三格式 r1/r2/r3 的编解码与强校验。

行格式（竖线分隔，字段宽度固定）：
  r1|<batch_id>|<request_id>|<output_digest>
  r2|<batch_id>|<request_id>|<param_version>|<elapsed_ms>|<worker_id>|<output_digest>
  r3|<batch_id>|<request_id>|<param_version>|<elapsed_ms>|<worker_id>|<rank>|<dup_of>|<output_digest>

字段宽度约定：
  batch_id      "batch-" + 4 位数字
  request_id    "req-" + 4 位数字
  param_version 1..6 位非负整数
  elapsed_ms    1..9 位非负整数（不得为负）
  worker_id     1..16 位 [a-z0-9_-]
  rank          1..4 位非负整数，批内唯一且不越界
  dup_of        空或 request_id（须指向批内存在的请求，允许指向自身的重复投递）
  output_digest 64 位小写十六进制

字段级错误报 E_FIELD，引用级错误（名次唯一/越界、dup_of 悬空）报 E_REF，
均定位首个非法回执的源行号与字段名。
"""

import re
from dataclasses import dataclass, field

from .errors import FieldError, RefError

RE_BATCH_ID = re.compile(r"^batch-\d{4}$")
RE_REQUEST_ID = re.compile(r"^req-\d{4}$")
RE_WORKER_ID = re.compile(r"^[a-z0-9_-]{1,16}$")
RE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
RE_PARAM_VERSION = re.compile(r"^\d{1,6}$")
RE_ELAPSED = re.compile(r"^\d{1,9}$")
RE_RANK = re.compile(r"^\d{1,4}$")

FORMAT_FIELDS = {
    1: ("batch_id", "request_id", "output_digest"),
    2: ("batch_id", "request_id", "param_version", "elapsed_ms", "worker_id", "output_digest"),
    3: ("batch_id", "request_id", "param_version", "elapsed_ms", "worker_id",
        "rank", "dup_of", "output_digest"),
}

ALL_FIELDS = ("batch_id", "request_id", "output_digest", "param_version",
              "elapsed_ms", "worker_id", "rank", "dup_of")


@dataclass
class Receipt:
    src_format: int
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int = None
    elapsed_ms: int = None
    worker_id: str = None
    rank: int = None
    dup_of: str = None
    src_file: str = "<mem>"
    src_line: int = 0
    raw: str = ""
    real_fields: frozenset = field(default=frozenset)

    def real(self, name):
        return name in self.real_fields


def _fail(line_no, field_name, message):
    raise FieldError(message, line=line_no, field=field_name)


def parse_line(line, line_no=0, src_file="<mem>"):
    raw = line.rstrip("\n")
    parts = raw.split("|")
    tag = parts[0] if parts else ""
    if tag not in ("r1", "r2", "r3"):
        _fail(line_no, "format", f"未知回执格式标签: {tag!r}")
    fmt = int(tag[1])
    fields = FORMAT_FIELDS[fmt]
    if len(parts) - 1 != len(fields):
        _fail(line_no, "format",
              f"{tag} 应有 {len(fields)} 个字段，实际 {len(parts) - 1} 个")
    values = dict(zip(fields, parts[1:]))

    if not RE_BATCH_ID.match(values["batch_id"]):
        _fail(line_no, "batch_id", f"批次标识宽度非法: {values['batch_id']!r}")
    if not RE_REQUEST_ID.match(values["request_id"]):
        _fail(line_no, "request_id", f"请求标识宽度非法: {values['request_id']!r}")
    if not RE_DIGEST.match(values["output_digest"]):
        _fail(line_no, "output_digest", "输出摘要须为 64 位小写十六进制")

    receipt = Receipt(
        src_format=fmt,
        batch_id=values["batch_id"],
        request_id=values["request_id"],
        output_digest=values["output_digest"],
        src_file=src_file,
        src_line=line_no,
        raw=raw,
        real_fields=frozenset(fields),
    )

    if fmt >= 2:
        if not RE_PARAM_VERSION.match(values["param_version"]):
            _fail(line_no, "param_version", "参数版本序号须为 1..6 位非负整数")
        if not RE_ELAPSED.match(values["elapsed_ms"]):
            _fail(line_no, "elapsed_ms", "耗时毫秒须为 1..9 位非负整数（不得为负）")
        if not RE_WORKER_ID.match(values["worker_id"]):
            _fail(line_no, "worker_id", f"工作进程标识非法: {values['worker_id']!r}")
        receipt.param_version = int(values["param_version"])
        receipt.elapsed_ms = int(values["elapsed_ms"])
        receipt.worker_id = values["worker_id"]

    if fmt >= 3:
        if not RE_RANK.match(values["rank"]):
            _fail(line_no, "rank", "名次须为 1..4 位非负整数")
        dup_of = values["dup_of"]
        if dup_of and not RE_REQUEST_ID.match(dup_of):
            _fail(line_no, "dup_of", f"重复投递引用宽度非法: {dup_of!r}")
        receipt.rank = int(values["rank"])
        receipt.dup_of = dup_of or None

    return receipt


def validate_refs(receipts):
    """引用级校验：名次批内唯一且不越界；dup_of 指向批内存在的请求。"""
    batch_sizes = {}
    batch_requests = {}
    for rec in receipts:
        batch_sizes[rec.batch_id] = batch_sizes.get(rec.batch_id, 0) + 1
        batch_requests.setdefault(rec.batch_id, set()).add(rec.request_id)
    seen_ranks = {}
    for rec in receipts:
        if rec.src_format < 3:
            continue
        key = (rec.batch_id, rec.rank)
        if rec.rank >= batch_sizes[rec.batch_id]:
            raise RefError(
                f"名次 {rec.rank} 越界（批 {rec.batch_id} 规模 {batch_sizes[rec.batch_id]}）",
                line=rec.src_line, field="rank")
        if key in seen_ranks:
            raise RefError(
                f"名次 {rec.rank} 在批 {rec.batch_id} 内重复（首次见于行 {seen_ranks[key]}）",
                line=rec.src_line, field="rank")
        seen_ranks[key] = rec.src_line
        if rec.dup_of is not None and rec.dup_of not in batch_requests[rec.batch_id]:
            raise RefError(
                f"重复投递引用 {rec.dup_of} 在批 {rec.batch_id} 内不存在",
                line=rec.src_line, field="dup_of")


def parse_text(text, src_file="<mem>"):
    receipts = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        receipts.append(parse_line(line, line_no=line_no, src_file=src_file))
    validate_refs(receipts)
    return receipts


def parse_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return parse_text(fh.read(), src_file=str(path))


def format_line(receipt):
    if receipt.src_format == 1:
        return f"r1|{receipt.batch_id}|{receipt.request_id}|{receipt.output_digest}"
    if receipt.src_format == 2:
        return (f"r2|{receipt.batch_id}|{receipt.request_id}|{receipt.param_version}"
                f"|{receipt.elapsed_ms}|{receipt.worker_id}|{receipt.output_digest}")
    return (f"r3|{receipt.batch_id}|{receipt.request_id}|{receipt.param_version}"
            f"|{receipt.elapsed_ms}|{receipt.worker_id}|{receipt.rank}"
            f"|{receipt.dup_of or ''}|{receipt.output_digest}")
