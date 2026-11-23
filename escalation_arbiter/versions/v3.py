"""v3: binary partitioned bundle with headers and offset tables.

File layout (all multi-byte integers are little-endian)::

    magic(4)=ARBV  version(1)=3  flags(1)  partition_count(2)
    global_header: decimals(2) quantum(4) half_life(8)
                    reason_count(2) reason_code_table[n * code(4)]
    partition_header[p]: part_id(2) record_count(4) offset_table_pos(8)
    offset_table[p]: record_offset(8) * record_count
    record:
      tid_len(2) tid(utf8)
      base_ts(8) decision_score(8) flags(1)
      priority_level(2) boundary(1) reason_count(2) reason_bitmap(8)
      evidence_count(2)
      evidence rows: seq(4) ts(8) confidence(4) quantized_score(8)
                     residual(4) etype_len(2) etype role_len(2) role
"""

import struct

from ..models import EvidenceScore, TicketDecision

MAGIC = b"ARBV"
FLAG_BOUNDARY = 1 << 0
FLAG_ESCALATE = 1 << 1


def _partition_of(tid: str, partition_count: int) -> int:
    return (sum(tid.encode("utf-8")) % partition_count)


def write_bundle(path: str, decisions, rules, config) -> dict:
    sidecar = {"version": "v3", "tickets": {}, "losses": []}
    code_table = sorted(rules.reason_dict.keys())
    code_index = {code: idx for idx, code in enumerate(code_table)}
    partition_count = 4
    partitions = [[] for _ in range(partition_count)]
    for decision in sorted(decisions, key=lambda d: d.tid):
        partitions[_partition_of(decision.tid, partition_count)].append(
            decision)

    reason_codes_not_in_dict = set()
    for decision in decisions:
        for code in decision.reason_codes:
            if code not in code_index:
                reason_codes_not_in_dict.add(code)
                sidecar["losses"].append({
                    "tid": decision.tid, "field": "reason_code",
                    "original": code, "stored": None,
                })
        sidecar["tickets"][decision.tid] = {
            "full_reason_codes": list(decision.reason_codes),
        }

    header = bytearray()
    header += MAGIC + struct.pack("<BBH", 3, 0, partition_count)
    header += struct.pack("<Hiq", config.decimals, config.quantum,
                          config.half_life_ms)
    header += struct.pack("<H", len(code_table))
    for code in code_table:
        header += struct.pack("<I", _encode_code(code))

    part_header_blob = bytearray()
    body_base = len(header)
    part_header_size = 14 * partition_count
    body_start = body_base + part_header_size
    cursor = body_start
    offset_tables = []
    encoded = []
    for rows in partitions:
        table = []
        part_encoded = []
        for decision in rows:
            table.append(cursor)
            record = _encode_record(decision, code_index, sidecar)
            part_encoded.append(record)
            cursor += len(record)
        offset_tables.append(table)
        encoded.append(part_encoded)
    body_blob = bytearray()
    for part_encoded in encoded:
        for record in part_encoded:
            body_blob += record
    table_base = cursor
    for part_id, table in enumerate(offset_tables):
        table_pos = table_base + sum(8 * len(t)
                                     for t in offset_tables[:part_id])
        part_header_blob += struct.pack("<HIQ", part_id, len(table),
                                        table_pos)
    for table in offset_tables:
        for off in table:
            body_blob += struct.pack("<Q", off)

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(part_header_blob)
        fh.write(body_blob)
    return sidecar


def _encode_code(code: str) -> int:
    kind = 1 if code.startswith("MX") else 2
    number = int("".join(ch for ch in code if ch.isdigit()) or "0")
    return (kind << 24) | number


def _decode_code(value: int) -> str:
    kind = value >> 24
    number = value & 0xFFFFFF
    return ("MX" if kind == 1 else "SP") + str(number)


def _encode_record(decision, code_index, sidecar):
    blob = bytearray()
    tid = decision.tid.encode("utf-8")
    blob += struct.pack("<H", len(tid)) + tid
    flags = 0
    if decision.boundary:
        flags |= FLAG_BOUNDARY
    if decision.escalate:
        flags |= FLAG_ESCALATE
    blob += struct.pack("<q", decision.base_ts)
    blob += struct.pack("<q", decision.decision_score)
    blob += struct.pack("<B", flags)
    blob += struct.pack("<H", decision.priority_level)

    bitmap = 0
    for code in decision.reason_codes:
        idx = code_index.get(code)
        if idx is None:
            continue
        if idx >= 64:
            sidecar["losses"].append({
                "tid": decision.tid, "field": "reason_bitmap",
                "original": code, "stored": None,
            })
            continue
        bitmap |= 1 << idx
    blob += struct.pack("<H", len(decision.reason_codes))
    blob += struct.pack("<Q", bitmap)

    blob += struct.pack("<H", len(decision.evidence_scores))
    for esc in decision.evidence_scores:
        etype = esc.etype.encode("utf-8")
        role = esc.role.encode("utf-8")
        blob += struct.pack("<qiqi", esc.seq, esc.ts, esc.confidence,
                            esc.quantized_score)
        blob += struct.pack("<i", esc.residual)
        blob += struct.pack("<H", len(etype)) + etype
        blob += struct.pack("<H", len(role)) + role
    return blob


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return values

    def bytes(self, size):
        out = self.data[self.pos:self.pos + size]
        self.pos += size
        return out


def read_bundle(path: str, sidecar: dict):
    with open(path, "rb") as fh:
        data = fh.read()
    reader = _Reader(data)
    magic = reader.bytes(4)
    if magic != MAGIC:
        raise ValueError("bad v3 magic")
    version, _flags, partition_count = reader.take("<BBH")
    decimals, quantum, half_life = reader.take("<Hiq")
    (reason_count,) = reader.take("<H")
    code_table = [_decode_code(reader.take("<I")[0])
                  for _ in range(reason_count)]

    decisions = []
    for _ in range(partition_count):
        part_id, record_count, table_pos = reader.take("<HIQ")
        next_pos = reader.pos
        reader.pos = table_pos
        offsets = [reader.take("<Q")[0] for _ in range(record_count)]
        reader.pos = next_pos
        for offset in offsets:
            saved = reader.pos
            reader.pos = offset
            decisions.append(_decode_record(reader, code_table, sidecar))
            reader.pos = saved
    decisions.sort(key=lambda d: d.tid)
    return decisions


def _decode_record(reader, code_table, sidecar):
    (tid_len,) = reader.take("<H")
    tid = reader.bytes(tid_len).decode("utf-8")
    base_ts, decision_score, flags, priority_level = reader.take("<qqBH")
    reason_count, bitmap = reader.take("<HQ")
    (evidence_count,) = reader.take("<H")
    evidence = []
    for _ in range(evidence_count):
        seq, ts, confidence, score = reader.take("<qiqi")
        (residual,) = reader.take("<i")
        (etype_len,) = reader.take("<H")
        etype = reader.bytes(etype_len).decode("utf-8")
        (role_len,) = reader.take("<H")
        role = reader.bytes(role_len).decode("utf-8")
        evidence.append(EvidenceScore(
            seq=seq, etype=etype, role=role, ts=ts, delta_ms=0, decay=0,
            weight=0, role_cred=0, confidence=confidence, factor_1=0,
            factor_2=0, factor_3=0, round_dir_1="", round_dir_2="",
            round_dir_3="", raw_score=score, quantized_score=score,
            residual=residual,
        ))
    bitmap_codes = [code_table[idx] for idx in range(len(code_table))
                    if idx < 64 and bitmap & (1 << idx)]
    full_codes = sidecar.get("tickets", {}).get(tid, {}).get(
        "full_reason_codes", bitmap_codes)
    return TicketDecision(
        tid=tid, base_ts=base_ts, evidence_scores=evidence,
        final_evidence_scores={e.seq: e.quantized_score for e in evidence},
        contributions=[], decision_score=decision_score,
        boundary=bool(flags & FLAG_BOUNDARY),
        priority_level=priority_level,
        priority_label="", escalate=bool(flags & FLAG_ESCALATE),
        reason_codes=list(full_codes), stage_keys={},
    )
