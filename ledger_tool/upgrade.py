"""自动升级与中断续做.

检出低于当前格式(r3)的记录后逐条原子重写; 每完成一条即更新检查点文件;
旧记录以只读文件保留在备份目录. 重启只能依据检查点继续:
已升级记录不重复处理也不与备份重复计数. 检查点损坏时以 E_UPGRADE 报明,
从可判定边界重做并列出将重做的记录标识.
"""

import json
import os
import stat

from .canon import canonical_json
from .errors import E_UPGRADE, LedgerError
from .normalize import normalize_record
from . import records as records_mod


def read_checkpoint(path):
    """返回 {"processed": [...]} 或 None(不存在); 损坏时抛 E_UPGRADE."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        processed = data["processed"]
        if not isinstance(processed, list) or not all(isinstance(x, str) for x in processed):
            raise ValueError("processed must be a list of strings")
        return {"processed": list(processed)}
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        raise LedgerError(E_UPGRADE, f"checkpoint file corrupt: {exc}",
                          stage="upgrade", extra={"path": str(path)})


def write_checkpoint(path, processed):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"processed": list(processed), "cursor": len(processed)},
                  fh, sort_keys=True)
    os.replace(tmp, path)


def reconcile(ledger, checkpoint):
    """返回 (valid_prefix, redo_ids).

    可判定边界: 检查点已处理序列中, 在账本里确实已是 r3 的最长前缀;
    边界之后声明过但未完成的记录与账本中全部非 r3 记录构成重做集合.
    """
    processed = checkpoint["processed"] if checkpoint else []
    valid = []
    for rid in processed:
        if ledger.fmt_of(rid) == records_mod.CURRENT_FORMAT:
            valid.append(rid)
        else:
            break
    done = set(valid)
    redo = [row["request_id"] for row in ledger.read_all()
            if row["fmt"] != records_mod.CURRENT_FORMAT and row["request_id"] not in done]
    return valid, redo


def upgrade(ledger, pipeline, backup_dir, checkpoint_path, logger, max_upgrades=None):
    """返回 (trail, events). trail 为本次完成的升级轨迹."""
    events = []
    try:
        checkpoint = read_checkpoint(checkpoint_path)
    except LedgerError as exc:
        logger.event(exc.code, stage="upgrade", message=exc.message,
                     field=exc.field, record_id=exc.record_id, **exc.extra)
        events.append({"code": exc.code, "message": exc.message})
        checkpoint = None
    valid, redo = reconcile(ledger, checkpoint)
    declared = checkpoint["processed"] if checkpoint else []
    if len(valid) != len(declared):
        msg = (f"checkpoint inconsistent with ledger: boundary at {len(valid)}, "
               f"redo {len(redo)} record(s): {redo}")
        logger.event(E_UPGRADE, stage="upgrade", message=msg, redo=redo)
        events.append({"code": E_UPGRADE, "message": msg, "redo": list(redo)})
    processed = list(valid)
    write_checkpoint(checkpoint_path, processed)

    os.makedirs(backup_dir, exist_ok=True)
    trail = []
    done = set(processed)
    upgraded = 0
    for row in ledger.read_all():
        if row["fmt"] == records_mod.CURRENT_FORMAT or row["request_id"] in done:
            continue
        if max_upgrades is not None and upgraded >= max_upgrades:
            events.append({"code": "INTERRUPTED", "remaining": len(redo) - upgraded})
            break
        rid = row["request_id"]
        record = records_mod.decode(json.loads(row["payload"]), pipeline.supported_slots)
        view, _meta = normalize_record(record, pipeline)
        payload = records_mod.encode(records_mod.CURRENT_FORMAT, view)
        backup_path = os.path.join(backup_dir, f"{rid}.json")
        if os.path.exists(backup_path):
            raise LedgerError(E_UPGRADE,
                              f"backup already exists for {rid}; refusing double count",
                              record_id=rid, stage="upgrade")
        with open(backup_path, "w", encoding="utf-8") as fh:
            fh.write(canonical_json({"fmt": row["fmt"],
                                     "payload": json.loads(row["payload"])}))
        os.chmod(backup_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ledger.rewrite(rid, records_mod.CURRENT_FORMAT, payload)
        processed.append(rid)
        write_checkpoint(checkpoint_path, processed)
        trail.append({"request_id": rid, "from": row["fmt"],
                      "to": records_mod.CURRENT_FORMAT, "cursor": len(processed)})
        upgraded += 1
    return trail, events
