"""强制终止恢复演练：在卷批次提交之间结束进程，再以整体替换方式恢复并重放校验。
用法：
  python -m scripts.kill_recovery_test --kill-after 5   # 第 5 批后强杀（退出码 42）
  python -m scripts.kill_recovery_test                  # 恢复并续跑到重放通过
"""

import argparse
import os
import shutil
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quota_workbench.arbitrate import RULE_VERSION
from quota_workbench.errors import WorkbenchError
from quota_workbench.ledger import Ledger
from quota_workbench.pipeline import (all_rows, build_view, load_declaration,
                                      load_events, load_rules, run_to_ledger,
                                      scheme_meta)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.environ.get("QW_SAMPLES_DIR", os.path.join(ROOT, "samples"))
BACKUP_DIR = os.path.join(SAMPLES, "backup")
CHECKPOINT = os.path.join(SAMPLES, "checkpoint.txt")


def _writable(path):
    if os.path.exists(path):
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def make_readonly_backup(ledger_path):
    """把当前账本复制到只读备份目录，并写检查点文件。"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(ledger_path):
        dst = os.path.join(BACKUP_DIR, "ledger.sqlite.bak")
        _writable(dst)
        shutil.copyfile(ledger_path, dst)
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _writable(CHECKPOINT)
    with open(CHECKPOINT, "w", encoding="utf-8") as fh:
        fh.write(f"ledger={ledger_path}\nbackup={BACKUP_DIR}\n")
    os.chmod(CHECKPOINT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-after", type=int, default=None)
    args = parser.parse_args()
    declaration_path = os.path.join(SAMPLES, "declaration.json")
    events_path = os.path.join(SAMPLES, "events.ndjson")
    rules_path = os.path.join(SAMPLES, "rules.json")
    ledger_path = os.path.join(SAMPLES, "ledger.sqlite")

    make_readonly_backup(ledger_path)
    ledger = Ledger(ledger_path)
    try:
        view, meta = run_to_ledger(declaration_path, events_path, rules_path,
                                   ledger, kill_after=args.kill_after)
        print(f"OK: 全部批次提交完成，重放校验通过，"
              f"内容摘要 {ledger.content_digest()[:16]}…")
    except WorkbenchError as exc:
        if exc.code != "E_SNAPSHOT":
            raise
        print(f"{exc.code}: 快照不一致，执行整体替换恢复：{exc.details}")
        declaration = load_declaration(declaration_path)
        events = load_events(events_path)
        rules = load_rules(rules_path)
        rule_version = rules.get("rule_version", RULE_VERSION)
        view = build_view(declaration, events, rules["target_profile"],
                          rule_version)
        meta = scheme_meta(declaration, rule_version, 1)
        ledger.full_replace(all_rows(view), meta)
        ledger.replay_verify(all_rows(view))
        print("OK: 整体替换恢复完成，恢复后重放校验通过，"
              f"内容摘要 {ledger.content_digest()[:16]}…")
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
