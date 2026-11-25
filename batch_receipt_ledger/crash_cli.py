"""崩溃模拟子进程入口：从 stdin 读取声明，部分批提交后 SIGKILL。"""

import json
import os
import sys

from .crash_sim import run_until_kill
from .spec import parse_spec


def main():
    spec_dict = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    spec = parse_spec(spec_dict)
    db_path = os.environ.get("BRL_DEMO_DB", "artifacts/run/crash.sqlite3")
    run_until_kill(spec, spec_dict, db_path, commit_before_kill=2)


if __name__ == "__main__":
    main()
