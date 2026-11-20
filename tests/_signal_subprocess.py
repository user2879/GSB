"""子进程辅助脚本：安装信号处理器并等待 SIGTERM，验证优雅退出契约。"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrep.config import Config
from vrep.engine import Engine
from vrep.injection import InjectionManager


def main() -> int:
    samples, workspace = sys.argv[1], sys.argv[2]
    config = Config.load(os.path.join(samples, "config.json"))
    stop_event = threading.Event()

    def handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    engine = Engine(workspace, InjectionManager(None), stop_event=stop_event)
    engine.seed_library(samples)

    # 通知父进程：信号处理器已安装，可以投递信号
    print("ready", flush=True)
    for _ in range(100):
        if stop_event.is_set():
            break
        time.sleep(0.05)

    # 信号到达后仍完整提交当前批次，再以 0 退出
    engine.run_batch(config, batch_no=1)
    print("graceful-shutdown-ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
