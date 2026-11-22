"""工作台启动入口。

用法：
  python scripts/run_workbench.py                # 启动 Tkinter 界面 + 回环服务
  python scripts/run_workbench.py --serve        # 仅启动 UDP/TCP 回环服务
  python scripts/run_workbench.py --no-forward   # 关闭 TCP 上游转发
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dns_workbench.app import DnsWorkbench  # noqa: E402
from dns_workbench.gui import run_gui  # noqa: E402
from dns_workbench.server import DnsServerPair  # noqa: E402

SAMPLES = ROOT / "dns_workbench" / "samples"


def main() -> int:
    parser = argparse.ArgumentParser(description="DNS 解析与缓存诊断工作台")
    parser.add_argument("--serve", action="store_true", help="仅运行回环 UDP/TCP 服务")
    parser.add_argument("--udp-port", type=int, default=5353)
    parser.add_argument("--tcp-port", type=int, default=5353)
    parser.add_argument("--upstream-port", type=int, default=5354)
    parser.add_argument("--no-forward", action="store_true")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "workbench"))
    args = parser.parse_args()

    workbench = DnsWorkbench(
        sample_dir=SAMPLES,
        data_dir=Path(args.data_dir),
        upstream_port=args.upstream_port,
        forward=not args.no_forward,
    )
    workbench.bootstrap(
        zone_path=SAMPLES / "zone.json",
        cache_paths=[SAMPLES / "cache_v1.json", SAMPLES / "cache_v2.json", SAMPLES / "cache_v3.json"],
    )
    pair = DnsServerPair(workbench.engine, udp_port=args.udp_port, tcp_port=args.tcp_port)
    pair.start()
    print(f"回环 DNS 服务已启动: UDP/TCP 127.0.0.1:{args.tcp_port}（上游 127.0.0.1:{args.upstream_port}）")
    if args.serve:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pair.stop()
            workbench.close()
        return 0
    started = run_gui(workbench)
    pair.stop()
    workbench.close()
    if not started:
        print("GUI 不可用，已退出。可使用 --serve 运行无界面服务，或运行 scripts/demo.py 查看完整演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
