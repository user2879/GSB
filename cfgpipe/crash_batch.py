"""强制终止模拟：开启一个批次（写入 inflight 标记）后立即 os._exit。

用于验证账本恢复：重启后该未提交批次中的节点须被标记为 unknown 并要求
重跑。本进程被强制终止时不生成、也不应被声称能生成任何退出报告。
"""

import argparse
import os
import sys

from .pipeline import Pipeline
from .scheduler import mono_ms


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cfgpipe.crash_batch")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--batch-id", default="batch_crash")
    parser.add_argument("--nodes", default="")
    args = parser.parse_args(argv)
    pipe = Pipeline(args.workdir)
    pipe.load()
    pipe.open_ledger()
    nodes = args.nodes.split(",") if args.nodes else [
        nd.node_id for nd in pipe.decl.nodes]
    pipe.ledger.begin_batch(args.batch_id, sorted(nodes), mono_ms())
    pipe.ledger.close()
    os._exit(1)  # 强制终止：不提交批次、不写退出报告
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
