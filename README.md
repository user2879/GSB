# GSB 事件总线订阅消费链路

本地确定性的事件总线订阅/消费流水线：JSON 声明订阅者与发布通道、NDJSON
事件流，订阅者以独立子进程在 `127.0.0.1` 真实监听，总线按主题与订阅关系
切批、经真实 TCP 派发，处理器用 NumPy 做整数定点聚合并回传 x1/x2/x3 三种
历史回执格式；回执归一、租约续约、背压调度与多版本采纳裁决全部落到 SQLite
原子账本，支持强制终止后的恢复重放与首个差异定位。

## 环境准备

```bash
virtualenv .venv                 # 本机 ensurepip 缺失，使用 virtualenv
source .venv/bin/activate
pip install numpy==1.26.4
export PYTHONPATH=src
```

## 快速开始

```bash
python -m gsb.cli gen --out samples/demo --events 48
python -m gsb.cli run --spec samples/demo/spec.json \
    --events samples/demo/events.ndjson --run-dir artifacts/runs/demo-final \
    --demo-stale-renew
python -m gsb.cli verify --spec samples/demo/spec.json \
    --events samples/demo/events.ndjson --run-dir artifacts/runs/demo-final
python -m gsb.cli recover --spec samples/demo/spec.json \
    --events samples/demo/events.ndjson --run-dir artifacts/runs/demo-final
./scripts/demo.sh                # 含强制终止恢复的一键演示
```

## 测试

```bash
PYTHONPATH=src python -W error::ResourceWarning -m unittest discover -s tests
```

## 代码布局

- `src/gsb/spec.py` 声明与事件流强校验（`E_SPEC`）
- `src/gsb/receipts.py` x1 定宽 / x2 / x3 编解码与归一补齐
- `src/gsb/batching.py`、`src/gsb/aggregation.py` 确定性切批、批标识、NumPy 定点聚合
- `src/gsb/processor.py`、`src/gsb/client.py` 子进程处理器与真实回环 TCP
- `src/gsb/lease.py`、`src/gsb/scheduler.py` 租约/续约与三重约束线程池调度
- `src/gsb/adoption.py` 多版本采纳裁决与三段背压归因
- `src/gsb/ledger.py` SQLite 原子账本、快照整体替换、恢复重放
- `src/gsb/cli.py` `gen/run/recover/verify` 命令
- `tests/` 24 个单元与端到端测试
- `docs/DESIGN.md` 详细设计与验证结果

错误码：`E_SPEC`、`E_FORMAT`、`E_BACKFILL`、`E_PORT`、`E_TIMEOUT`、`E_LEASE`。
