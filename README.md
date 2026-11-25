# RLCB — 本地多服务限流 / 熔断降级链路

离线、确定性的本地限流熔断系统。每个服务在 `127.0.0.1` 回环端口以**真实子进程**托管
NumPy 整数定点执行器，主进程经真实 TCP 套接字逐行通信；策略结论按整数定点口径计算，
在任意批次切分、字典遍历顺序与线程交错下逐字节一致。不使用 Docker，不访问外部网络。

## 环境准备（WSL Ubuntu / Python 3.11）

```bash
python3 -m venv .venv            # 若 ensurepip 缺失，见文末“环境备注”
source .venv/bin/activate
pip install numpy==1.26.4
python -c "import sqlite3, tkinter; print('stdlib ok')"
```

本仓库的 `.venv` 用 `python3 -m venv --without-pip --system-site-packages .venv` 创建，
复用基础解释器自带的 numpy 1.26.4；Tk 运行库以用户态解压到 `/tmp/tkdeps2`，
`run_acceptance.sh` 会在该目录存在时自动注入 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY`。

## 一键验收

```bash
source .venv/bin/activate
./run_acceptance.sh
```

包含：环境检查、54 项 unittest（含真实回环子进程、SIGKILL 恢复）、确定性样本生成、
真实子进程全量运行、从零重算逐请求一致性校验、强制终止恢复演示。

## 常用命令

```bash
# 生成确定性样本（3 个服务、37 个请求，含触发熔断与恢复的失败模式）
python -m rlcb.cli sample --out samples/acceptance

# 强校验（非法输入报 E_SPEC，含首个字段与 JSON 行号）
python -m rlcb.cli validate \
  --services samples/acceptance/services.json \
  --requests samples/acceptance/requests.json

# 全量运行（真实 127.0.0.1 子进程 + 线程池 + SQLite 原子提交）
python -m rlcb.cli run \
  --services samples/acceptance/services.json \
  --requests samples/acceptance/requests.json \
  --ledger artifacts/acceptance/ledger.db \
  --out artifacts/acceptance --mode full

# 参数演进后的增量重算（仅失效子集重算，未受影响结论逐字节保持）
python -m rlcb.cli run --services <svc.json> --requests <req.json> \
  --edits edits.json --ledger artifacts/inc/ledger.db \
  --out artifacts/inc --mode incremental

# 与从零全量重算逐请求一致性校验
python -m rlcb.cli verify --services <svc.json> --requests <req.json> \
  --ledger artifacts/acceptance/ledger.db

# 强制终止后的恢复重放（未提交批次先回收配额再补齐，逐项比对）
python -m rlcb.cli recover --services <svc.json> --requests <req.json> \
  --ledger artifacts/crash/ledger.db --in-process

# 快照版本/摘要不一致时整体替换
python -m rlcb.cli snapshot --ledger <db> --fingerprint <fp.json>

# Tkinter 仪表盘（无显示环境自动降级为文本视图）
python -m rlcb.cli ui --report artifacts/acceptance/run_report.json
```

## 目录职责

- `rlcb/spec.py`、`rlcb/linejson.py`：本地 JSON 强校验与首个非法字段行号定位（E_SPEC）。
- `rlcb/ratelimit.py`：确定性整数准入，rps=0 全拒绝。
- `rlcb/breaker.py`：纯滑动窗口熔断/降级状态机，万分比整数定点。
- `rlcb/lease.py`：租约签发/续约/真实过期、乐观版本与唯一归属裁决重放。
- `rlcb/clock.py`：真实单调毫秒生产钟 + 可推进假钟。
- `rlcb/fixedpoint.py`：整数定点与固定种子 NumPy 确定性执行器。
- `rlcb/worker.py`：回环子进程、逐行 JSON 协议与权威租约版本表。
- `rlcb/transport.py`：子进程生命周期、真实套接字超时与 E_PORT/E_TIMEOUT/E_REPLY。
- `rlcb/scheduler.py`：全局租约/全局批次/每服务并发三重约束的真实线程池调度。
- `rlcb/planner.py`、`rlcb/invalidate.py`：全量/增量选择规则与失效传播。
- `rlcb/ledger.py`：SQLite 单事务原子提交、快照整体替换与崩溃恢复重放。
- `rlcb/engine.py`：策略层与运行时层编排。
- `rlcb/cli.py`、`rlcb/ui.py`：命令行与 Tkinter 界面（无头降级）。
- `tests/`：54 项 unittest；`scripts/demo_force_kill.py`：真实 SIGKILL 恢复演示。
- `docs/DESIGN.md`：架构与裁决细节；`docs/acceptance-output.txt`：最近一次验收实跑输出。

## 确定性样本

固定种子 `20261126`。svcA 窗口 4/阈值 4000bp、svcB 窗口 5/阈值 5000bp，失败模式手工设计为
先触发熔断、再经显式恢复探测窗闭合；svcC 的 rps=0，所有请求在进入窗口前被限流拒绝。
最近一次全量运行结论分布：`executed=13, degraded=18, rate_rejected=6`。

## 环境备注

- 若 `python3 -m venv` 报 ensurepip 不可用且无 sudo，可改用
  `python3 -m venv --without-pip --system-site-packages .venv`，复用基础解释器的 numpy。
- Tk 系统共享库缺失时，可在用户态 `apt-get download tk8.6 tcl8.6 libtcl8.6` 后
  `dpkg-deb -x` 到本地目录，再经上述三个环境变量接入，无需 root。
