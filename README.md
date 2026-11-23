# kvdiag — 键值索引分片与过期位点诊断组件

面向 Windows 本地环境（Python 3.10+，仅标准库）的键值存储索引分片与过期
位点诊断组件：本地 SQLite 维护键空间快照链与索引分片元数据，本地 JSON
声明分片方式、确定性样本、索引规则与过期租约，支持快照回放、租约续约、
收敛判定、强制终止恢复与异常注入。

## 环境准备（Windows 本地）

1. 安装 Python 3.10 及以上（python.org 安装包自带 tkinter）。
2. 确认标准库可用：`python -c "import sqlite3, tkinter, threading, subprocess"`。
3. 无需网络与外部账号；不使用 Docker。

## 快速开始

```bat
REM 生成确定性示例树（声明 + 样本 + 续约计划 + 注入清单）
python -m kvdiag gen-samples --out examples --seed 42

REM 运行流水线：加载校验 -> 索引构建 -> 租约续约 -> 过期回收 -> 快照链 -> 收敛判定 -> 事务提交
python -m kvdiag run --spec examples\keyspace.json --db artifacts\kvdiag.sqlite --renewals examples\renewals.json --session-out artifacts\session_snapshot.json

REM 快照回放校验（与不经过快照的直接重建逐字节一致）
python -m kvdiag replay --spec examples\keyspace.json --db artifacts\kvdiag.sqlite

REM 恢复：版本/摘要识别整体替换 + 未知分片重跑
python -m kvdiag recover --spec examples\keyspace.json --db artifacts\kvdiag.sqlite --session artifacts\session_snapshot.json --rerun-unknown

REM Tkinter 诊断界面
python -m kvdiag ui --spec examples\keyspace.json --db artifacts\kvdiag.sqlite
```

## 异常注入（仅修改输入文件，不改源码）

```bat
python -m kvdiag inject sample-corrupt --spec examples\keyspace.json --shard shard-1   REM -> E_SPEC
python -m kvdiag inject lease-version-regression --renewals examples\renewals.json     REM -> E_LEASE
python -m kvdiag inject ledger-digest-mismatch --session artifacts\session_snapshot.json REM -> E_SNAPSHOT
```

## 强制终止恢复演示

```bat
python -m kvdiag run --spec examples\keyspace.json --db artifacts\crash.sqlite --renewals examples\renewals.json --session-out artifacts\crash_session.json --crash-before-batch 2
REM 退出码 3：批次 2 未提交。重启恢复：
python -m kvdiag recover --spec examples\keyspace.json --db artifacts\crash.sqlite --session artifacts\crash_session.json --renewals examples\renewals.json --rerun-unknown
```

## 测试

```bat
python -m pytest tests -q
```

## 目录结构

- `kvdiag/` 可集成 Python 包（spec/index/snapshots/leases/converge/scheduler/ledger/recovery/inject/ui/cli）
- `scripts/gen_samples.py` 确定性样本生成脚本
- `examples/` 本地 JSON 示例声明、样本、续约计划与异常注入清单
- `tests/` 35 项单元与端到端测试
- `docs/TECHNICAL.md` 技术说明与验证记录；`docs/TASK.md` 原始任务书
