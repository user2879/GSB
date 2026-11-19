# 多人协作对局状态同步管线（syncpipe）

确定性的事件流合成、冲突裁决、快照版本链、崩溃恢复与独立验证工具。固定种子
生成样本，两次相同输入的所有持久化产物逐字节一致；仅依赖 Python 标准库与 NumPy，
无 Docker、无远程服务、无账号。

## 快速开始

```bash
# 1. 虚拟环境与依赖
python -m venv .venv
# Linux/macOS:
.venv/bin/python -m pip install -r requirements.txt
# Windows:
# .venv\Scripts\python -m pip install -r requirements.txt

# 2. 一条命令跑完整验收（生成样本→合成→压缩→强制终止→截断恢复→验证器比对）
.venv/bin/python run_demo.py
#   等价于：.venv/bin/python -m syncpipe.cli demo --workspace workspace --seed 20261119

# 3. 独立验证入口
.venv/bin/python -m syncpipe.cli validate --root workspace/runs/normal_game --seed 20261119

# 4. 单元测试（23 个，覆盖 7 个错误码异常链路）
.venv/bin/python run_tests.py
```

## 目录

- `syncpipe/` — 管线实现（解析、状态原语、引擎、持久化、恢复、验证器、样本生成器）
- `tests/` — 单元与集成测试
- `docs/DESIGN.md` — 架构、错误码、确定性与恢复设计、实际验证结果
- `workspace/` — demo 运行产物
  - `samples/` 固定种子事件流（JSONL）
  - `runs/<场景>/` 快照链、CRC 操作日志、manifest、隔离区
  - `reports/version_hashes.txt` 逐版本状态哈希清单
  - `reports/recovery_report.txt` 强制终止恢复报告
  - `reports/demo_report.json`、`hash_manifest.json`、`validation_summaries.json`

## 四个样本场景

- `normal_game`：spawn/move/update/destroy 全操作常规局
- `conflict_game`：同 tick 同实体并发写字段级 LWW 裁决
- `catchup_game`：滞后 base 事件挂起（E_BASE_VERSION）后追赶应用
- `kill_game`：子进程强制终止 + 日志截断（E_LOG_TRUNCATED）+ 恢复一致性

## 错误码

`E_BASE_VERSION`、`E_PEER_GAP`、`E_SNAPSHOT_HASH`、`E_LOG_TRUNCATED`、
`E_VERSION_DIVERGE`、`E_CONFLICT_UNRESOLVED`、`E_CHAIN_BROKEN`，每条均带出错版本
与触发事件定位，详见 `docs/DESIGN.md` 第 7 节。
