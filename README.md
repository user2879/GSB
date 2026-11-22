# TxDiag：事务隔离与锁等待诊断工作台

TxDiag 是一个仅依赖 Python 标准库的本地事务诊断工作台。它读取事务声明 JSON 与操作序列 NDJSON，在本地 SQLite 账本上以真实工作线程执行确定性锁协议，支持四种隔离级别、行级/区间加锁、稳定等待队列、真实超时、死锁牺牲、逐轮持久化、崩溃恢复、重放校验与 Tkinter 诊断界面。

## 环境

- Python 3.10 及以上（本次验证使用 Python 3.11）
- 标准库：`sqlite3`、`tkinter`、`threading`、`json`
- 无第三方依赖；不使用 Docker 或其他容器

Windows 命令行自检：

```bat
py -3.11 --version
py -3.11 -c "import sqlite3,tkinter,threading,json; print(sqlite3.sqlite_version, tkinter.TkVersion); r=tkinter.Tk(); r.update_idletasks(); r.destroy(); print('Tk OK')"
```

若出现 `No module named tkinter` 或缺少 `tcl/tk` 动态库，请安装 python.org 官方 Windows 构建并勾选 “tcl/tk and IDLE”。

## 目录

```text
txdiag/             引擎、规范校验、SQLite 账本、CLI、Tkinter 界面
samples/            确定性 JSON/NDJSON 样本
ledger/             默认账本目录
tmp/                临时验证区域
artifacts/          验收记录与逐字节一致的轨迹
tests/              unittest 测试套件
scripts/            样本生成与完整验收脚本
docs/               设计与实际验证说明
```

## 快速开始

```bat
py -3.11 -m txdiag validate --spec samples\all_isolation.json

py -3.11 -m txdiag run --spec samples\all_isolation.json ^
  --ledger ledger\txdiag.sqlite ^
  --snapshot ledger\session.snapshot.json ^
  --max-running 4 ^
  --quota read_committed=2 ^
  --wait-timeout-ms 2000 ^
  --wait-budget-ms 10000 ^
  --path auto ^
  --report artifacts\all_isolation.report.json

py -3.11 -m txdiag replay --spec samples\all_isolation.json ^
  --ledger ledger\txdiag.sqlite ^
  --snapshot ledger\session.snapshot.json

py -3.11 -m txdiag gui --spec samples\deadlock.json ^
  --ledger ledger\gui.sqlite ^
  --snapshot ledger\gui.snapshot.json
```

同时执行两条锁路径并比较规范化字节：

```bat
py -3.11 -m txdiag run --spec samples\large_range.json --path both ^
  --ledger tmp\both.sqlite --snapshot tmp\both.snapshot.json ^
  --trace-out tmp\both.trace.json --report tmp\both.report.json
```

输出为 `tmp\both.trace.row.json` 与 `tmp\both.trace.range.json`；两份文件应逐字节相同。

## 输入格式

声明 JSON：

```json
{
  "schema_version": 1,
  "row_keys": ["acct:A", "acct:B"],
  "dependency_edges": [
    {"from": "acct:A", "to": "acct:B", "kind": "reference"}
  ],
  "initial_versions": {
    "acct:A": {"value": 0, "version": 0, "status": "committed"},
    "acct:B": {"value": 0, "version": 0, "status": "committed"}
  },
  "transactions": [
    {"id": "T1", "start_seq": 1, "isolation": "read_committed"},
    {"id": "T2", "start_seq": 2, "isolation": "serializable"}
  ],
  "operations_file": "example.ndjson"
}
```

操作 NDJSON，每行一个对象：

```json
{"txid":"T1","seq":1,"kind":"read","row":"acct:A"}
{"txid":"T1","seq":2,"kind":"write","row":"acct:A","value":10}
{"txid":"T2","seq":1,"kind":"read","row":"acct:B"}
```

也可在声明事务的 `operations` 数组内嵌操作；两种方式不能混用。隔离级别接受 `read_uncommitted`、`read_committed`、`repeatable_read`、`serializable`，也接受“读未提交/读已提交/可重复读/串行化”。

严格校验失败统一返回 `E_SPEC`，并给出首个非法事务与字段；检查项包括事务 ID 唯一、起始序号非负且唯一、行键存在、操作序号从 1 连续、写操作值合法及依赖图端点合法。

## 调度与锁语义

- 等待图与死锁牺牲规则：等待队列按 `(start_seq, txid)` 排序；牺牲事务取环中 `start_seq` 最大者，仍相同则取字典序最大的 `txid`。
- 并发限制：全局运行事务上限与每个隔离级别的配额同时生效，超出配额的事务进入稳定队列。
- 加锁路径：小规模默认按行加锁；事务数 ≥ 8、行键数 ≥ 12、依赖边数 ≥ 10，或三者总和 ≥ 24 时默认区间合并加锁。阈值可通过 `--range-threshold` 覆盖。
- 隔离可见性：RU 可读最新未提交版本；RC 读最新已提交版本；RR 固定事务内首次读到的版本；Serializable 对访问行同时持有 S/X 锁并持有到提交。

阈值覆盖示例：`--range-threshold tx=10,rows=16,edges=12,total=30`。

## 账本、恢复与错误码

每个调度轮在同一 SQLite 事务中写入轮次、事件、内容摘要和观测计时；快照文件整体原子替换。声明或配置摘要不匹配时清除旧账本并整体替换，输出稳定差异列表，不做增量合并。

强制终止后，账本中未提交的轮次不会生成退出报告。恢复会找出执行中事务并标记为 `unknown`，要求重跑；已提交轮次可按账本路径、隔离级别与版本重新执行并逐项比较，首个差异以 `E_REPLAY` 定位到事务、行键和字段。

| 代码 | 含义 | CLI 退出码 |
|---|---|---|
| `E_SPEC` | 声明或配置非法 | 2 |
| `E_TIMEOUT` | 真实单调时钟等待超时 | 3 |
| `E_DIVERGE` | 两条锁路径的规范化轨迹不一致 | 5 |
| `E_REPLAY` | 账本重放不一致或未完成 | 6 |

`E_TIMEOUT` 命中后运行仍会完成账本写入，并以事务中止结果结束；其他正常完成或已消解死锁的运行退出码为 0。

## 验证

```bat
py -3.11 -m unittest discover -v
py -3.11 scripts\validate_acceptance.py
```

完整结果见 `artifacts\validation.json`、`artifacts\unittest.log` 和 `artifacts\acceptance\`。设计细节与计时口径见 `docs\DESIGN.md`，实际验证记录见 `docs\VALIDATION.md`。
