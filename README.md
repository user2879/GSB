# 缓存失效传播中间件（本地、标准库、可复现）

面向 Windows 本地环境的缓存失效传播组件，验证“条目注释版本如何决定失效
传播的可复现性”。仅依赖 Python 3.11+ 标准库（`sqlite3/json/threading/
subprocess/tkinter`），不使用容器、第三方包或在线服务。

## 快速开始

```powershell
python -m cache_invalidation.cli env
python run_tests.py
python -m cache_invalidation.cli demo
python -m cache_invalidation.cli loopback
python -m cache_invalidation.gui
```

## 能力概览

- 命名空间、条目与注释依赖图的本地 JSON 加载和强校验（`E_GRAPH`/`E_INHERIT`）。
- v1/v2/v3 三版本快照编解码、归一与字段级定位（`E_FORMAT`/`E_SNAPSHOT`）。
- 显式失效、注入逻辑时钟过期回收、命名空间刷新三类传播，稳定拓扑顺序，
  多次运行逐字节一致。
- 条目级线程互斥：每轮每条目至多失效一次、注释版本只递增一次；时钟倒退
  报 `E_CLOCK` 且不改变状态。
- SQLite 失效账本；恢复时识别格式/状态版本差异并整表替换，稳定输出版本差异。
- 恢复后逐轮重放，首个不一致定位到触发条目、受影响条目与字段（`E_REPLAY`），
  v1 升级口径缺失如实报告。
- 三类只改输入的异常注入与 JSON Lines 结构化日志。
- Tkinter 诊断界面展示继承链、注释边、每轮触发/受影响集合、注释版本与业务码。

## 目录

- `cache_invalidation/`：实现代码，模块职责见 `docs/TECHNICAL.md`。
- `samples/`：确定性命名空间与条目声明。
- `tests/`：46 个标准库 `unittest` 用例。
- `artifacts/demo/`：演示产物（SQLite、v1/v3 快照、账本总结）。
- `docs/TECHNICAL.md`：设计、业务码对照与实际验证结果。

## 异常注入

```powershell
python -m cache_invalidation.cli --log-file artifacts/faults.jsonl inject `
  --fault annotation_cycle --namespaces samples/namespaces.json --entries samples/entries.json
python -m cache_invalidation.cli inject --fault missing_inherit `
  --namespaces samples/namespaces.json --entries samples/entries.json
python -m cache_invalidation.cli inject --fault truncated_snapshot `
  --snapshot artifacts/demo/snapshot_v3.json
```

三类注入分别对应 `E_GRAPH`、`E_INHERIT`、`E_SNAPSHOT`，只修改临时输入
副本或快照参数，不修改实现代码。
