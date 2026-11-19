# 嵌入式键值存储索引与过期组件 — 技术说明

面向本地 Windows（亦可在 macOS/Linux/WSL 运行）的纯 Python 嵌入式组件，以本地
SQLite 文件持久化键值记录、前缀二级索引、派生计数视图与会话元数据；全程离线，
不访问网络或外部服务。

## 1. 目录结构

```
kvstore/
  errors.py      业务错误码与异常（E_RECORD/E_TRACE/E_EXPIRED/E_INDEX/
                 E_VERSION/E_SNAPSHOT/E_VISIBLE/E_REPLAY）
  structlog.py   结构化 JSONL 日志
  codec.py       键/前缀规则、记录与索引条目编解码、合法性校验、确定性 JSON/摘要
  storage.py     SQLite 表结构、连接与锁、重建暂存表与原子换表
  engine.py      过期判定、PUT/DEL/PURGE、增量索引与派生计数、一致性校验
  rebuild.py     工作线程重建、delta 同步、暂存核对、原子切换
  trace.py       JSONL 轨迹加载/校验、确定性操作摘要
  replay.py      权威重放、E_VISIBLE 校验、快照恢复后 E_REPLAY 比对
  snapshot.py    快照摘要、版本识别、整体替换恢复、版本回退检测、只读留档
  inject.py      三类异常注入（只生成被篡改输入/参数，不改源码）
  numpyutil.py   NumPy 向量化计数/可见性矩阵比对
  session.py     会话运行器：推进轨迹、驱动重建、记录结论、产出快照
  cli.py         命令行入口
  gui.py         可选 Tkinter 交互界面（无显示环境用 CLI 即可）
data/
  generate_sample.py  确定性样本生成器（无随机源）
  initial_records.json 初始键值记录
  sample_trace.jsonl   五类操作的确定性轨迹（15 个操作）
tests/test_kvstore.py  22 项 unittest 用例
run_demo.py            端到端可复现实验，产出 results/
run_demo.bat/run_tests.bat  Windows 一键脚本
```

## 2. 数据模型（SQLite）

- `kv_records(key PRIMARY KEY, value, expire_ms, write_seq, format_version)`
- `index_entries(prefix, ref_key, index_value, entry_version, PRIMARY KEY(prefix, ref_key))`
- `prefix_counts(prefix PRIMARY KEY, cnt)`：由索引派生的结构计数
- `meta(key, value)`：键空间版本 `keyspace_version`、轨迹游标 `trace_cursor` 等

键规则：段为 `[A-Za-z0-9_-]+`，以 `/` 分隔；前缀必须段对齐，空前缀表示键空间根。
一个键 `a/b/c` 维护 `a`、`a/b`、`a/b/c` 三条前缀路径上的索引条目，因此任意段对齐
前缀都能直接走索引得到键集合与计数。`expire_ms == 0` 约定永不过期。

## 3. 八项业务要求到实现的映射

1. 持久化编解码与合法性：`codec.validate_record/validate_index_entry`，同一键
   `write_seq` 严格单调（`check_write_seq_monotonic`）；初始加载先全量校验再单事务
   写入，首错整体拒绝，E_RECORD 的 details 给出首个出错键与字段。
2. 确定性轨迹：`trace.load_trace` 校验 `op_seq` 连续唯一、`clock` 单调不减、PUT 必带
   键值与非负 `expire_ms`，违规 E_TRACE 报告首个出错序号；`operations_digest` 对规范
   化 JSON 取 SHA-256，同轨迹多次加载逐字节一致。
3. 快照时钟过期语义：`Engine.visibility_basis/get` 仅使用会话/轨迹注入的整数逻辑时钟，
   判定依据逐步输出且 `system_clock_used=false`；`clock >= expire_ms` 即 E_EXPIRED 且
   不计入前缀计数；`purge` 物理删除键与索引条目并递增键空间版本。
4. 增量索引与派生计数：`_apply_put/delete/purge` 在单事务内同步更新 kv、索引条目
   （覆盖语义、条目版本 +1、不重复累积）与 `prefix_counts`；`verify_index_consistency`
   按键集合重算逐项核对，首错 E_INDEX 报告偏差前缀；另由 NumPy 向量比对复核。
5. 并发重建一致视图：`RebuildWorker` 在独立线程把键快照重建到 `idx_stage/cnt_stage`
   暂存表，前台始终读正式表，杜绝半写；期间变更经 `delta_listener` 同步暂存，换表前
   drain + 不变式核对，再在单事务内 `ALTER ... RENAME` 原子切换；输出开始/完成的逻辑
   时钟与墙上时间。
6. 快速查询与权威重放分离：`query_prefix` 走增量索引/计数缓存即时返回；`replay.
   verify_visible` 在独立 DB 从初始状态逐操作重放，分歧 E_VISIBLE 报告前缀、快速结论、
   权威结论与分歧操作序号。
7. 快照版本识别与整体替换：`snapshot.save_snapshot/open_snapshot` 持久化键空间版本、
   游标、索引与条目版本、计数、重建进度、逐操作结论及内容摘要；摘要或版本不一致即
   `replace_all` 整体替换（绝不增量合并），旧会话只读留档并给按版本的差异摘要。
8. 恢复重放与异常注入：`verify_replay_after_restore` 重放到快照游标并逐项比对键集合、
   索引/版本、计数与 GET 过期结论，首错 E_REPLAY 定位操作序号/键/前缀；`inject.py`
   仅生成被篡改输入（轨迹跳号/坏 JSON→E_TRACE、索引条目负版本→E_VERSION、快照摘要
   篡改→E_SNAPSHOT），全部写入结构化日志，源码零改动。

## 4. 并发正确性论证

进程内以 `Storage.lock`（可重入锁）串行化所有 SQLite 访问，配合 WAL 与单连接：

- 读永远只访问正式表，重建的所有中间写入都落在 `*_stage` 暂存表，故不可能观察到
  半写条目或中间态计数。
- 重建期间的前台变更在持锁回调中同步到暂存表；换表前再 drain 一次 delta 并断言
  “暂存索引对 == 当前 kv 派生对、暂存计数 == 暂存索引重算”，通过后单事务换表。
- 因此任一时序下，查询结果都等于对该时刻已提交状态重放的结果（实验场景 3 与
  `test_concurrent_rebuild_never_shows_intermediate_state` 实证）。

## 5. 复现步骤（Windows）

```powershell
python --version          # 需 3.10+
py -m pip install numpy   # sqlite3、tkinter 随官方安装包提供
python data\generate_sample.py
python run_demo.py        # 端到端实验，产出 results\
python -m unittest discover -s tests -v
```

或直接双击/执行 `run_demo.bat`。GUI（可选）：

```powershell
python -m kvstore.gui --db results\main.db --trace data\sample_trace.jsonl --initial data\initial_records.json
```

常用 CLI：

```powershell
python -m kvstore.cli run --db results\m.db --trace data\sample_trace.jsonl --initial data\initial_records.json --snapshot results\s.json
python -m kvstore.cli query --db results\m.db --prefix user --clock 1300
python -m kvstore.cli get   --db results\m.db --key user/1/name --clock 1300
python -m kvstore.cli verify --db results\m.db --trace data\sample_trace.jsonl --initial data\initial_records.json --clock 1300
python -m kvstore.cli restore --db results\r.db --snapshot results\s.json --trace data\sample_trace.jsonl --initial data\initial_records.json
python -m kvstore.cli inject --kind trace_corrupt --trace data\sample_trace.jsonl
```

## 6. 实验产物

- `results/run_summary.json`：含墙上时间的完整结果。
- `results/baseline_summary.json`：剔除墙上时间的确定性基准，重复运行逐字节一致。
- `results/events.jsonl`：结构化日志（可见性判定、PURGE 依据、重建起止、注入业务码）。
- `results/snapshot.json`、`results/readonly/`：快照与旧会话只读留档。
- `results/injected/`：被注入异常的输入文件。
