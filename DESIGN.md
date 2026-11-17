# 列式存储压缩与扫描系统 — 技术说明

面向 Windows 本地环境的纯 Python（3.11+）实现，依赖 NumPy；Tkinter 随官方
Python 发行版提供。无网络、无容器、无外部账号，所有数据落在项目目录内。

## 1. 目录与模块

```
colstore/
  errors.py          业务码与异常体系（E_SEGMENT/E_PAYLOAD/E_ENCODING/E_QUERY/
                     E_BUDGET/E_DIVERGE/E_SNAPSHOT）
  model.py           领域数据结构：列/行组/段定义、块坐标、预算参数、任务、统计、结果
  encoding.py        RAW / DICTIONARY / RLE 三种整数编码、确定性选择与内存估算
  segment.py         段文件二进制格式、写入器、全量校验读取器、异常注入
  query.py           查询解析、规范化（空白/括号不敏感）、列裁剪、谓词下推计划
  scheduler.py       三重约束块级并行调度器（线程池 + 条件变量）
  engine.py          扫描引擎：版本管理、陈旧防护、整体替换、结果装配、一致性校验
  snapshot.py        SQLite 快照：版本识别、整体替换恢复、只读旧快照与差异项
  structured_log.py  结构化 JSONL 日志（带单调序号 seq）
  prepare_samples.py 确定性样本段定义、查询、异常参数、旧版本快照生成
  cli.py             命令行入口（scan/check/estimate/inject/snapshot/prepare）
  demo.py            三场景演示，输出 JSONL 日志与验证总结（JSON + Markdown）
  ui.py              Tkinter 桌面应用（惰性导入 tkinter，无头环境不影响核心）
  tests/test_core.py 18 项 unittest
  sample_data/       样本定义、查询、fault_params.json、旧快照 sessions
  data/              运行期段文件、SQLite、日志、演示输出
run_all.py           一键验收入口；run_gui.bat / run_demo.bat 为 Windows 启动脚本
```

## 2. 段文件二进制格式（`.colseg`）

```
magic "CSEG0001" (8B) | u32 header_len | header_len 字节 JSON 目录(TOC) | 载荷区
```

TOC 中每个块记录：列名、行组号、编码方式、压缩标记、块头行数、列宽、
**压缩前/压缩后字节数**、载荷相对偏移、CRC32、min/max。压缩仅作用于载荷
（zlib level 9），块头保持明文 JSON 便于检查。

读取是**全量、整体**校验：magic/头长度/JSON → 块集合与定义交叉校验
（编码受支持、行数为正、列宽一致、字节数为正、偏移在文件内）→ 逐块提取、
解压、CRC、按声明编码完整解码并核对行数。任一损坏即抛业务码并携带
`column / row_group / payload_offset`，不返回任何半成品段。

## 3. 编码与解码内存联动

- 编码载荷布局固定且自带计数与宽度前缀，解码时校验“声明长度 == 实际载荷
  长度”、字典索引不越界、RLE 游程长度为正且游程总和等于头行数。
- 估算输出全部为确定性整数：给出基数、游程数、索引宽度、各中间字节项与
  公式；压缩比保留 4 位小数，量化误差上界固定 `0.5e-4`，同输入同结果。
- 自动选择规则：取编码后载荷字节最小者；并列时按固定全序
  `RLE > DICTIONARY > RAW` 决胜，并在 `reason/tie_broken/tied_candidates`
  中显式标注依据。调度使用估算的 `decoded_peak_bytes` 预留解码内存，因此
  编码选择直接决定可调度性（压缩选择与并行调度联动）。

## 4. 查询计划与谓词下推

- 支持算子：`eq/ne/gt/ge/lt/le/in/and`；其余算子返回 `E_QUERY`。
- 列裁剪：仅保留 `select ∪ 谓词引用` 列，未引用列的块永不解码。
- 下推：AND 平坦化后逐谓词用块 min/max 元数据证明“不可能匹配”并跳过，
  计划摘要记录每条跳过依据；下推只减少候选块，不改变结果语义。
- 规范化：AND 展平 + 叶子按稳定 key 排序 + select 排序，语义相同但空白、
  括号分组、AND 顺序不同的查询得到同一 canonical 文本与 query_id。

## 5. 三重约束并行调度

`scheduler.py` 用一个锁 + 条件变量维护：全局在跑任务数、已预留解码内存、
每段在跑计数。任务仅在同时满足
`在跑数 < max_parallelism`、
`已预留内存 + 本块内存 <= memory_budget_bytes`、
`该段在跑数 < per_segment_concurrency`
时启动，否则留在 FIFO 队列并记录 `wait_start_ms`；启动记录 `start_ms`，
完成记录 `end_ms`、占用内存、完成行数与编码方式。单块内存超预算直接
`E_BUDGET`（不存在合法调度），零/负预算在引擎构造期即 `E_BUDGET`。

结果装配只依赖确定性的块坐标与行序，与线程完成时序无关，因此任意合法
并行度下结果行集合、列序与结果哈希逐字节一致。

## 6. 版本、陈旧防护与整体替换

- 预算三要素任一变更 → `budget_version` 单调 +1；旧 `ScanResult.stats.stale=True`，
  旧任务标 `abandoned`；新一轮扫描整体替换，`ReplacementReport` 稳定列出
  受影响块集合与放弃任务数；`current_summary()` 拒绝预算版本不一致的结果进入汇总。
- 强制改写编码 → `encoding_selection_version` 递增并失效旧结果；段定义带
  `format_version`，统计与快照均记录每段格式版本。

## 7. 一致性校验（E_DIVERGE）

对同一查询分别在并行度 1 与高并行度、以及 RAW/DICTIONARY/RLE 三种编码下
执行，逐行逐列比较并比对 SHA-256 结果哈希与 `rows_completed/rows_matched`。
首个差异携带行标识、列名、两条路径取值；`kind=semantic` 表示结果语义差异，
`kind=statistical` 表示仅统计口径差异。

## 8. SQLite 快照与整体替换恢复

表：`snapshots / segment_summaries / task_traces / replacement_diffs`。
快照保存段集合摘要（含 file_hash、格式版本、压缩标记、行数）、编码选择
版本、预算版本、任务轨迹、结果摘要与统计。重开时先与当前磁盘/当前编码
版本比对：任一不一致即把旧快照置只读（`read_only=1`、`superseded_by`），
差异写入 `replacement_diffs` 并可列出，当前视图整体替换而非增量合并。

## 9. 异常注入（只改输入/参数，不改源码）

| 注入 | 做法 | 业务码 |
|---|---|---|
| payload_truncation | TOC 声明载荷长度比实际少 1 字节，触发长度自洽失败 | E_PAYLOAD |
| encoding_mismatch | 翻转块编码标记但保留原载荷，解码结构校验失败 | E_ENCODING |
| zero_budget | memory_budget_bytes=0 作为参数传入 | E_BUDGET |

异常写入结构化日志，且已完成块级任务通过引擎历史保持可核验
（`verifiable_completed_tasks`）。

## 10. 需求到代码的对应

1. 段加载/编解码、整体拒绝：`segment.py`、`encoding.py`
2. 列裁剪/谓词下推/规范化/E_QUERY：`query.py`
3. 编码选择与内存估算联动：`encoding.py`（`select_encoding/estimate_encoding`）
4. 块级并行调度与时间线：`scheduler.py`、`engine.py`
5. 预算版本/陈旧防护/整体替换：`engine.py`（`update_budget/_invalidate_results`）
6. 跨并行度跨编码一致性：`engine.py`（`consistency_check/_compare_paths`）
7. 快照版本识别与整体替换：`snapshot.py`
8. 异常注入与结构化日志：`segment.py` 注入函数、`cli.py`、`structured_log.py`

## 11. Windows 运行方式

```powershell
py -3.11 -m pip install numpy
py -3.11 run_all.py            # 一键：样本+测试+三场景+异常码+一致性
py -3.11 -m colstore.ui        # 启动 Tkinter 界面
# 或双击 run_demo.bat / run_gui.bat
```

SQLite 与段文件均位于 `colstore/data`、`colstore/sample_data`，运行前请保证
项目目录可写。
