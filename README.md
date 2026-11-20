# 分布式哈希连接与 Shuffle 分区组件（本地确定性实现）

纯 Python（3.10+）+ NumPy 的本地组件，无网络、无外部服务、无 GPU、无容器。
两张逻辑表与一条内连接查询由本地 JSON 声明，执行分两阶段：

1. **Shuffle 分区**：连接键先量化为定点整数，再按显式哈希规则映射到固定分区数。
2. **分区内哈希连接**：左侧为构建侧建哈希表，右侧按桶内行序表探测输出。

桶文件支持三种整数格式版本（v1/v2/v3），支持记录级升级、按桶检查点续算、
强制终止恢复、会话快照整体替换恢复，以及独立的 E_VERIFY 校验工具。

## 目录结构

- `dhj/` — 组件源码
  - `errors.py`：`E_SCHEMA / E_BUCKET / E_RESUME / E_VERIFY / E_SNAPSHOT`
  - `jsonio.py`：规范化 JSON（排序键、无空白）与内容摘要
  - `fixedpoint.py`：定点量化（半离零舍入、64 位范围、误差上界）
  - `hashing.py`：FNV-1a 64 位哈希与 `hash_int(q) mod n` 分区规则
  - `schema.py`：表/查询加载与强校验（整体拒绝、字段路径聚合）
  - `bucket.py`：v1/v2/v3 二进制桶编解码、严格校验、记录级升级
  - `shuffle.py`：确定性分区、分版本生成、跨版本等价判定
  - `join.py`：谓词、构建侧哈希表、连接、逐行误差归因、续算交叉校验
  - `stats.py`：NumPy 列式分区/全局统计与定点误差上界
  - `checkpoint.py`：生成/升级/连接三阶段按桶检查点
  - `snapshot.py`：快照版本识别与整体替换恢复
  - `engine.py`：升级编排、崩溃恢复扫描、快照状态比对
  - `verify.py`：独立重算分区归属与连接结果（E_VERIFY）
- `tools/run_experiment.py`：可重复端到端实验
- `tools/verify_tool.py`：独立校验 CLI
- `tools/crash_worker.py`：真实强制终止 worker（SIGKILL / Windows `os._exit`）
- `tests/`：31 个 `unittest` 用例
- `data/`：确定性样本（`sample.json` 整数键、`query_float.json` 浮点键查询）
- `buckets/`：各版本桶、升级目录、连接结果、检查点
- `snapshots/`：会话快照与只读备份
- `logs/`：结构化 JSONL 日志
- `artifacts/`：实验报告

## 桶文件二进制格式

固定头 + 若干长度前缀段，全部大端序：

| 区域 | 内容 |
| --- | --- |
| 固定头 | 魔数 `DHJBUCK1`、2 字节版本号、4 字节头 JSON 长度、规范化头 JSON |
| v1 | `SUMMARY` 段：行数 + 每行 `(row_id:int64, row_digest:uint64)` |
| v2 | v1 全部内容 + `COLUMNS` 段：列式列名/类型/非空位图/int64 或 float64 |
| v3 | v2 全部内容 + `KEYS` 段（连接键定点量化值 int64）+ `ORDER` 段（行序表 `(row_id, ordinal)`） |

段头为 `(tag:uint32, length:uint32)`。解码为**全量严格校验**：魔数、版本受支持、
各长度字段自洽、必需段齐全、列行数与摘要行数一致、v2+ 重算行摘要逐行匹配、
v3 重算量化键与行序表，任何非法输入抛 `E_BUCKET`（带版本号与偏移），不返回部分结果。

## 显式哈希与量化规则

- 量化：`q = round_half_away_from_zero(value * 2^fraction_bits)`，有符号 64 位；
  空键保留 `INT64_MIN`，不参与分区与匹配。
- 分区：`partition = FNV1A64(canonical_json(q)) mod num_partitions`。
- 行摘要：`FNV1A64(canonical_json({表名, row_id, side, 连接键, 全行值, 空值位图}))`，
  与桶版本、分区号无关，因此升级路径与直写路径可逐字节比对。
- 桶内行序：`(q_key 是否空, q_key, row_id)` 升序，保证 v3 字节确定性。

## 升级与续算语义

- v1→v3 补齐 `columns, row_count, quantized_join_keys, row_order_table`；
  v2→v3 仅补齐后两个字段。v1 无列数据，升级须传入已校验源表，并先逐行匹配摘要。
- 每桶在内存完成记录级升级 → 原子重写（同目录临时文件 + `os.replace`），
  旧桶保留为 `*.bak-vN` 只读（0444）备份，每完成一桶 fsync 检查点。
- 重启只依据检查点处理未完成桶；已完成桶不会重复处理。
- 未完整提交/不可解析桶标记为 `unknown`，清除其检查点条目并重跑。
- 连接续算复用未受影响分片结果，随后做全量交叉校验，首个不一致分片抛 `E_RESUME`。

## 快照恢复

快照持久化表摘要、桶格式版本集合、分区参数指纹、已完成桶指纹、连接结果摘要，
并自带内容摘要。重开时与磁盘当前状态逐项比较：一致则续用；任一不一致（含不可解析
状态）整体把桶/结果/快照目录移动为只读 `*.replaced` 备份后重建，绝不增量合并。

## 运行方式（Windows PowerShell / 类 Unix 均可）

```powershell
python -m pip install -r requirements.txt
python tools/run_experiment.py
python tools/run_experiment.py --query data/query_float.json --partitions 5 --fraction-bits 4
python tools/verify_tool.py --sample data/sample.json --bucket-dir buckets/gen-v3 `
  --result-dir buckets/results-full --partitions 4 --fraction-bits 4
python -m unittest discover -s tests -v
```

## 已验证结果

- v1/v2/v3 三版本生成、严格解码与非法桶 `E_BUCKET`（版本号+偏移）。
- v1→v3、v2→v3 升级产物与直写 v3 **逐字节一致**，旧桶只读备份齐全。
- 生成/升级/连接三阶段强制终止（含真实 SIGKILL 子进程）后按检查点恢复，
  恢复后桶指纹与无中断运行一致。
- 续算连接与全量重算结果文件逐字节相同；不一致时 `E_RESUME` 定位首个分片。
- 分区数 3/5/8 下浮点键连接行集合与逐行归因完全一致（与分区/执行顺序无关）。
- 独立校验工具对直写 v3、升级 v3 重算通过；篡改时返回 `E_VERIFY`/`E_BUCKET`。
- 快照指纹不一致触发整体替换并保留只读备份。

详细机读结果见 `artifacts/experiment-report.json`、
`artifacts/experiment-report-int.json`、`artifacts/experiment-report-float.json`。
