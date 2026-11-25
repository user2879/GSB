# 列式块组多表连接代价估算与回环执行系统 — 技术说明

## 1. 总览

系统在纯本地、零网络依赖下完成一条多表连接查询的「编译 → 枚举 → 定点估价 →
选优 → 分片 → 回环 TCP 执行 → 回执归一 → SQLite 账本持久化 → 崩溃恢复重放」全链路。

目录结构：

- `coljoin/errors.py`：业务错误码 `E_SPEC/E_FORMAT/E_REPLAY/E_PORT/E_PROTOCOL`。
- `coljoin/canonical.py`：与字典/哈希遍历顺序无关的规范化 JSON 与 SHA-256 摘要。
- `coljoin/prng.py`、`coljoin/storage.py`：SplitMix64 固定种子样本、plain/dictionary/rle 编码、NumPy 直方图。
- `coljoin/loader.py`、`coljoin/model.py`：块组/查询/权重/会话加载与强校验。
- `coljoin/statistics.py`：统计口径指纹、选择率、NDV 与等值连接基数。
- `coljoin/planner.py`：连接图、连通序枚举、笛卡尔积剪除、三算法组合、内容型计划标识。
- `coljoin/fixedpoint.py`、`coljoin/cost.py`：整数定点算术、饱和/floor 规则、误差区间、稳定裁决。
- `coljoin/receipts.py`：q1/q2/q3 三版回执编解码与跨版本归一。
- `coljoin/invalidation.py`：分层指纹与依赖链失效传播。
- `coljoin/executor.py`、`coljoin/scheduler.py`、`coljoin/protocol.py`：真实回环 TCP 子进程与双约束调度。
- `coljoin/executor_logic.py`、`coljoin/sharding.py`：确定性连接聚合运算与分片切分。
- `coljoin/ledger.py`：SQLite 原子事务、快照整体替换、恢复与重放校验。
- `scripts/`：样本生成、演示 CLI；`tests/`：35 个 unittest 用例。

## 2. 强校验（要求 1）

- 块组标识唯一、列名唯一、类型仅 `integer/float`、编码仅 `plain/dictionary/rle`。
- 行数为非负整数；空值位图长度必须等于行数；直方图区间首尾相接、不重叠且计数和等于行数。
- 连接/过滤/投影/分组/聚合引用列必须存在；代价权重为非负整数且和落在声明窗口。
- 任一非法立即抛 `E_SPEC`，携带首个非法对象 `object` 与 `field`（回执错误额外带 `line`）。

## 3. 计划枚举与规范化（要求 2）

- 连接图先做连通性检查；不连通（必然产生无等值谓词笛卡尔积）直接 `E_SPEC`。
- 仅枚举连通排列：每步新块组必须与已加入集合存在等值边，从根上剪除笛卡尔积顺序。
- 每个连接步骤独立选择 `hash/sort_merge/nested_loop`，计划数 = 连通序数 × 3^(n-1)。
- 计划标识 `P` + 规范化计划体 SHA-256 前 16 位，是内容指纹而非枚举序号；
  计划集合按 plan_id 排序返回，任意声明编辑顺序/枚举顺序下逐字节一致。

## 4. 整数定点代价模型（要求 3）

- 会话 `scale`（10 的幂）把行计数类指标量化为定点整数；权重为无量纲整数。
- 加权采用「先乘后反量化」的整数乘除：`(metric * weight) // scale`，除法一律 floor 截断。
- 所有中间量夹到 63 位有符号饱和区间 `[-2^62, 2^62-1]`，饱和上限随估算结果显式输出。
- 每次 floor 除法登记 1 个截断单位，形成对称误差区间 `[-epsilon, epsilon]` 与逐源清单。
- 代价向量四元组：`total_cost/access_rows/compares/peak_memory`，全部为 Python int。
- 统计口径版本 = 全部块组直方图/NDV 的规范化 SHA-256；重排块组加载顺序估算逐位一致。

## 5. 误差内稳定裁决（要求 4）

- 排序键仅含整数：`(total_cost, plan_id)`，不出现任何浮点胜负判断。
- 当候选与榜首的整数差值 `delta <= epsilon_winner + epsilon_candidate` 时，
  判定为误差不可区分，显式决胜顺序为 `plan_id 字典序 → 算子序列摘要序`，
  并在结果中标注 `error_influenced=true`，附两候选代价向量、差值与容忍度。
- 测试在 `scale=1/10/1000/100000` 下验证胜出计划不变，即不随小数位宽翻转结论。

## 6. 三版回执归一（要求 5）

- 磁盘为 JSON Lines；q1 仅 shard_id+聚合，q2 增 plan_version/elapsed_ms，
  q3 再增 executor_id/cost_vector/clock_start_ms/clock_end_ms。
- 低版本缺失字段按 `DEFAULTS` 显式补齐，归一结果携带 `_source_format`、
  `_filled_fields`（含每条补齐依据 basis）与 `_normalized_version`。
- 未知版本、必需字段缺失、负耗时、代价向量长度不符均抛 `E_FORMAT`，定位到行号与字段名。

## 7. 失效传播与仅失效子集重算（要求 6）

依赖链分层指纹：

1. `data_digest`：块组空值位图 + 列编码载荷；
2. `stats_digest`：直方图/NDV 统计口径；
3. `cost_input_digest`：统计口径 + 权重 + 缩放/粒度；
4. `plan_digest`：规范化计划体；
5. `shard_digest`：计划 + 分片批次；
6. `receipt_digest`：采纳回执。

- 块组数据/直方图变更只失效引用该块组的计划（直接失效）；
  权重/缩放/统计口径变更驱动全部计划重估。
- 受影响计划的分片与采纳回执作为间接失效，按 bg_id/plan_id 稳定排序输出，每步带 basis。
- 计划版本号单调递增；未变更计划的估算摘要与采纳回执因输入指纹不变而逐字节保持。

## 8. 回环 TCP 调度（要求 7）

- 执行器为独立子进程，在 `127.0.0.1` 真实 `bind/listen/accept`；
  主进程真实 socket 连接，JSONL 分帧 + HELLO/READY 握手，收发均设真实超时。
- 端口先探测空闲高位端口；声明端口被占用时按 `E_PORT` 报告。
- 双约束：全局 `max_in_flight` 与单执行器 `per_executor_slots`；
  越界即排队，等待起止用 `time.monotonic_ns()//1e6` 记录，不含任何人造 sleep 排队。
- 等待配额期间回收的回执由 `_acquire_handle` 显式返回，杜绝共享缓冲丢帧。

## 9. 账本、快照与崩溃恢复（要求 8）

- 每个分片在同一 SQLite 事务内写 batches/invalidations/cursors，带格式版本与三类内容摘要。
- 快照指纹与当前账本不一致时，整体重写快照并把旧文件保留为 `0444` 只读备份。
- 恢复只承认 `status='committed'` 记录；未提交事务由 SQLite 回滚，
  其分片在 `unknown_shards` 中要求重跑，配额随进程退出自然回收。
- 重放逐分片比对回执/代价摘要，首个分歧定位到 shard_id、plan_version、字段名并抛 `E_REPLAY`。
- 系统不假设强制终止时一定存在退出报告；崩溃演示通过 `SystemExit` 在提交边界模拟 kill。

## 10. 验证结果

执行 `python3 -m unittest discover -s tests -v`，35 个用例全部通过，覆盖：

- 规格非法对象/字段 E_SPEC（8）；
- 计划枚举确定性、笛卡尔积剪除、三算法、定点代价与误差裁决（9）；
- 三版回执归一与 E_FORMAT 行号定位（6）；
- 数据/权重失效传播与未失效摘要稳定（3）；
- 真实回环调度、双约束、排队时钟、端口占用 E_PORT（3）；
- 账本原子提交、快照只读备份、未知分片重跑、E_REPLAY 与强制 kill 恢复（6）。

常用命令：

```bash
python3 -m scripts.generate_sample --workspace workspace
python3 -m scripts.run_demo --workspace workspace plan
python3 -m scripts.run_demo --workspace workspace run --executors 3 --max-in-flight 2 --per-executor 1 --shard-rows 30
python3 -m scripts.run_demo --workspace workspace invalidate --delta 11
python3 -m scripts.run_demo --workspace workspace crash
```

## 11. 环境备注

- 已验证 Python 3.11、NumPy 1.26.4、sqlite 3.37.2，满足 Python>=3.10、NumPy>=1.24。
- 当前容器缺 `libtk8.6.so`，标准库 Tk 可视化查看器未启用；核心功能不依赖 Tk，
  安装 `python3-tk` 后可另行扩展，不影响账本与执行链路。
