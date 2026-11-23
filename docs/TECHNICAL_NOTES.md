# 告警降噪与根因关联管线 — 技术说明

## 1. 工程结构

```
alert_pipeline/
  errors.py        E_SPEC / E_FIELD / E_REPLAY / E_SNAPSHOT 四类错误码
  util.py          规范化 JSON 字节、SHA-256 摘要、稳定 ID
  models.py        NormalizedAlert / AggregateAlert / RootCandidate / Correlation
  codec.py         v1/v2/v3 编解码、版本探测、显式补齐规则 FILL_RULES
  topology.py      依赖图强校验、确定性 Kahn 拓扑、BFS 最短推导路径
  rules.py         静默窗口与根因规则规格、强校验、生效窗口选择
  loader.py        JSON/NDJSON 加载与流级 E_SPEC 校验
  dedup.py         指纹+静默窗口降噪、开放窗口状态快照/恢复
  correlation.py   上下游根因候选、层级与稳定拓扑排序、关联账本数据
  fixedpoint.py    整数定点统计、三段误差界、不可区分决胜
  scheduler.py     全局+每服务双配额调度、真实超时与重试、峰值并发
  ledger.py        SQLite 单事务批次、UNKNOWN 标记、摘要校验、陈旧标记
  replay.py        恢复后逐字段重放比对（E_REPLAY 定位首个分歧）
  evolution.py     版本单调校验、陈旧防护、整体视图替换 diff
  pipeline.py      端到端编排（运行/崩溃注入/恢复/演进）
  samplegen.py     numpy 整数种子确定性样本与异常注入样本
  logging_setup.py 结构化 JSONL 日志
run_demo.py        四个验收场景的可复现演示
tests/             25 个 unittest 用例
samples/           生成的本地样本与 injected/ 异常样本
```

## 2. 关键算法与判定规则

### 2.1 三版本归一（业务点 2）
- 版本来自显式 `version` 字段；未知版本、缺字段、类型错误、负静默窗口、悬空根因引用均 `E_FIELD`，带 1 基行号与字段名。
- 补齐规则集中在 `codec.FILL_RULES`，每个被补齐字段都能回溯到规则文本：
  - v1 补齐 `fingerprint`（由 service/severity/timestamp 派生 `fp_`+hash16）、
    `silence_window_ms`（规则默认值）、`labels`、`root_cause_refs`、`strategy_id`；
  - v2 补齐 `labels`、`root_cause_refs`、`strategy_id`；
  - v3 仅在缺省时补齐 `labels`。
- 归一摘要只包含逻辑字段（不含行号），键排序、紧凑分隔、UTF-8，保证任意行序下逐字节一致；
  身份 ID 同样只基于逻辑内容。行号仅作为 provenance 保留。

### 2.2 强校验分层（业务点 1）
- 记录内字段问题（含静默窗口为负）在 codec 边界统一 `E_FIELD`；
- 跨记录/跨文件不变量在 loader/topology/rules 边界 `E_SPEC`：服务 ID 唯一、依赖引用存在、
  图无环、时间戳单调不减、指纹正则合法、窗口非负整数。错误信息给出首个违规行号与字段。

### 2.3 降噪窗口边界（业务点 3）
- 分组键 `(service_id, fingerprint)`；首条告警以 `anchor` 开窗口，
  `ts <= anchor + silence_window_ms`（闭区间）判定为抑制，
  第一条 `ts > anchor + silence_window_ms` 的告警重开窗口。
- 抑制数为纯整数累加；聚合输出首/末时刻、抑制条数、参与行号集合、窗口版本、策略与来源 schema 版本。
- 输入按 `(timestamp, line, id)` 排序处理，分组遍历顺序不影响结果。
- 仍开放的窗口通过 `open_state()/restore_state()` 随批次落盘，恢复后续算窗口边界完全一致。

### 2.4 根因关联（业务点 4）
- 边 `a -> b` 表示 a 依赖 b；BFS 顺边得到上游候选、逆边得到下游候选，路径即推导路径。
- 候选排序键 `(layer, direction 秩, 拓扑秩, service_id)`；拓扑序由确定性 Kahn（就绪集合排序）产生，
  与节点插入顺序无关。账本逐条记录聚合 ID、候选集合、路径与规则版本。

### 2.5 双配额调度（业务点 5）
- 条件变量同时约束全局在途任务数与单服务在途任务数；配额不足时排队，
  记录等待开始/结束的 `time.monotonic_ns()` 毫秒值。
- 每次尝试有真实墙钟截止时间（`Event.wait(timeout)`）；超时按 `max_attempts` 重试，
  任务函数为纯函数，重试不改变批语义。批次输出起止单调时钟、完成条数、峰值并发、超时/重试计数。

### 2.6 定点统计与误差界（业务点 6）
- 会话缩放因子 `S = 10**k`（默认 k=4），统计值为非负整数定点，显式 floor 截断，
  每指标仅一次整除，中间求和均为精确整数。
- 误差界（ULP 整数计数）拆三段：
  - 量化截断 `quantization_truncation`：每次 floor 至多 1 ULP；
  - 静默窗口边界 `silence_window_boundary`：按窗口数/总量的保守 ULP 上界；
  - 依赖层级归一 `dependency_layer_normalization`：按命中层敏感比例的保守 ULP 上界；
  - 取非零最大段标注为主导段。
- 两指纹定点值之差不超过二者合成 ULP 界时判为不可区分，按 `(固定值降序, 指纹升序)` 稳定决胜并打标。

### 2.7 演进与整体替换（业务点 7）
- 规则文档声明 `declared_version`，演进必须单调递增，否则 `E_SPEC`。
- 演进在单事务内把旧批次/聚合/关联整体置 `stale=1`，当前视图只读取最新版本，不做增量合并；
  输出新增/删除/变更聚合 ID 的稳定排序 diff。新版本写入按 `(batch, version, id)` 隔离，不会冲刷陈旧标记。

### 2.8 恢复、重放与异常注入（业务点 8）
- 每批次在一个 SQLite 事务内写批次摘要、聚合、关联、降噪开放状态与进度游标；
  强制终止只可能留下完整提交或预写的 `UNKNOWN` 标记。重启只信任 COMMITTED 行，
  UNKNOWN 批次要求重跑，且不生成任何退出报告。
- 恢复时先重算输入/图/规则摘要确认输入未被静默修改，再确定性重放至最后 COMMITTED 游标，
  对密封聚合、关联结论与开放窗口状态逐字段比对，首个分歧以 `E_REPLAY` 报出聚合 ID 与字段。
- 三类注入：依赖成环 `E_SPEC`、负静默窗口 `E_FIELD`、账本 payload 篡改致摘要不符 `E_SNAPSHOT`。

## 3. 需求追溯矩阵

| 业务点 | 实现位置 | 测试 |
| --- | --- | --- |
| 1 强校验 E_SPEC | topology.py, rules.py, loader.py | test_validation_topology.py, test_pipeline_e2e.py |
| 2 三版本归一 E_FIELD | codec.py, models.py | test_codec.py |
| 3 降噪聚合/边界 | dedup.py | test_dedup_correlation.py |
| 4 根因关联/账本 | correlation.py, ledger.py | test_dedup_correlation.py, test_pipeline_e2e.py |
| 5 双配额调度/超时重试 | scheduler.py | test_scheduler.py |
| 6 定点误差界/不可区分 | fixedpoint.py | test_fixedpoint.py |
| 7 陈旧防护/整体替换 | evolution.py, ledger.py, pipeline.py | test_pipeline_e2e.py |
| 8 恢复重放/异常注入 | ledger.py, replay.py, pipeline.py, run_demo.py | test_pipeline_e2e.py |

## 4. Windows 本地复现步骤

```bat
py -3.11 -m pip install numpy==1.26.4
py -3.11 run_demo.py
py -3.11 -m unittest discover -s tests -v
```

不使用 Docker；SQLite 仅用标准库 `sqlite3`。所有路径均为相对路径，换行写文件显式使用 `\n`，
规范化字节与 JSON 键排序保证跨平台逐字节一致。

## 5. 实际验证结果（本次执行环境 Python 3.11，Linux 容器内等价验证）

- `python -m unittest discover -s tests`：25/25 通过。
- `python run_demo.py`：正常运行、崩溃注入→UNKNOWN→恢复重放、版本演进整体替换、
  三类异常码（E_SPEC/E_FIELD/E_SNAPSHOT）全部符合预期。
- 同种子重复生成样本：四个样本文件 SHA-256 完全一致（确定性可复现）。
- 账本目录与结构化日志 `logs/pipeline.jsonl` 均在工作区自动创建。
