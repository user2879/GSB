# 多轮会话状态管理管线与验证工具

这是一个纯本地、确定性、可离线重复运行的会话状态处理项目。它按事件到达顺序推进每个会话的状态机，维护槽位、置信度、转移与被拒绝转移，并支持事件追加、话语替换、轮次删除、槽位模式变更和合并策略切换后的精确增量重算。

项目只依赖 Python 3.10+ 与 PyYAML，不使用 Docker、数据库、网络服务或第三方编排组件。

## 目录

```text
session_pipeline/          核心库
  loading.py               输入加载、YAML/JSONL 校验、规范化
  merge.py                 类型解析、四类槽位合并、定点置信度
  engine.py                单会话状态机、轮级检查点、缓存键
  edits.py                 增量编辑计划应用
  injections.py            独立异常注入
  scheduler.py             线程池、会话配额、全局预算、稳定提交
  snapshot.py              阶段产物、临时目录、整体替换、CURRENT 指针
  runner.py                全量/增量入口、诊断报告
scripts/
  generate_samples.py      固定种子样本生成器
  build_full.py            串行全量构建入口
  run_incremental.py       增量运行/恢复入口
  verify_consistency.py    逐批次全量与增量逐字段对照
  fault_demo.py            异常注入演示
samples/scenario_01/       固定种子输入样本
tests/test_pipeline.py     自动化验收测试
artifacts/                 本次实际运行报告与验证结果
```

## 环境与运行

当前容器实际验证环境：Python `3.11.0rc1`、PyYAML `5.4.1`、Linux。代码只使用标准库与 PyYAML，路径和文件替换实现兼容 Windows；Windows 上可使用同样的命令（将 `python3` 换成 `py -3`）。

```bash
python3 -m pip install -r requirements.txt
python3 scripts/check_environment.py

# 重新生成固定种子样本
python3 scripts/generate_samples.py

# 串行全量构建
python3 scripts/build_full.py

# 从空工作区做增量运行
python3 scripts/run_incremental.py --reset

# 中断或强制终止后省略 --reset，会从最近完整快照恢复
python3 scripts/run_incremental.py

# 每个编辑批次都与对应串行全量重建逐字段对照
python3 scripts/verify_consistency.py

# 独立异常注入演示（默认普通运行不启用注入）
python3 scripts/fault_demo.py

# 真实强制终止、重启恢复与全量重建对照
python3 scripts/crash_recovery_demo.py

# 自动化测试
python3 -m unittest discover -s tests -v
```

Windows PowerShell 示例：

```powershell
py -3 -m pip install -r requirements.txt
py scripts\check_environment.py
py scripts\build_full.py
py scripts\run_incremental.py --reset
py scripts\verify_consistency.py
py scripts\crash_recovery_demo.py
py -m unittest discover -s tests -v
```

## 输入契约

### 事件流 `events.jsonl`

每行一个对象，核心字段示例：

```json
{"session_id":"S001","turn":1,"role":"user","text":"我想预订","slot_updates":[{"slot":"intent","value":"book","confidence":"0.800000","source":"explicit"}],"timestamp_ms":1000,"requested_target":"greeted"}
```

- `session_id` 必须非空；同一会话轮次从 1 开始严格递增且无空洞。
- `timestamp_ms` 非负，并在全局接收顺序上单调不减。
- `slot_updates` 可包含多个槽位更新；置信度使用十进制字符串或数值，范围为 `[0,1]`。
- `requested_target` 可选；存在时必须引用已声明状态，否则记录 `E_INPUT_INVALID` 并跳过该条事件。
- 单条事件合并异常记录 `E_SLOT_VALUE_INVALID`，单条事件转移异常记录对应业务码，其他事件与会话继续。

### 状态机 `state_machine.yaml`

包含状态集合、唯一初始状态、槽位模式、合并策略、冲突优先级与转移规则。槽位类型支持 `string`、`int`、`bool`、`decimal`、`enum`。规则按 `id` 字典序决定同一轮多个候选的胜者；规则条件支持 `eq/ne/gt/gte/lt/lte/exists/not_exists/in` 与 `text_contains`。

### 管线配置 `pipeline_config.json`

关键字段：

- `worker_count`：线程池并发数。
- `max_events_per_batch`：单个编辑批的事件变更上限。
- `global_state_budget`：全局状态条目预算。
- `session_state_quota`：单会话状态条目配额；条目数定义为 `1 + 已物化槽位数`。
- `snapshot_interval_batches`：快照批次间隔。
- `confidence_decimals`：置信度定点小数位宽。
- `conflict_priority`：同值/同置信度时的来源优先级。
- `business_codes`：业务码映射。
- `merge_strategy_overrides`：可选槽位策略覆盖。

### 增量编辑 `edit_plan.json`

支持 `append`、`replace_utterance`、`delete_turn`、`patch_slot_schema`、`set_merge_strategy`、`set_conflict_priority`。删除中间轮次时，同一会话后续轮次自动前移一格，保证最终事件流无空洞。

### 异常注入 `fault_injections.json`

支持 `illegal_slot_value`、`turn_jump`、`unknown_transition_target`、`quota_overflow`。注入只针对指定批次、会话和轮次，并在报告 `injection_hits` 中保留注入项编号。注入默认关闭，只有 `--injections` 或 `fault_demo.py` 启用。

## 状态演进语义

### 槽位合并

- `overwrite`：新值覆盖旧值；同值或同置信度时按来源优先级裁决。
- `max_value`：保留较大值；同值先比较置信度，再按来源优先级；同置信度按优先级。
- `weighted`：数值按置信度加权平均，置信度使用定点整数累积：
  `new = old + ((scale - old) * incoming + scale/2) / scale`。
- `reject_overwrite`：已有值时拒绝覆盖，原值保持不变。

所有置信度在内部以 `confidence_fixed = round(confidence * 10^confidence_decimals)` 保存。事件按轮次升序处理；每次槽位更新输出前后值、策略、来源、接受/拒绝结果、裁决依据、事件/参数指纹和缓存键。

### 转移

规则只从当前状态（或 `from: "*"`）出发。候选规则按 ID 字典序排序，取第一条。显式 `requested_target` 没有匹配规则时，保持原状态并记录 `E_TRANSITION_REJECTED`。转移记录包含前后状态、触发轮次、规则 ID、候选 ID 列表、相关参数指纹和缓存键。

## 增量污染与恢复

每个会话持久化轮级检查点，包括该轮后的状态、槽位快照、转移/拒绝/变更/失败计数和状态缓存键。编辑批次完成后：

1. 计算受影响会话、最早受影响轮次和污染槽位集合。
2. 未受污染会话直接深拷贝复用持久化状态，不重放任何事件。
3. 受污染会话从最早污染轮次之前的检查点恢复，只重放污染后缀。
4. 话语替换、删除/轮次重排、槽位模式变化与合并策略切换均进入同一污染模型。
5. 编辑后的结果与从空白状态按最终事件流全量重建的结果逐字段比较。

缓存键分层构造：槽位键由相关事件内容、槽位模式、合并策略与冲突参数派生；转移键由事件指纹、相关规则和前置状态键派生；会话状态键由状态、槽位键集合和转移模式派生。

## 调度、预算与快照

- 受影响会话通过 `ThreadPoolExecutor` 并行计算；最终按会话 ID 稳定顺序整会话提交。
- 单会话超过 `session_state_quota` 时停止该会话，记录 `E_SESSION_QUOTA_EXCEEDED` 和待处理游标。
- 全局预算不足以完整容纳某个会话时，不提交截断状态，记录 `E_STATE_BUDGET_EXHAUSTED` 和游标。
- 快照目录先写入临时目录，阶段产物 `loading/`、`state/`、`summary/` 完成后写清单，再整体重命名；最后原子替换 `CURRENT` 指针。
- `manifest.json` 记录全局版本、配置/状态机哈希、已完成批次数、三个阶段的版本与产物哈希。
- 中断信号只设置停止标记；当前批次完成并提交快照后以退出码 `0` 退出。
- 恢复只读取完整清单指向的快照。强制终止不生成退出报告，重启报告中的 `recovery.recovery_point` 给出恢复版本和批次。

## 输出

- `artifacts/full-report.json`：串行全量诊断。
- `artifacts/incremental-report.json`：最终增量诊断。
- `artifacts/verification.json`：逐批次全量/增量对照。
- `artifacts/fault-report.json`：异常注入命中与失败清单。
- `artifacts/recovery-report.json`：强制终止、恢复点与全量对照证据。
- `artifacts/test-results.txt`：自动化测试的实际运行输出。
- `artifacts/full-snapshot-work/snapshots/`：串行全量构建的原子快照。
- `artifacts/incremental-work/snapshots/`：原子快照与清单。

报告包含会话摘要、完整槽位与置信度、状态、转移轨迹、拒绝转移、槽位变更、预算消耗、污染/复用计数、快照版本、恢复点、注入命中与失败清单。报告使用同目录临时文件加 `os.replace` 原子写入。

## 本次实际验证结果

- 自动化测试：`12` 项通过，覆盖固定种子样本字节级复现、加载校验、合并策略、转移候选、逐批一致性、预算、会话配额、注入、真实 SIGTERM 优雅停止与强制终止恢复。
- 最终全量状态投影哈希：`981054ed8274a722bc64839c9dc6fb319ae3924cc8f3fe8bb51d3d1f37d1bdba`。
- 最终增量状态投影哈希：同一哈希；5 个会话全部提交，13 条转移、3 条拒绝转移。
- 增量最终批次只重算 4 个受影响会话、重放 12 条事件；`S003` 整会话复用且重放事件数为 0。
- 四个批次逐批对照均相等：第 1、3 批各复用 `S003`；第 1 批污染 3 个槽位，第 2 批 8 个，第 3 批 4 个。
- 强制终止测试在批次 1 快照后返回 `137`，且没有生成报告；重启从 `completed_batch=1` 的完整快照恢复，最终状态与全量重建一致。
- 注入演示命中 `INJ-UNKNOWN-TARGET`、`INJ-ILLEGAL-SLOT`、`INJ-TURN-JUMP`、`INJ-QUOTA`，分别产生输入、槽位、轮次与配额隔离记录。

## 判定边界

- 全量/增量一致性比较槽位值与置信度、当前状态、转移轨迹、拒绝转移、槽位变更诊断、状态/轨迹缓存键、待处理游标和状态条目数。
- `contamination` 与 `reuse` 是运行诊断字段：全量重建没有复用，增量运行会有污染和复用计数，因此不能把这两个运行元数据直接要求相等；验证器会单独确认复用会话逐字段不变且重放事件为 0。
- 本项目的“状态条目”按 `1 + 已物化槽位数` 计，便于对预算和会话配额做确定性验证。
