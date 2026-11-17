# 增量特征与样本链路组件

面向搜索推荐场景的本地、确定性在线特征与训练样本链路。实现使用 Python 3.10+ 标准库，不依赖网络、真实账号、外部服务或容器。数值聚合使用固定顺序的 `math.fsum`，因此不需要 NumPy 也能保证同一输入逐字段一致。

## 交付内容

- `feature_pipeline/`：加载校验、拓扑特征计算、失效重算、样本生成、异常注入、阶段快照与恢复、报告输出。
- `scripts/generate_samples.py`：固定随机种子 `173` 生成事件、定义表、标签、编辑序列和注入配置。
- `scripts/run_acceptance.py`：执行增量/全量、并发/串行、恢复、注入和业务码验收，并输出机器可读报告。
- `tests/`：13 个标准库 `unittest` 测试。
- `samples/`：本地确定性样本与编辑/异常注入配置。
- `artifacts/acceptance_report.json`：本次实际验收结果。

## 快速运行

```bash
# 1. 重新生成样本（可选，仓库已包含生成结果）
python3 scripts/generate_samples.py --output samples

# 2. 单次全量运行
python3 -m feature_pipeline run \
  --config samples/config.json \
  --events samples/events.jsonl \
  --features samples/features.json \
  --labels samples/labels.json \
  --state-dir runtime \
  --report reports/full.json \
  --reset

# 3. 查看完整报告
python3 -m json.tool reports/full.json

# 4. 运行全部测试
python3 -m unittest discover -s tests -v

# 5. 运行验收并生成 artifacts/acceptance_report.json
python3 scripts/run_acceptance.py
```

CLI 正常完成、收到可处理退出信号或在 `--stop-after` 阶段边界停止时均返回整数退出码 `0`。进程被 `SIGKILL` 等不可处理方式终止时不承诺生成退出报告；下一次启动只读取已完整替换的世代快照与 `CURRENT` 指针恢复。

## 输入格式

### 事件 JSONL

每行一个对象，只允许以下字段：

```json
{
  "event_id": "e0001",
  "user_id": "u1",
  "item_id": "i1",
  "event_type": "view",
  "timestamp": 1000,
  "context": {"price": 35.5, "dwell_ms": 437, "rating": 4.3}
}
```

- `event_id` 全局唯一；`user_id`、`item_id` 非空；`event_type` 必须在 `config.json` 枚举内。
- `timestamp` 为非负整数，输入文件内严格递增。
- `context` 只允许有限数值；未知事件顶级字段产生 `E_SCHEMA_UNKNOWN_FIELD`。
- 非法 JSON、重复 ID、未知行为类型、非有限数值等产生 `E_EVENT_INVALID`；非法时间戳产生 `E_TIMESTAMP_INVALID`。
- 坏行进入失败清单，其他合法事件继续处理。

### 特征定义表

```json
{
  "features": [
    {
      "name": "user_purchase_value_10s",
      "entity": "user",
      "window_ms": 10000,
      "aggregation": "sum",
      "source_field": "context.price",
      "empty_value": 0,
      "filter": {"field": "event_type", "equals": "purchase"},
      "upstream_features": []
    },
    {
      "name": "user_purchase_value_smooth_12s",
      "entity": "user",
      "window_ms": 12000,
      "aggregation": "mean",
      "source_feature": "user_purchase_value_10s",
      "empty_value": 0,
      "upstream_features": ["user_purchase_value_10s"]
    }
  ]
}
```

- `entity` 为 `user` 或 `item`。
- 聚合函数为 `count`、`sum`、`mean`、`latest`。
- `source_field` 使用点路径读取原始事件字段，例如 `context.price`。`source_feature` 使用同实体上游特征的逐事件快照值作为输入；它必须列在 `upstream_features` 中。
- `upstream_features` 同时定义依赖图和失效传播边。非源上游可只声明依赖关系；数值输入由 `source_feature` 明确指定。
- 窗口为闭区间 `[anchor_timestamp - window_ms, anchor_timestamp]`。
- 聚合输入固定按 `(event_timestamp, event_id)` 升序；`latest` 取排序后的最后一条。相同输入与定义下结果稳定。
- 没有符合过滤条件的事件时输出 `empty_value`，并设置 `empty: true`。
- 未知字段、重复特征名、缺失上游、自依赖等产生 `E_SCHEMA_UNKNOWN_FIELD`；依赖环产生 `E_FEATURE_CYCLE`。

### 样本标签定义

```json
{
  "name": "purchase_quality",
  "trigger_event_types": ["purchase"],
  "delay_ms": 5000,
  "feature_names": [
    "user_purchase_count_10s",
    "item_avg_price_10s"
  ],
  "positive_rule": {
    "all": [
      {"feature": "user_purchase_count_10s", "operator": "ge", "value": 1},
      {"feature": "item_avg_price_10s", "operator": "ge", "value": 60}
    ]
  },
  "negative_rule": {
    "any": [
      {"feature": "user_purchase_count_10s", "operator": "lt", "value": 1},
      {"feature": "item_avg_price_10s", "operator": "lt", "value": 60}
    ]
  }
}
```

- 触发事件在 `event_timestamp + delay_ms` 到达标签生效边界；只有事件游标到达该边界后才生成样本。
- 样本特征读取触发事件锚点处已经存在的特征快照，不会读取锚点之后到达的事件。
- 规则支持 `all`、`any`、`not`、`else` 和条件算子 `gt/ge/lt/le/eq/ne`。
- 先判定正例；正例不成立时再判定负例。两者都不成立时不生成样本并记录 `E_LABEL_UNDEFINED`。
- 每个样本含 `features`、`label`、`generation_timestamp`、`feature_snapshot_version`、`feature_cache_keys` 和空窗标记。

### 配置、编辑与注入

`samples/config.json`：

```json
{
  "allowed_event_types": ["view", "click", "add_to_cart", "purchase"],
  "max_workers": 4,
  "stage_delay_ms": 0
}
```

`samples/edits.json` 支持按顺序应用：

- `backfill`：按 `event_id` 修正历史事件，可修改 `timestamp` 与点路径 `patch`。修正后时间戳若发生变化且早于已完成事件游标，产生 `E_TIMESTAMP_INVALID`。
- `feature_window`：修改某特征 `window_ms`。
- `feature_aggregation`：切换某特征聚合函数。
- `label_delay`：修改标签生效延迟。

`samples/injections.json` 包含五类真实输入故障：非有限上下文、未知行为类型、未知字段、标签引用缺失特征、回填修正时间戳早于游标。报告中的 `injected_items` 给出注入编号、命中状态与业务码映射。

## 核心设计

### 确定性特征

每个特征值记录包含：

- `feature`、`entity_type`、`entity_id`、`event_id`、`event_timestamp`。
- `value`、`empty`。
- `definition_hash`。
- `cache_key`：由特征定义指纹、窗口参数、聚合函数指纹、锚点、以及窗口内按固定顺序排列的事件内容指纹/上游缓存键共同派生。

事件指纹只覆盖该特征实际依赖的实体、时间、源字段与过滤字段，避免无关上下文回填导致伪失效。缓存键不含运行时刻或全局版本，因此增量与全量结果可直接比较。

### 失效闭包

- 特征窗口、聚合函数、源字段、过滤条件等定义变化：直接失效该特征。
- 回填：根据变化后的时间戳、实体、源字段与过滤字段定位直接依赖特征。
- 直接失效特征通过依赖图传播到全部下游特征；其他特征保留原缓存记录。
- 新事件仅为其锚点追加缺少的特征记录，不重算未失效历史记录。
- 报告和持久账本中的特征名均按字典序排列，并记录回填事件影响的特征与样本。

### 样本失效

样本在其锚点特征快照（值、空窗标记或缓存键）发生变化时失效重生成；标签延迟变化会改变生成边界，从而重算受影响样本。未变化的样本直接复用。

### 并发与顺序

特征按拓扑层级调度。同层任务提交线程池，最大并发数来自 `max_workers`；每个任务携带 `(拓扑位次, 特征名)` 稳定排序键。线程结果消费、特征记录、缓存账本、样本、失败清单都再做规范排序，因此并发度不改变逐字段结果。

### 阶段快照与恢复

- 阶段顺序：`events` → `features` → `samples`。
- 每阶段先写 `runtime/.tmp/commit-*`，再以目录原子重命名提交为 `runtime/snapshots/gNNNNNN`，最后原子替换 `runtime/CURRENT`。
- `manifest.json` 记录全局版本、定义表哈希、已完成事件游标、完成阶段、阶段产物版本与文件哈希。
- 半写入的 `.tmp` 目录、无有效清单的世代、哈希不匹配的世代均被忽略。
- 启动时从 `CURRENT` 或最高有效完整世代恢复，并在报告 `recovery_point` 输出恢复点。
- 若特征阶段未完成，恢复时保守重算全部特征以确保正确性；正常增量运行仍按精确失效集合执行。

### 报告

每次完整运行输出 JSON，主要字段：

- `global_version`、`completed_stages`、`snapshot_versions`、`definition_hash`、`event_cursor`。
- `invalidation_ledger`：失效/复用/计算特征、失效/重生成/复用样本、原因。
- `backfill_impacts`：本轮回填事件 ID、修正后时间戳、影响特征和样本；`backfill_ledger` 保存跨世代累计回填账本。
- `stage_durations_ms`、`empty_window_feature_count`。
- `injected_items` 与 `failures`。
- `feature_snapshot_version`、`sample_snapshot_version`。
- `recovery_point`、`stopped_gracefully`、`stop_reason`。

## 实际验证结果

环境：Python `3.11.0rc1`，Linux，本地文件系统；未使用 Docker、网络或外部服务。当前环境未提供 pip/NumPy，因此实现采用零第三方运行时依赖；这不会改变固定运算顺序和确定性要求。

验证命令：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_acceptance.py
```

本次结果：

- `unittest`：13 个测试全部通过。
- 增量编辑序列：基础 10 事件 → 追加 24 事件 → 窗口变更 → 聚合函数切换与标签延迟调整 → `e0010` 回填。
- 最终特征记录 `216` 条、样本 `4` 条、空窗特征 `40` 条。
- 增量与全量重建：事件层、特征层、样本层、规范缓存账本全部逐字段相等。
- `max_workers=4` 与 `max_workers=1`：全部逐字段相等。
- 窗口变更仅失效 `item_view_count_5s`；回填仅失效源字段相关特征及其下游，`item_view_count_5s`、`user_purchase_count_10s`、`user_view_count_5s` 直接复用。
- `SIGTERM` 在当前阶段提交后以退出码 `0` 结束；续跑结果与全量重建逐字段相等。
- 五类注入全部命中：`E_EVENT_INVALID`、`E_EVENT_INVALID`、`E_SCHEMA_UNKNOWN_FIELD`、`E_SCHEMA_UNKNOWN_FIELD`、`E_TIMESTAMP_INVALID`。
- 特征快照内容版本在增量与全量中均为：
  `a077c324565620583f9faedfc1588d08f14547b8cae11a0ffc7f0ddb584f2f96`。

完整机器可读结果见 `artifacts/acceptance_report.json`。

## 说明

- 每轮运行的失效/复用集合是过程观测数据；从空白全量重建不会经历同样的历史失效过程。因此逐字段等价比较针对最终特征、样本和规范缓存账本，过程账本则单独核验失效范围与复用集合。本次验收同时核验了这两部分。
- `feature_snapshot_version` 使用样本锚点特征快照的内容哈希，而不是递增的全局世代号；它可在增量与全量之间直接比较。全局世代号仍记录在清单与报告中。
- 当前样本触发类型由 `trigger_event_types` 配置；到达每个触发事件的标签延迟边界时生成一个样本。该字段是本地标签定义的扩展，便于限定训练目标事件。
