# 中文工单意图分层路由与槽位回填工作台

这是一个从零实现的本地确定性工作台，面向中文工单话语 JSONL、分层路由配置和编辑命令流。它把领域路由、领域内意图路由、槽位回填、置信度再校准和阶段快照串成一条可恢复的状态演进流水线。

核心保证：

- 两级路由、否定消解、最长触发词匹配和槽位候选排序全部使用确定性的整数/有理数逻辑；
- 任意编辑命令序列后的增量结果，与相同最终数据/配置从空白全量重建的业务结果逐字段一致；
- 每条话语的阶段缓存键由内容指纹、触发词指纹、槽位模式指纹与配置指纹派生；
- 单批路由点数预算耗尽时保留稳定顺序的待办队列，下一批优先处理；
- 快照在临时目录完整写入后整体替换，支持强制终止后的游标续跑与陈旧快照检测；
- 线程池并发与串行执行的业务结果、抑制清单、预算账本和失败顺序一致；
- 独立故障注入覆盖非有限权重、未声明槽位、轮次跳变、词项越界和运行期异常隔离。

运行不访问网络、不使用 Docker、不需要账号或外部服务。

## 环境

已验证环境为 Python 3.11.0rc1、numpy 1.26.4、PyYAML 6.0.1；代码使用 Python 3.10+ 语法。

Windows 示例：

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install numpy==1.26.4 PyYAML==6.0.1
```

也可直接执行：

```bat
python -m pip install -r requirements.txt
```

Linux/macOS 使用对应的 `python3` 与虚拟环境激活命令即可。

## 快速开始

```bash
# 1. 重新生成固定样本（仓库已带一份生成结果）
python -m routing_workbench generate-samples --samples samples

# 2. 应用命令流并增量运行
python -m routing_workbench run \
  --samples samples \
  --state state \
  --report reports/run.json

# 3. 无改动再跑：全部阶段命中缓存
python -m routing_workbench run \
  --samples samples \
  --state state \
  --report reports/run-noop.json

# 4. 完整验收阶梯（增量/全量、串行/并发、恢复、陈旧、预算、注入、pytest）
python -m routing_workbench verify \
  --samples samples \
  --workdir .verify \
  --report reports/verification.json

# 5. 单独运行真实故障注入
python -m routing_workbench inject \
  --samples samples \
  --report reports/injection-report.json

# 6. 查看快照清单
python -m routing_workbench snapshot-info --state state
```

`--full` 忽略已有状态；`--no-resume` 从输入基线重建。`run` 返回退出码 0；模拟终止返回 75，可随后不带 `--crash-after-commands` 续跑。

## 样本

`samples/` 由固定代码生成，不依赖随机数：

- `utterances.jsonl`：13 条话语，覆盖账号与账单两个领域、6 个意图、槽位真值、否定、触发词吞并和边界样本；
- `routing.yaml`：领域触发词、意图触发词、否定规则、两级阈值、槽位模式、必需槽位、量化参数、并发、预算和快照间隔；
- `commands.json`：12 条编辑命令，包含触发词增删改、槽位模式切换、阈值/再校准调整和单条话语替换；
- `injections.yaml`：5 个独立真实故障；
- `vocabulary.txt`：受控词表文本副本。

## 输入契约

### 工单话语 JSONL

每行一个 JSON 对象。推荐字段：

```json
{"utterance_id":"S001-U1","session_id":"S001","turn":1,"tokens":["忘记","密码","手机号","13800138000"],"domain":"account","intent":"password_reset","slots":{"phone":"13800138000"}}
```

兼容别名：`id/uid`、`conversation_id/session`、`turn_index/round/seq`、`terms/words`、`domain_label/domain_truth`、`intent_label/intent_truth`、`slot_truth/slot_triples`。

校验规则：

- `session_id + turn` 唯一；同一会话轮次从 1 开始且严格连续；
- 词项非空并且全部落在受控词表；
- 领域标签属于配置领域集合，意图标签属于该领域；
- 槽位真值支持字典或 `[slot, value, ...]` 三元组列表，槽位名必须已声明；
- 单行错误产生 `E_ROUTE_INVALID` 失败记录，不阻断其他行。

### 路由配置 YAML

完整样本见 `samples/routing.yaml`。最小结构：

```yaml
vocabulary: [账号, 密码, 手机号, 13800138000]
domains:
  account:
    triggers:
      - {term: 账号, weight: 3}
      - {term: 密码, weight: 4}
    intents:
      password_reset:
        triggers:
          - {term: 忘记密码, weight: 5}
          - {term: 密码, weight: 1}
        slot_patterns: [phone_after_label]
        required_slots: [phone]
slot_patterns:
  phone_after_label:
    name: phone_after_label
    intent: password_reset
    slot: phone
    tokens: [手机号, "{phone}"]
    window: 2
    direction: after
slots:
  phone: {type: phone}
required_slots:
  password_reset: [phone]
negation_rules:
  - {negators: [不是, 没有], scope: 2, mode: cancel, direction: before}
thresholds: {domain: "0.20", intent: "0.30"}
fixed_decimals: 5
quantization_levels: 100000
quantization_scale: 100000
thread_pool_size: 4
max_thread_pool_size: 8
batch_point_budget: 4096
snapshot_interval_commands: 2
recalibration:
  completeness_weight: "0.12"
  suppression_penalty: "0.03"
  minimum: "0.0"
  maximum: "1.0"
  suppression_cap: 4
```

槽位模式有两种写法：

- 占位符模式：`tokens: [手机号, "{phone}"]`，花括号位置消费一个词项；
- 锚点模式：不给占位符，使用 `direction: before|after|both` 和 `window` 捕获锚点附近词项。

顶层槽位模式可被多个意图复用；引用时以引用该模式的意图为所有者。`quantization_levels` 与 `quantization_scale` 表示定点刻度分母，例如 100000 对应 5 位小数的整数表示。

配置不合法统一产生 `E_ROUTE_INVALID`，包括：阈越界、非有限/负数触发词权重、表内重复触发词、重复否定词、槽位模式引用未声明槽位、量化参数非法、并发超过上限、预算非正等。

### 编辑命令流

`commands.json` 可以是数组，也可以形如 `{"commands": [...]}`。命令同时支持 `type/op/action`。已实现：

| 类型 | 作用 |
| --- | --- |
| `replace_utterance` / `add_utterance` / `upsert_utterance` | 替换、增加或合并单条话语 |
| `delete_utterance` | 删除话语 |
| `add_trigger` / `update_trigger` / `delete_trigger` | 按领域或意图编辑触发词 |
| `switch_slot_pattern` / `update_slot_pattern` | 切换槽位模式的词项、窗口、方向或槽位 |
| `set_required_slots` | 修改意图必需槽位 |
| `set_threshold` / `update_config` / `set_recalibration` | 修改路由、再校准和量化参数 |
| `set_concurrency` / `set_budget` | 修改运行参数 |
| `noop` / `checkpoint` | 只推进命令游标 |

命令执行前保存配置与话语快照。命令载荷非法或使配置整体不合法时回滚该命令，写入 `E_ROUTE_INVALID`，后续命令继续执行。

## 算法

### 触发词匹配

触发词在 UTF-8 文本上用字符位置匹配，覆盖同一触发词跨词项出现的情况。所有命中候选先按：

1. 字符匹配长度降序；
2. 同长按触发词字典序；
3. 起始字符位置升序；
4. 条目 ID 字典序

排序。与已接受候选重叠者记为 `overlap_longest_match` 抑制项，并记录胜出条目。

### 领域分数

对每个领域：

```text
domain_score = Σ(去重后命中触发词权重) / Σ(领域全部触发词权重)
```

只选择分数不低于领域阈值的候选；最高分并列时按领域名字典序选择，并标记 `score_tie_lexicographic`。分数恰等于阈值时标记 `score_equals_threshold`。

### 意图分数与否定

仅在被选领域内计算：

```text
intent_score = Σ(去重后有效触发词权重) / Σ(意图全部触发词权重)
```

否定规则按方向和词项距离检查触发词前的否定词。`cancel` 将有效权重置零并记入抑制清单；`penalty` 将权重乘以 `1 - penalty`。否定后的意图选择仍遵循阈值、字典序和边界标记规则。

### 定点量化

所有比例先以 `Fraction` 精确计算，再按半入方式量化：

```text
score_int = round_half_up(score * quantization_scale)
```

报告中同时保留 `*_int` 与固定小数位字符串，例如 `31250` 和 `"0.31250"`。修改 `fixed_decimals` 或量化刻度会使缓存键与输出位数真实变化。

### 槽位回填

模式在话语词项序列上滑窗匹配。同一槽位的候选按：

1. 匹配长度降序；
2. 起始词项位置升序；
3. 取值字典序；
4. 模式名字典序

选出唯一值，其余进入 `slot_suppression`。`completeness` 为已填必需槽位数 / 必需槽位总数。

### 置信度再校准

使用定点整数执行：

```text
after = clamp(before + round(completeness_weight × completeness)
                     - round(suppression_penalty × min(suppressed_count, cap)),
              min, max)
```

其中 `before` 是意图分数定点值，`suppressed_count` 是槽位抑制候选数。最终值同时裁剪到 `[0, 1]`。

## 缓存与精确失效

每个阶段缓存键包含：

| 阶段 | 内容键 |
| --- | --- |
| validation | 话语内容指纹 + 词表/领域/意图集合/槽位声明 |
| domain | 话语内容指纹 + 全领域触发词 + 领域阈值 + 量化参数 |
| intent | 话语内容指纹 + 已选领域 + 该领域意图触发词/否定规则/意图阈值 |
| slots | 话语内容指纹 + 已选意图 + 槽位模式/声明/必需槽位 |
| recalibration | 话语内容指纹 + 已选领域/意图 + 槽位产物摘要 + 基础置信度整数 + 再校准参数 |

因此：

- 修改某领域触发词会重算领域阶段；其他领域的意图阶段仍可复用；
- 修改某意图触发词只重算该领域的意图阶段及其真实受影响下游；
- 切换槽位模式只重算对应意图的槽位与再校准阶段；
- 替换一条话语只影响该话语；
- 并发度、预算和快照间隔等运行参数不会改变业务缓存键。

报告中 `cache.reuse_recompute` 按 `(utterance_id, stage)` 稳定排序，并附带 `task_key = sha256(utterance_id + stage)`。每个结果包含 `stage_cache_keys`、引用的触发词条目和话语集合。

## 预算调度

- 预算由配置或 `--batch-budget` 提供，每个 `run` 调用为一个新批次；
- 待办项优先，其后按话语 ID 稳定顺序；
- 每条需要重算的话语消耗 `词项数 × 非空触发词命中数` 点；
- 完全缓存命中的话语不扣预算；
- 首个放不下的未缓存话语及其后的未缓存项保留到待办队列，并输出 `E_ROUTE_BUDGET_PENDING`；
- 账本逐条记录命令序号、话语 ID、词项数、命中数、消耗量、消耗前后累计和剩余预算；
- 下一次运行优先处理待办，预算足够时业务结果与一次性全量运行一致。

## 快照与恢复

`state/snapshot/` 包含 `manifest.json`、`state.json` 和 `stages/*.json`。提交过程：

1. 在 `snapshot.tmp-*` 完整写入所有阶段、状态与清单并 `fsync`；
2. 将当前代目录整体改名为 `snapshot.backup-*`；
3. 将临时目录整体改名为 `snapshot/`；
4. 删除更旧备份。启动时可从中断在步骤 2/3 之间的备份恢复。

清单记录：

- 格式版本与全局递增版本；
- 当前配置哈希、触发词表指纹；
- 已完成命令游标与已应用命令前缀指纹；
- 已消耗预算、待办队列、数据指纹；
- validation/domain/intent/slots/recalibration 的摘要、条目数、修订号。

续跑时按“已提交游标前缀”计算期望配置哈希，并核对基础输入指纹、命令前缀与可选全局版本。因此：

- 中断后重跑会从最后快照继续；
- 追加新命令可以直接续跑；
- 修改已提交命令、基础输入或配置会输出 `E_SNAPSHOT_STALE`，随后从空白全量重建；
- 全局版本不匹配同样输出 `E_SNAPSHOT_STALE`。

## 并发与故障隔离

- 话语级阶段任务由 `ThreadPoolExecutor` 执行，工作线程数不超过配置上限；
- 结果按稳定话语顺序合并，线程调度不改变输出；
- 单个话语阶段异常记录话语 ID、阶段和业务码，其他话语继续；
- `injections.yaml` 会在独立配置/数据副本上真实制造五类故障，并输出注入编号、命中状态、业务码、有效话语数与结果数。

## 主要输出字段

报告包含：

- `summary`：有效/无效/结果/失败/待办/边界数量；
- `results`：领域与意图判定、两级定点分数、贡献与抑制、槽位、再校准前后置信度、阶段缓存键与引用；
- `failures`：稳定排序的业务码、话语、阶段、命令序号和字段；
- `boundary_samples`：阈值等值与词典序并列样本；
- `cache`：按话语与阶段的复用/重算/待办记录；
- `budget`：批次上限、剩余量、完整扣减账本与待办；
- `diagnostics`：陈旧快照等系统级诊断；
- `snapshot`：恢复状态、全局版本、命令游标与阶段版本；
- `business_digest`：仅由业务结果、失败、边界样本、待办与注入命中派生的稳定摘要。

完整示例可通过快速开始命令生成。

## 测试与验收

```bash
python -m pytest -q
python -m routing_workbench verify --samples samples --workdir .verify --report reports/verification.json
```

本机实际验证结果记录在：

- `artifacts/pytest.txt`
- `artifacts/acceptance-report.json`
- `reports/verification.json`

验收阶梯逐项检查：样本路由、增量=全量、无改动全复用、串行=并发、强制终止续跑、陈旧快照全量重建、预算待办与优先续跑、五类故障注入隔离以及 pytest 全绿。
