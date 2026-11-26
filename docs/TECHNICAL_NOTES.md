# 事件论元复核账本：技术说明

## 1. 落地范围

项目在无 Docker、无外部服务的本地目录中实现：

- 确定性样本生成：`src/arg_review/samplegen.py`，固定种子 `20261127`。
- JSONL 语料、JSON 模式修订链、缩进 YAML 配置和 JSONL 编辑命令加载。
- 模式版本驱动的触发匹配、定点论元打分、必填/互斥裁决、跨事件引用合并。
- 内容指纹账本、精确失效、持久化复用和增量/全量等价核验。
- 三重预算约束下的线程池调度与稳定待办队列。
- 临时目录写入、符号链接整体切换的阶段快照与恢复判定。
- 五类真实输入故障注入、隔离和诊断报告。

运行时只访问本地文件和 `127.0.0.1` 回环检查，不访问未授权目标。

## 2. 目录

- `samples/events.jsonl`：7 条主样本事件。
- `samples/invalid_events.jsonl`：5 类结构故障加 1 行坏 JSON。
- `samples/schema_revisions.json`：`rev-1` 至 `rev-3` 的无环父修订链。
- `samples/extraction_config.yaml`：权重、定点档位、阈值、预算和并发。
- `samples/edit_commands.jsonl`：13 步编辑序列。
- `samples/injections.json`：5 个注入项。
- `src/arg_review/`：核心包。
- `tests/`：25 个 pytest 用例。
- `artifacts/ledger.json`：持久化内容指纹账本。
- `artifacts/snapshots/current`：当前阶段快照符号链接。
- `artifacts/edit_equivalence_report.json`：13 步增量/全量等价报告。
- `artifacts/diagnostic_report.json`：基线报告与注入套件汇总。

## 3. 偏移与排序约定

样本中的 `trigger_span` 和论元 `start/end` 采用半开词项索引：

- `0 <= start <= end <= len(tokens)`。
- `[start, end)` 对应 `tokens[start:end]`。
- 触发词首要排序使用派生字符跨度：短语词字符数加内部空格数。

所有对外列表使用稳定键排序：

- 事件：`event_id`。
- 论元候选：`(role, start, end, arg_id)`。
- 触发输出：`(start, end, term)`。
- 引用处理：`(source_event_id, source_start, source_ref, target_event_id, target_start, relation_id)`。
- 合并组：首个成员的 `(event_id, start, arg_id)`。

## 4. 阶段与业务码

阶段为 `validation -> trigger_match -> argument_extraction -> role_adjudication -> reference_merge`。

主要业务码：

- `E_SCHEMA_INVALID`：未知事件类型、越界偏移、非千分比置信度、坏引用、角色不在集合、断链等。
- `E_EVENT_INCOMPLETE` / `E_REQUIRED_ROLE_MISSING`：必填角色无过线候选。
- `E_MUTEX_SUPPRESSED`：互斥角色候选被抑制。
- `E_BOUNDARY_CANDIDATE`：与阈值边界距离不超过量化半步。
- `E_REF_MERGED` / `E_REF_MERGE_EDGE_ACCEPTED`：引用边或合并组结论。
- `E_DEDUP_BELOW_THRESHOLD` / `E_REF_MUTEX_SUPPRESSED`：引用边不合并。
- `E_ARG_BUDGET_EXCEEDED`：全局打点或单事件配额阻断。
- `E_ARG_PENDING`：稳定待办单元。

报告中的 `business_code_counts` 与 `stage_fault_code_counts` 分离；阶段异常使用
`STAGE_*` 代码，并保留 `injection_id`、事件和故障类型。

## 5. 定点置信度

分数使用 `fractions.Fraction` 精确计算，不使用浮点作为裁决依据。公式按
角色名升序、候选起始位置升序处理候选，组件累加顺序固定为：

1. 角色权重；
2. 触发距离衰减；
3. 片段长度归一；
4. 原始千分比先验。

定点配置：

- `frac_digits`：小数位宽，`scale = 10 ** frac_digits`。
- `level`：`fine=1`、`standard=5`、`coarse=25` 的量化量子倍数。
- 每个候选记录 `unquantized_fixed_int`、`fixed_residual_*`、`quantum`、
  `quantized_fixed_int`、`rounding_direction` 和量化残差。
- 边界候选输出高一位宽 `frac_digits + 1` 的参考值。

切换位宽会改变 `scale`、未量化整数和残差；切换档位会改变 `quantum`、
量化整数、量化残差和舍入方向。

## 6. 指纹与精确失效

账本位于 `artifacts/ledger.json`，以内容缓存键保存产物，并以稳定 artifact ID
维护当前 manifest：

- `trigger:<event_id>`
- `arg:<event_id>:<arg_id>`
- `decision:<event_id>`
- `merge:<content_fingerprint>`

触发键包含事件词项内容指纹、触发规则的来源修订和规则包指纹。论元键包含：

- 词项/论元内容指纹；
- 触发结果缓存键和 `trigger_match_id`；
- 角色规则切片与角色权重；
- 抽取参数；
- 量化位宽和档位指纹。

裁决键包含全部候选键、必填/互斥规则切片和过线阈值。合并键包含排序后的论元
缓存键、关系指纹和去重阈值。

修订链按事件类型解析“最后实际变更该规则切片的修订”。因此只改 meeting 互斥
规则时，其他事件类型的触发和论元可继续复用；meeting 裁决及引用下游失效。
回滚到历史修订时，若内容键仍在内容寻址账本中，可直接复用当时产物，同时把
活动修订号和顺序型 `merge_id` 按当前状态规范化，避免残留旧版本位置。

## 7. 增量与全量一致性

`arg-review verify-edits` 对每一步命令：

1. 在同一个暖账本目录执行增量运行；
2. 深拷贝相同最终状态到新的空白目录一次性全量运行；
3. 比较 `canonical_projection` 的规范 JSON。

投影包含校验结果、触发位置/抑制、全部论元字段与量化残差、裁决/抑制/边界、
引用边/合并组、manifest 缓存键和稳定顺序数组。运行态的线程峰值、快照路径和
账本命中/未命中统计属于执行历史，不进入业务投影。

## 8. 预算与并发

`BudgetScheduler` 按 `(event_id, role)` 稳定排序工作单元：

- `global_arg_score_points`：全部论元候选打点预算。
- `per_event_arg_quota`：单事件候选打点配额。
- `max_trigger_matches_per_batch`：单批事件级触发匹配任务数。
- `budget.max_workers`：`ThreadPoolExecutor` 最大线程数，调度器记录峰值并断言不超过配置。

严格按稳定前缀准入；遇到首个全局或单事件超限单元时记录
`E_ARG_BUDGET_EXCEEDED`，该单元及后缀保留为 `E_ARG_PENDING`。提高预算后的下一
批从队首待办开始，顺序不变。

## 9. 快照与注入

`SnapshotStore` 先把阶段 JSON 写入 `tmp-v*` 临时目录，再重命名为不可变快照目录，
最后通过 `current.new-*` 符号链接和 `os.replace` 整体切换 `current`。恢复时清理
遗留临时目录；若 `current/commit.json` 可用则复用，否则判定全量重建。

注入配置支持：

- `offset_out_of_bounds`
- `bad_confidence`
- `missing_relation_arg`
- `unknown_role`
- `broken_revision_chain`

诊断套件对每个注入项使用独立状态目录，避免全局断链污染其他用例；报告写入注入
编号、命中状态、业务码、`STAGE_*` 阶段码和存活事件。持久化状态在移除注入后可
续跑，并与干净全量投影一致。

## 10. 常用命令

```bash
source .venv/bin/activate
python scripts/generate_samples.py
python scripts/check_loopback.py
pytest -q
arg-review run --samples samples --state-dir artifacts
arg-review verify-edits --samples samples --state-dir artifacts/verification \
  --report artifacts/edit_equivalence_report.json
arg-review diagnose --samples samples --state-dir artifacts
arg-review all --samples samples --state-dir artifacts
```

一键脚本：`bash scripts/run_acceptance.sh`。

## 11. 环境备注

本机为 Python 3.11.0rc1。标准 `python3.11 -m venv .venv` 因 Debian/Ubuntu 缺少
`ensurepip` 组件失败，且当前账号不能免密 `sudo apt install python3.11-venv`。
实际使用已安装的 `virtualenv -p python3.11 .venv` 创建等价虚拟环境，并在其中
安装 `numpy` 与 `pytest`。运行时代码不依赖 PyYAML，内置仅覆盖本项目配置格式的
YAML 子集解析器；若环境已安装 PyYAML，也可自行替换为 `yaml.safe_load`。
