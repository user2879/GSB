# Edge Ledger

本地、确定性的指令样本构造与增量账本。项目不访问外部账号或业务网络资源，样本、模板修订链、配置、故障注入与编辑命令均由本地示例脚本生成。

原始任务说明保留在 `docs/requirements.md`。

## 快速开始

```bash
# 1. 使用发行版 Python 3.11 与 venv
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# 2. 生成确定性样本、配置、注入清单与编辑序列
python scripts/generate_samples.py --root .

# 3. 增量构建、输出报告
python -m edge_ledger.cli --root . build
python -m edge_ledger.cli --root . report

# 4. 完整验收（逐步编辑、增量/全量比对、强杀恢复）
python -m edge_ledger.cli --root . verify --output artifacts/verification.json

# 5. 测试（含 Tk 界面测试；无显示时 GUI 用例会跳过）
./scripts/tk_python.sh -m unittest discover -v
```

也可以使用 `make generate build verify test report`。GUI 使用 `make gui`；无显示环境可用 `./scripts/run_headless_gui_check.sh` 验证 Tk 界面构建。

## 交付内容

- `edge_ledger/`：校验、模板修订链、确定性渲染、定点评分、去重、调度、缓存键、快照、报告、CLI 与 Tk GUI。
- `data/`：10 个模板修订、20 条样本（三批有效样本与四类故障样本）、注入清单。
- `config/pipeline.yaml`：去重阈值、评分权重、定点档位、线程数、预算与配额。
- `commands/edit-sequence.yaml`：新增批次、单条替换/删除、模板新增/约束/回滚、阈值、量化与调度编辑。
- `state/`：提交后的不可变代际快照与 `CURRENT` 原子指针。
- `reports/latest.json` 与 `reports/latest.md`：可复核报告。
- `ledger/ledger.jsonl`：逐次构建的追加式摘要账本。
- `artifacts/verification.json`：最近一次自动验收证据。

## 命令

```bash
# 生成初始数据（会重建 data/samples/*.yaml）
python scripts/generate_samples.py --root .

# 增量构建；--full 强制冷启动全量构建；--state-dir 指定隔离快照目录
python -m edge_ledger.cli --root . build
python -m edge_ledger.cli --root . build --full --state-dir /tmp/edge-full

# 应用命令流后构建；命令失败时不会写入任何成功命令
python -m edge_ledger.cli --root . apply --commands commands/edit-sequence.yaml

# 报告、验收与 GUI
python -m edge_ledger.cli --root . report
python -m edge_ledger.cli --root . verify
./scripts/tk_python.sh -m edge_ledger.cli --root . gui

# 在阶段提交中模拟异常或真实 os._exit 强杀
python -m edge_ledger.cli --root . build --crash-after-stage render --crash-mode exit
```

## 数据模型

模板修订使用 `(template_id, revision)` 唯一键，`parent_revision` 必须显式存在，根修订为 `null`。每个模板恰有一个根，允许分支持有多个子修订，但任一有效链必须无环。

```yaml
templates:
  - template_id: tpl_summary
    revision: 2
    parent_revision: 1
    instruction_template: "Summarize {topic} for {audience} in {style}."
    answer_template: "For {audience}: {topic} ({style})."
    slots:
      - {name: audience, min_length: 3, max_length: 60}
    constraints:
      style: {allowed_values: [concise, detailed]}
```

有效槽位与约束按祖先链逐级合并。同名槽位由后出现的修订覆盖，追加槽位进入集合，`remove_slots` 可删除；每条槽位记录最终 `source_revision` 和完整 `overrides` 来源。模板占位符只允许简单 `{identifier}`，禁止格式说明符与非确定转换。

样本包含指令模板修订、回答模板修订、来源分片与 `slot_fills`。填充集合必须与指令模板有效槽位集合一致，回答模板槽位必须被覆盖。重复槽位按 `(priority, input_index)` 最大者胜出；同优先级时后出现者胜出，其余值写入 `suppressed`。

## 确定性与账本规则

### 渲染

- 槽位按名称升序形成填充顺序，并以内容寻址的 `render_cache_key` 复用。
- `render_digest = H(模板链指纹, 回答模板链指纹, 样本内容指纹, 槽位值集合, 指令文本, 回答文本)`。
- 所有哈希均为 UTF-8 规范 JSON 的 SHA-256；相同输入重复渲染逐字符一致。

### 定点质量分

质量分取值域为 `0..1_000_000` 的整数：

```text
raw = (完整度*Wc + 模板覆盖率*Wt + 回答长度衰减*Wa) / 1000
quantized = round_half_up(raw / quantum_step) * quantum_step
residual = raw - quantized
```

- 完整度：非空槽位 / 有效槽位数。
- 模板覆盖率：指令与回答模板实际引用的槽位 / 被引用槽位总数。
- 回答长度衰减：超过 `answer_soft_limit_chars` 后按 `answer_decay_per_char` 递减，受 `answer_floor` 约束。
- `Q100/Q1000/Q10000` 对应步长 `10000/1000/100`；小数位宽允许 `2/3/4/6`。

### 去重

文本经 NFKC、casefold、空白归一后生成词元、字符 n-gram 与紧凑串 n-gram 子词集合。相似度为整数千分比 Jaccard：

```text
similarity_permille = floor(|A∩B| * 1000 / |A∪B|)
```

相似度 `>= threshold_permille` 即视为近重复；恰等于阈值时固定判为重复，并写入 `boundary_pairs`。锚点按 `(-quality_score, sample_id)` 排序，锚点与所有仍未被分配的、相似度达阈值的样本组成星形簇，保留锚点；被剔除项记录 `keeper_id`、相似度和边界标记。

### 缓存与失效

| 阶段 | 缓存键上游内容 |
|---|---|
| 模板解析 | 模板标识、修订号、链指纹、有效模板指纹 |
| 渲染 | 指令/回答模板链指纹、样本内容指纹、去重后的槽位值 |
| 质量分 | 渲染摘要、评分参数指纹、量化档位与步长 |
| 去重 | 阈值、子词宽度、样本集合的样本/分数/渲染/子词指纹 |
| 统计 | 去重键、配置哈希、逐样本引用、全部输入样本引用、失败引用、逻辑点数 |

因此模板修订只重算引用该修订路径的样本；替换/删除只重算对应样本；阈值调整只重算去重与统计；量化调整只重算质量分、去重与统计。实际命中的持久化代际和缓存键写入 `execution.cache_ledger`。

### 调度

线程池最大并发取自配置。调度顺序先处理上一批待办，再按 `(source_shard, sample_id)` 稳定排序；单分片累计点数超过配额后，该分片其余样本本批记 `E_QUOTA_EXCEEDED`；全局预算不足的样本进入待办并记 `E_RENDER_BUDGET_PENDING`。渲染点数是调度容量，缓存命中同样占用逻辑点数，从而冷/热构建得到相同调度结论。每个任务另有 `H(source_shard, sample_id, stage)` 稳定键；线程完成顺序不影响最终输出。

## 快照与恢复

每次构建把加载校验、模板解析、调度、渲染、质量分、去重和统计产物，连同待办队列、已完成模板修订游标、已消耗预算、缓存和输入指纹写入 `state/.txn-*`。全部文件 `fsync` 后，目录整体改名到 `state/snapshots/<generation>`，最后原子替换 `state/CURRENT`。提交使用 POSIX 文件锁串行化代际分配。

`manifest.json` 记录：

- `global_version`（规范输入内容哈希）、`config_hash`、模板链指纹；
- 各阶段产物版本与文件 SHA-256；
- 已完成模板修订游标、待办队列、已消耗预算和父代际。

启动时读取 `CURRENT`，校验清单与每个载荷哈希，再比对输入版本和配置哈希：相同则 `resume`，变化则 `continue_incremental`，无有效快照则 `cold_build`。若 `CURRENT` 损坏，会回退到最近的有效代际。阶段写盘时被强杀不会改变 `CURRENT`；下次启动从上次已提交快照继续。

## 故障注入与业务码

`data/injections.yaml` 通过独立编号注入：

- `template_cycle`：制造模板父链环，记 `E_TEMPLATE_INVALID`；
- `missing_slot`：删除指定槽位填充，记 `E_SAMPLE_INVALID`；
- `missing_revision`：引用不存在的修订，记 `E_SAMPLE_INVALID`；
- `nonfinite_quality`：真实进入非有限浮点检查，记 `E_SCORE_NONFINITE`。

任何单条故障都不会中止其他样本。报告中的 `injection_hits` 列出注入编号、应用状态、命中阶段、业务码与消息。

其他业务码包括 `E_CONFIG_INVALID`、`E_COMMAND_INVALID`、`E_QUOTA_EXCEEDED`、`E_RENDER_BUDGET_PENDING` 与 `E_INJECTION_INVALID`。

## Tk 诊断界面

界面页签展示：

- 模板修订链、父修订、生效路径、有效槽位及来源修订；
- 逐样本渲染摘要、质量分、残差、去重簇、保留/剔除状态；
- 失败清单、业务码计数、待办队列和实际缓存复用/重算依据；
- 选中模板修订的指令/回答/槽位 JSON 编辑器；
- 去重阈值编辑与增量重算入口。

界面所有快速结论直接来自同一 `BuildResult`，并由 `compare_quick_view()` 与权威账本逐项对照；差异项携带所属阶段。

## 验收结果

本地验证覆盖 21 项单元/集成测试，全部通过。`verify` 逐步施加 10 条编辑命令，每一步比较增量结果与冷启动全量结果，并执行真实子进程 `os._exit(91)` 强杀。当前证据文件为 `artifacts/verification.json`，其中包含：

- 每步增量/全量 canonical 摘要一致；
- 模板修订新增不影响旧样本、约束修改只影响引用该路径的样本；
- 阈值变化不重算渲染/质量分；量化变化不重算渲染；
- 强杀前后 `CURRENT` 代际不变，重启后 canonical 结论与强杀前一致；
- 无变化重启除确定性失败样本的质量分失败记录外全部复用持久化产物。

### 关于“复用/重算清单”的逐字段比较

真实冷启动全量构建不会命中缓存，而增量构建会命中；要求两者实际 reuse/recompute 事件完全相同会与缓存验证目标矛盾。因此本实现把**权威构造结论**放在 canonical projection 中逐字段比较，包括所有样本输出、渲染摘要、质量分与残差、簇账本、统计、缓存键、失败与待办顺序；实际运行期复用/重算证据独立保存在 `execution.cache_ledger`，通过失效范围和命中依据断言验证。这样既不伪造冷启动命中，也保证增量与全量结论完全一致。

## 本机环境记录

工作机为 WSL Ubuntu 22.04，系统提供 Python `3.11.0rc1`，但 `python3.11-venv`、`python3-tk` 与 `xvfb` 未安装且当前账号无免密 sudo。实际落地采用：

- 从 Ubuntu 仓库下载 `python3.11-venv`、`python3-pip-whl`、`python3-setuptools-whl` 的 deb，用户态解包后建立 `.venv`；虚拟环境解释器仍为 `/usr/bin/python3.11`。
- 从 Ubuntu 仓库下载 Tk/Tcl 8.6、Xvfb 及所需运行库 deb，解包到 `.runtime/tkroot`；以 `LD_LIBRARY_PATH`、`TCL_LIBRARY`、`TK_LIBRARY` 使用。
- `.venv` 内安装 `numpy 2.4.6` 与 `PyYAML 6.0.4`。
- 在 Xvfb 显示服务下构建 Tk 根窗口、刷新模板/样本/失败表、编辑阈值并触发增量构建，快速结论与账本一致。
- `scripts/tk_python.sh` 在解释器启动前注入本地 Tk/Tcl 动态库与数据目录；无 `.runtime/tkroot` 的标准环境会透明回退到 `.venv/bin/python`。

在可安装发行版包的环境，直接执行 `scripts/setup_env.sh` 即可使用标准 `python3.11-venv`/`python3-tk`/`xvfb` 路径。
