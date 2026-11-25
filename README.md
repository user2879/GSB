# 对话式答案证据链诊断组件

这是一个从空白目录首次落地的 Linux 本地项目，用于诊断“多轮检索 -> 证据去重 -> 话题归属 -> 定点置信度/引用链”链路。项目不依赖网络、外部数据集、Docker 或容器，样本、配置和命令流全部由固定整数种子确定性生成。

## 快速开始

当前项目已经创建 `.venv` 并验证 `numpy==1.26.4`。若在另一台 WSL Ubuntu 上首次运行：

```bash
python3 --version                 # 需要 >= 3.10
python3 -m venv .venv             # Debian/Ubuntu 如缺 ensurepip，先 sudo apt install python3-venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选 GUI；本项目已准备无 root 的本地 Tk 依赖
# 新环境执行：bash scripts/install_local_tk.sh
```

一键生成样本、全量/增量对照、误差诊断、崩溃恢复验证：

```bash
python -m evidence_diag --root . demo
```

分步执行：

```bash
python scripts/generate_sample.py --root .
python -m evidence_diag --root . run --mode full
python -m evidence_diag --root . run --mode incremental
python -m evidence_diag --root . diagnose
python -m evidence_diag --root . snapshot-run --mode incremental --reset
python -m evidence_diag --root . snapshot-run --mode incremental --resume
python -m unittest discover -v
python -m evidence_diag --root . gui --smoke-test
```

`--root` 是全局参数，应放在子命令之前。可选 Tk 面板：

```bash
python -m evidence_diag --root . gui --demo
```

当前基座自带 Python 的 `_tkinter` 缺少系统动态库且无 sudo。项目已把 Ubuntu 官方 `libtk8.6/libtcl8.6` 等依赖无 root 解包到 `.local/tk`，GUI 启动时按绝对路径预加载；`gui --smoke-test` 已验证 Tk 8.6.12 可创建窗口。新机器可运行 `bash scripts/install_local_tk.sh` 复现。无显示环境时核心 CLI 与全部测试仍不受影响。

## 目录

```text
config/demo.json           检索、话题、定点、预算与参考位宽配置
sample/session.json        固定种子生成的轮次、查询、通道声明、证据库
sample/commands.jsonl      确定性编辑命令流
sample/generation.json     种子及内容指纹
snapshots/                 原子阶段快照版本与 CURRENT 指针
reports/                   等价性、定点误差、快照恢复与最终结果报告
evidence_diag/             可集成 Python 包与 CLI
scripts/generate_sample.py 独立样本生成入口
tests/                     26 项单元/集成测试
scripts/install_local_tk.sh 无 root 的本地 Tk 依赖安装脚本
.local/tk/                 当前环境使用的 Tk 8.6 本地动态库与 Tcl/Tk 数据
```

## 数据与命令

配置包含：

- `channel_weights`：召回通道权重，有限非负数。
- `fixed_point`：小数位宽 `{8,12,16,20}`、量化步长 `{1,2,4,8}`、舍入模式 `{half_up,half_even,down}`。
- `topic.switch_threshold_bp`：组合话题边界阈值，整数基点。
- `confidence`：轮次距离衰减、跨话题惩罚、答案结论阈值。
- `limits`：线程池上限 `1..32` 与单批证据点数预算。
- `reference`：诊断用高精度参考域。

每轮包含严格连续的 `turn_id` 和 `channels: {通道名: [证据ID...]}`。每条证据拥有全局唯一 `evidence_id`、文本及权重。同一证据可在一轮内重复、可跨通道出现，也可被多轮、多话题引用。

命令流支持：

`add_turn`、`replace_turn_query`、`delete_turn`、`set_topic_threshold`、`set_channel_weight`、`set_evidence_weight`、`set_evidence_text`、`upsert_evidence`、`set_channel_declaration`、`set_fixed_point`、`set_budget`、`set_max_workers`、`set_answer_threshold`、`set_turn_decay`、`set_cross_topic_penalty`。

删除轮次后会按当前顺序重新编号以保证“严格递增无空洞”。非法命令事务回滚，不改变已有配置/样本，只追加一条带 `E_INPUT_INVALID` 的隔离失败记录。命令序号必须从 1 连续递增。

## 计算口径

### 1. 召回与去重

每个通道以自己的定点权重给证据贡献得分。同一证据在一个通道内的重复出现累计计次；跨通道冲突时按 `(得分降序, 通道名字典序升序)` 选择权威来源，并记录全部 `source_channels`、`covered_channels`、`score_breakdown` 和 `reference_count`。

### 2. 话题切分

相邻轮的边界分数为：

```text
combined_bp = (query_similarity_bp * 3 + evidence_similarity_bp * 2) // 5
split = combined_bp < threshold_bp
```

`query_similarity_bp` 是查询词/CJK 二元组 Jaccard 相似度；`evidence_similarity_bp` 来自**去重后的证据集合** Jaccard，因此召回/去重变化会向话题阶段传播。恰等阈值固定采用“继续同一话题”，输出 `exact_equal=true`、`decision=continue`、`equal_rule=continue_on_equal`。话题 ID 由起始轮内容与证据分布的内容指纹生成，不包含定点权威标签，避免跨位宽误报。

证据归属记录全部引用轮次与全部话题 ID；跨两个及以上话题时 `cross_topic=true`，不会静默归到单话题。

### 3. 定点置信度

所有权重先量化到定点整数。对每轮引用证据按 `evidence_id` 字典序排序后逐项累加，顺序固定。单条贡献：

```text
reliability = authority_score / max_authority_score_in_turn
effective_base = evidence_weight * reliability
distance_decay = turn_decay ^ (current_turn - first_turn_in_topic)
contribution = effective_base * distance_decay * cross_topic_factor
confidence = sum(contribution) / sum(effective_base)
```

`cross_topic_factor = 1 - cross_topic_penalty`，仅跨话题证据使用。每次乘除都按配置的定点位宽、量化步长和舍入模式重新量化。输出同时包含整数单位、十进制显示、分项贡献、引用链和累加顺序。切换位宽/量化档位会改变真实中间整数和最终置信度。

### 4. 误差归因

诊断以 `reference` 高精度域完整重算为参考，在召回、去重、话题归属、置信度四阶段输出逐轮逐证据的 `actual`、`reference`、`delta`、`exceeds_tolerance`，并标注每轮首个超差阶段。答案结论在两域不一致时输出：

- `dedupe_merge_order`：权威通道/合并结果发生变化；
- `cross_topic_penalty`：跨话题集合变化或惩罚项量化差异超过容差；
- `rounding_direction`：目标域相对参考域向上或向下舍入导致翻转。

每次翻转包含 `causes` 和 `primary_cause`，同时保留两域阈值、置信度与结论。

## 增量与失效

每个阶段产物有内容派生缓存键：

- 召回：轮次内容指纹、通道声明、通道权重参数、定点域；
- 去重：召回产物指纹、定点域；
- 话题归属：查询/证据集合指纹、边界键、话题起点内容、定点域；
- 置信度：去重与归属指纹、证据内容/权重指纹、衰减/惩罚/阈值参数、定点域。

替换单轮查询只使该轮召回/去重/置信度和相邻边界/所属话题失效；改证据权重只使引用该证据的置信度及相关话题失效；改通道权重只影响声明该通道的轮次。`invalidation` 报告逐命令列出失效/复用轮次、话题、证据及原因。实际缓存命中数在独立的 `execution` 元数据中，因此 `mode=full` 与 `mode=incremental` 的规范化业务结果和逻辑失效清单可逐字段比较。

预算按待合成轮次的去重证据条数计费，轮次升序调度。放不下的轮次进入 `todo`，标记 `E_BUDGET_PENDING`；后续批次继续保留并优先处理。预算提高后可自动清空待办。

## 并发与快照

线程池最大并发数严格取自 `limits.max_workers`。任务键为 `turn_id:stage` 或 `left:right:topic-boundary`，提交和回收均排序；因此调度顺序和并发度不改变业务产物。

快照提交过程：

1. 在 `snapshots/.tmp-*` 中写完整 state、result、execution、四阶段产物和缓存；
2. 生成文件哈希及 `manifest.json`；
3. 目录整体 `rename` 到 `snapshots/versions/vNNNNNNNN`；
4. 原子替换 `snapshots/CURRENT` 指针。

`manifest.json` 记录全局版本、引擎版本、命令游标、配置哈希、证据库指纹、基准配置/证据指纹、命令前缀哈希、各阶段版本和文件哈希。提交使用 Linux `flock`，并检查期望父版本；父版本变化时抛出版本冲突，不逐文件覆盖。

恢复时只接受完整且哈希校验通过的版本。命令游标、配置哈希、证据指纹、命令前缀任一不符即 `full_recompute`；全部匹配则从快照中的阶段缓存续跑。强制终止后遗留的 `.tmp-*` 会被忽略并可安全清理。

## 实际验证

`python -m unittest discover -v`：26 项全部通过。`reports/` 中保留了本次实际运行结果，详见 [VALIDATION.md](VALIDATION.md) 和 [docs/DESIGN.md](docs/DESIGN.md)。

关键结果：

- 18 个游标（基线 + 17 条命令）增量与全量逐字段一致；
- 命令 9 把预算降至 5 后，7 个轮次标记 `E_BUDGET_PENDING`，恢复预算后清空；
- 命令 13 的负通道权重被回滚并隔离为 `E_INPUT_INVALID`；
- 8 位/4 量化目标域相对 24 位参考域，8 轮全部出现首超差，置信度最大绝对差约 `0.01518`；
- 1 个答案结论翻转，归因为去重权威变化（同时列出跨话题惩罚与舍入方向证据）；
- 在命令 10 提交途中以退出码 87 强制终止，重启从游标 9 续跑到 17，结果哈希与全量回放完全一致；
- 错误基准配置哈希触发 `full_recompute/base_config_hash_mismatch`。

## 说明

- 当前样本故意包含一个非法命令和一次低预算批次，以验证隔离、待办与等价性。
- 删除轮次会重编号后续轮次，因此轮次距离与下游缓存键可能变化；报告会按实际依赖列出失效项。
- Tk 面板只读取诊断报告，不参与计算；无 Tk 动态库时 CLI 仍完整可用。
