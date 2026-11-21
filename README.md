# NER 确定性并行调度与资源预算诊断工作台

本工作区实现了一个完全离线、可从零复现的中文字级命名实体识别（NER）诊断工作台。它包含固定种子合成语料生成器、严格输入校验、字级哈希特征、定点发射打分、对数域 BIESO 维特比解码、实体片段修复、全局字级预算与 LRU 字块缓存双重调度、阶段化原子快照、故障注入隔离、JSON 诊断报告与自包含 HTML 可视化面板。

实现不使用 Docker、不访问网络、不引入非确定性随机初始化。核心结果以整数和稳定排序键生成；线程完成顺序与快照提交时机不会改变确定性结果投影。

## 环境

- Python 3.10+；本机验证版本为 Python 3.11.0rc1。
- numpy 1.26.4、PyYAML 6.0.2、pytest 8.3.3（以本地安装版本为准）。
- 工作区已准备 `.venv`（使用系统可用的本地包，无外部模型或数据下载）。

```bash
# 本机 ensurepip 不可用时使用已安装的 virtualenv；工作区已准备好 .venv。
virtualenv --system-site-packages .venv
.venv/bin/python -m pip install -e . --no-build-isolation --no-deps
.venv/bin/python -c 'import numpy, yaml, pytest; print(numpy.__version__, yaml.__version__, pytest.__version__)'
```

工作区 `.venv` 已完成本地可编辑安装，入口命令为 `.venv/bin/ner-workbench`。若直接使用源码目录，给 Python 加上 `PYTHONPATH=src` 即可；`pyproject.toml` 已为 pytest 配置同样的源码路径。

## 快速运行

```bash
# 1. 生成固定种子样本目录
PYTHONPATH=src python3 -m ner_workbench generate --out samples --seed 20261121 --samples 32

# 2. 并发 1 与并发 4 各完整运行一次
PYTHONPATH=src python3 -m ner_workbench run --samples samples --out runs/workers1 --workers 1
PYTHONPATH=src python3 -m ner_workbench run --samples samples --out runs/workers4 --workers 4

# 3. 比较确定性投影
PYTHONPATH=src python3 -m ner_workbench compare \
  runs/workers1/report.json runs/workers4/report.json \
  --out runs/comparison-workers1-workers4.json

# 4. 查看面板
PYTHONPATH=src python3 -m ner_workbench serve --directory runs/workers4 --port 8765
# 浏览器打开 http://127.0.0.1:8765/dashboard.html
```

`run` 的常用诊断参数：

- `--resume`：从 `OUTPUT/snapshots/CURRENT` 续跑；配置哈希、参数版本、模型哈希、语料哈希或预期全局版本不一致时输出 `E_SNAPSHOT_STALE` 并退出 2。
- `--expected-version N`：额外要求快照全局版本等于 N。
- `--delay-ms N`：每个样本任务开始前休眠 N 毫秒，仅用于复现信号中断。
- `--stop-after N`：完成 N 个样本任务后触发优雅停止，便于自动化续跑演练。

## 输入契约

样本目录包含四个输入文件，均由生成器产出：

- `corpus.jsonl`：逐行 JSON。键为 `sample_id`、空格分隔的 `chars`、空格分隔的 `labels`、`entity_types` 与可审计的 `entities`。`labels` 为 BIESO，类型为 `人物/地点/机构/时间`，例如 `B-人物 I-人物 E-人物`。实体 `start/end` 为闭区间字索引。
- `model_params.json`：模型版本、标签枚举、`feature_dim`、`hash_buckets`、哈希盐、字/位置/词表命中长度嵌入、按 `label,bucket,dimension` 排列的发射权重、`labels × labels` 转移权重与起始权重。所有权重先量化成整数再参与计算。
- `inference.yaml`：处理单元（`sentence` 或 `sliding_window`）、窗口大小/步长、字级批大小、最大迭代步数、定点小数位宽、量化档位、线程池上限、全局字级预算、单样本配额、缓存块预算、字块大小、交互延迟预算、快照提交间隔与种子。
- `injections.json`：`{"injections": [{injection_id, sample_id, stage, kind}, ...]}`。支持的注入：
  - `nonfinite_score` → `emission`：把指定样本首个发射打分替换为非有限值，在发射阶段抛出 `E_NONFINITE_SCORE`。
  - `truncated_labels` → `decode`：向解码器送入截断的标签长度，在解码阶段抛出 `E_LABEL_LENGTH_MISMATCH`。
  - `hash_bucket_mismatch` → `emission`：运行时破坏哈希桶数与发射张量形状的一致性，抛出 `E_PARAM_SHAPE_MISMATCH`。
  - `quota_lt_min_chars` → `schedule`：把有效单样本配额压到小于最小样本长度，记 `E_SAMPLE_QUOTA_EXCEEDED`。

生成器默认产出 32 个样本、741 个字符、全局预算 489、单样本配额 30、缓存预算 5 块、字块大小 4。该规模必然覆盖全局预算耗尽、LRU 淘汰与淘汰后重算。

## 校验与错误码

加载层逐行处理语料；合法样本继续执行，失败记录进入稳定排序的失败清单。主要错误码：

- `E_CORPUS_INVALID`：空句、重复 ID、非法 JSON、实体区间越界/重叠、实体文本或标注结构不一致等。
- `E_LABEL_LENGTH_MISMATCH`：字序列与标签序列长度不符；也用于解码阶段注入的截断标签。
- `E_LABEL_UNKNOWN`：BIESO 符号或实体类型不在枚举内。
- `E_PARAM_SHAPE_MISMATCH`：特征维度、哈希桶、嵌入长度、发射张量或转移矩阵形状不符；也用于运行时桶数注入。
- `E_SAMPLE_QUOTA_EXCEEDED`：样本所需字数超过有效单样本配额。
- `E_GLOBAL_BUDGET_EXHAUSTED`：全局剩余字符预算不足以完整容纳下一个稳定 ID 样本。
- `E_NONFINITE_SCORE`：发射打分出现 NaN/Inf。
- `E_SNAPSHOT_STALE`：快照格式版本、预期全局版本、配置哈希、参数版本/哈希或语料哈希不一致。

## 确定性算法契约

### 定点整数

配置的 `fixed_decimals=d` 定义十进制缩放 `S=10^d`，`quantization_level=q` 定义量化步长 `2^q` 个定点单位。实数权重先乘 S，按 half-to-even 取整，再吸附到最近的 `2^q` 倍数（同距取偶）。发射点积的每次乘加后立即按 S 做整数 half-to-even 重标定，报告保留十进制位宽供审计。

### 特征与哈希

每个字符位置固定构造五类特征：当前字 `U:`、前字 `L:`、后字 `R:`、四档相对位置 `P:`、受控词表最长命中长度 `V:`。特征键使用 BLAKE2b + 固定盐映射到桶；嵌入向量与对应桶的发射权重做定点点积。报告输出占用桶、冲突桶、去重特征键数与冲突事件数。两次运行的逐字发射分数按整数逐位比较。

### 对数域维特比

起始权重加首位置发射分作为初始前向分数。之后按位置升序递推；每个当前位置按前驱标签编码升序检查。严格大于才替换，得分相等保留较小标签编码，因此并列决断稳定。回溯后再次用所选标签、发射分和转移分独立复算路径总分，结果写入 `path_score_verified`。报告同时给出每个位置的完整候选标签分数、选定路径的回退指针。

### 实体修复

解析 BIESO 时记录每次非法结构及受影响位置：

- `I` 前无 `B`：提升为 `B`。
- `E` 前无 `B`：提升为单字实体。
- `B` 被 `B/O/S` 打断：在区间末隐式补 `E`。
- `I/E` 的类型与已开启片段冲突：保留冲突候选类型并记录 `TYPE_JUMP_ON_I/E`。
- 句末未闭合：在末位置隐式补 `E`。
- 片段存在多个候选类型：按类型字符串字典序选择，并记录 `TYPE_TIE_LEXICOGRAPHIC`。

输出实体以 `start/end/type/candidate_types/text/sample_id` 排序，`end` 为闭区间。

### 双重预算调度

有效处理样本按 `sample_id` 全局排序。调度器按此顺序做一次性预算账本规划：先扣除单样本配额约束，再从全局剩余字符预算中分配；每条扣减记录包含序号、请求字数、有效配额、分配字数、扣减前后余额、状态码和注入编号。被拒绝样本不再进入线程池，已批准样本才提交。

字符特征块以 `char_block_size` 为边界划分。每个样本批次在 `emission` 与 `decode` 阶段都会访问字块。所有样本/批次/阶段/字块事件由 `sample_id | bNNNNNN | stage | block` 稳定键排序，日志重放 LRU：新块超预算时淘汰最近最少使用块，访问序号并列时淘汰字典序更小的字块键；被淘汰块再次访问记入重算清单，并比较重算 SHA-256 与首次计算是否一致。

线程池只负责可并行的样本流水线；预算与缓存驻留决策不依赖线程完成顺序，因此并发 1、4 或其他上限的确定性投影一致。

## 快照与恢复

结果目录约定：

```text
runs/<run-label>/
  report.json                 # 完整报告（含诊断耗时）
  deterministic.json          # 仅确定性投影
  dashboard.html              # 自包含离线可视化面板
  snapshots/
    CURRENT -> versions/vNNNNNNNN
    versions/vNNNNNNNN/
      manifest.json
      load.json
      emission.json
      decode.json
      merge.json
      runtime.json
    .staging-*                # 提交过程临时目录，与正式目录同卷
```

提交时先把所有阶段产物写入 `snapshots/.staging-*`，逐文件 fsync，再将整个目录重命名为 `versions/vNNNNNNNN`，最后通过同卷 `os.replace` 原子替换相对 `CURRENT` 符号链接。因此任何时刻 `CURRENT` 只指向完整版本；强制终止最多留下不可达的临时目录或新版本目录，不会暴露半写快照。

`manifest.json` 记录全局版本、快照格式版本、配置哈希、模型版本/哈希、语料哈希、已完成样本前缀、已消耗预算、各阶段版本、产物哈希、恢复点和状态。快照只提交按 `sample_id` 排序的连续已完成前缀，从而保证续跑后的账本、缓存事件序列与完整单线程运行逐字段相同。

SIGINT/SIGTERM 处理器只设置停止事件：不再派发新样本，等待当前已提交到线程池的样本完成，随后提交快照并以退出码 0 结束。SIGKILL 则由下一次 `--resume` 从最后被原子替换的快照恢复，并在报告 `resume.recovery_point_sample_id` 中给出恢复点。

## 报告字段

`report.json` 的 `deterministic` 区段用于一致性核对，包含：

- 逐样本字符、预测标签、实体片段、路径总分、逐位置候选分、修复记录、任务/批次/缓存稳定键与预算账本项。
- 稳定排序的失败清单与注入命中清单。
- 全局预算账本、LRU 淘汰序列、重算清单与校验一致性。
- 哈希冲突统计、配置/语料/参数/注入哈希、快照完成前缀与摘要。

`diagnostics` 区段包含总墙钟、逐样本各阶段毫秒耗时、汇总阶段耗时和交互延迟超预算记录。耗时天然受机器负载影响，因此不参与确定性比较。`compare` 只比较 `deterministic`；本机并发 1 与并发 4 的 SHA-256 摘要为
`c5c3ad595c99304511dfa4247045d1a5866dab88e54675636035e352ec66c9fa`（以最终重跑记录为准）。

## 测试与已执行验证

```bash
PYTHONPATH=src pytest -q
```

自动化用例覆盖：加载错误码与继续处理、参数形状错误、定点发射逐位确定性、真实哈希冲突、维特比并列决断与路径复算、BIESO 修复及类型并列、预算配额、LRU 淘汰/重算、并发一致性、四类注入、优雅停止后续跑、快照原子符号链接与陈旧拒绝。

最终验证结果见 `artifacts/validation_summary.json` 与 `VALIDATION.md`。本机记录：

- 13 个 pytest 用例全部通过。
- 样本 32 个、字符 741、全局预算 489；单线程与并发 4 均 17 个成功实体结果、15 个终态失败、148 个实体片段、216 次淘汰、103 次淘汰后重算、327 次哈希冲突，确定性摘要完全一致。
- 第 12 个样本后优雅停止：已完成前缀 12、消耗 286 字符预算；续跑后与完整单线程结果零差异。
- SIGTERM：进程退出码 0；SIGKILL：退出码 137，恢复版本 4、恢复点 `SMP-0012`；两者续跑后均与完整结果零差异。
- 以并发 2 对并发 1 快照执行陈旧续跑：退出码 2，错误码 `E_SNAPSHOT_STALE`。

## 主要代码

- `src/ner_workbench/generator.py`：固定种子样本与参数包生成。
- `src/ner_workbench/loader.py` / `params.py`：语料与模型严格加载校验、量化。
- `src/ner_workbench/features.py` / `decoder.py` / `entities.py`：特征哈希、维特比、片段修复。
- `src/ner_workbench/scheduling.py`：预算账本、稳定批次键、确定性 LRU 重放。
- `src/ner_workbench/snapshot.py`：同卷分阶段原子快照。
- `src/ner_workbench/runner.py`：线程池流水线、注入隔离、恢复与报告编排。
- `src/ner_workbench/reporting.py` / `dashboard.py`：审计报告与离线 HTML 面板。
- `tests/`：13 个自动化验证用例。
