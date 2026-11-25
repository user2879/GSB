# 语音命令词注册诊断组件

本仓库实现一个可复现、可增量的本地语音命令词注册诊断流水线。输入是由固定整数种子确定性合成的 16 kHz / 单声道 / PCM16 样本，输出逐模板声学向量、定点模板、混淆对、增量重聚类、逐片段结论、混淆仲裁和可复核混淆矩阵。

样本不依赖网络、真实录音或预置数据集。仓库内的 `sample/` 已生成一份 6 命令词、18 模板、28 片段（含 4 个负样本）的本地样本；也可以用 `voice-diag generate` 从任意固定种子重建。

## 环境

已在 Linux + Python 3.11 + NumPy 2.4.6 + pytest 8.4.2 下验证。

```bash
python3 -m venv --system-site-packages .venv  # 若系统无 ensurepip，可使用 virtualenv
virtualenv --clear .venv
. .venv/bin/activate
pip install -e '.[test]'
```

当前容器实际使用 `virtualenv` 创建环境：

```bash
virtualenv --clear .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## 快速运行

```bash
# 1. 确定性生成本地语料
.venv/bin/voice-diag generate --out sample --seed 20261126

# 2. 首次运行（作为增量基线）
.venv/bin/voice-diag run \
  --config sample/config.json \
  --state sample-state \
  --result sample/result.json \
  --report sample/report.json \
  --mode auto

# 3. 再次运行，未变更产物从内容缓存复用
.venv/bin/voice-diag run \
  --config sample/config.json \
  --state sample-state \
  --result sample/result-2.json \
  --report sample/report-2.json \
  --mode auto

# 4. 完整编辑序列验收：新增词、替换/删除模板、退化、量化与阈值切换
.venv/bin/voice-diag scenario --out /tmp/voice-diag-scenario --seed 20261126

# 5. 规范结果逐字段比较
.venv/bin/voice-diag compare \
  --incremental /tmp/voice-diag-scenario/result-incremental.json \
  --full /tmp/voice-diag-scenario/result-full.json
```

`result.json` 是只由最终状态决定的规范结果，可用于增量/全量逐字段比较。`report.json` 记录本次运行的模式、缓存命中/失效事件、线程峰值和快照指针，包含运行过程信息，不参与规范结果比较。

## 代码结构

```text
voice_diag/
  config.py       配置、枚举、行号/字段校验
  wavio.py        RIFF 分块解析、PCM16/非有限采样校验
  corpus.py       片段扫描、真值与混淆对标注校验
  synth.py        固定种子本地合成器
  features.py     有序分帧、元音/擦音统计、配置派生投影
  quantization.py 定点整数、逐维步长/残差/截断记录
  similarity.py   整数域余弦相似度与候选材料化
  clustering.py   代表模板、命令词簇、跨词合并
  recognition.py  逐片段识别、拒识、边界、重叠候选
  arbitration.py  混淆对定点分歧分与仲裁
  matrix.py       混淆矩阵、计数不变量、高一位宽参考
  cache.py        内容寻址缓存与复用审计
  scheduler.py    稳定顺序预算调度与有界线程池
  snapshot.py     临时目录写阶段快照，原子切换 current 指针
  engine.py       全流水线编排
  scenario.py     编辑序列验收与增量/全量比较
  cli.py          CLI
tests/            27 项 pytest 验收
sample/           已生成的本地样本、配置与规范结果
```

## 配置契约

配置文件是 JSON。关键字段：

- `version`：快照续跑/版本判定字段。
- `paths`：模板目录、片段目录、JSONL 标注文件，相对于配置文件目录。
- `audio`：必须为 `16000`、`1`、`16`。
- `features.frame_length` / `frame_shift`：正整数。
- `features.dimension`：正偶数。投影矩阵从配置指纹、投影种子和种子版本派生。
- `features.cepstral_coefficients`：倒谱统计维数。
- `quantization.profile`：`q8/q12/q16/q20/q24` 之一，且必须在 `allowed_profiles` 中。
- `quantization.fractional_bits`：`2/4/6/8/10/12` 之一，且必须在 `allowed_fractional_bits` 中。
- `quantization.reference_profile`：边界格子的高一位宽参考档位；若未高于当前档位，加载时按实现枚举自动提升到下一宽度。
- `thresholds.reject_permille`：整数拒识阈值，范围 0..1000。
- `thresholds.confusion_candidate_permille`：整数混淆候选阈值。
- `thresholds.cluster_merge_permille`：整数跨命令词簇合并阈值。
- `registration.template_min`：每词注册模板下限。
- `registration.max_workers`：线程池最大并发数，1..64。
- `registration.comparison_budget`：全局模板比对点数预算。
- `registration.overlap_candidates`：逐片段保留的最近代表模板候选数。
- `registration.representative_rule`：`composite/lexicographic/most_frames`。

标注/命令流文件为 JSONL，每行字段为：

```json
{"clip_id":"pos_lin_01","path":"clips/pos_lin_01.wav","true_command":"LIN","confusion_with":"LING","kind":"positive"}
```

`path` 相对于语料工作区；`true_command` 必须属于配置中的命令词或为 `NEGATIVE`；`confusion_with` 为命令词或 `null`；`kind` 为 `positive/negative`。缺失标注、重复路径、未知词或非法行都会带 JSONL 行号进入失败清单。

配置错误返回 `E_CONFIG`，并附带源文件、行号（可定位时）、字段名。音频错误按文件隔离，返回 `E_AUDIO_FORMAT`；IEEE-float 非有限样本返回 `E_AUDIO_NONFINITE`，不会中断其余文件。

## 核心规则

### 声学与定点

- 帧仅来自该模板自己的 PCM 数据，按帧序号升序聚合。
- 每段统计能量、过零率、倒谱标准差、4 个频带能量与 3 个共振峰位置；元音段、擦音段分别记录有效帧数。
- 投影矩阵不使用运行时全局随机状态；每个输出维度记录 `projection_seed_version` 和 `valid_frames`。
- 相似度完全在整数域计算，按维度升序累加点积，并用 `math.isqrt` 归一化到 0..1000。
- 相似度等于候选阈值时保留候选并设置 `equality_boundary=true`，模板对始终按模板键字典序规范化。

### 代表模板与簇

- 每个命令词只选一个代表模板。综合分为该词内模板两两相似度的整数均值；平局按规则和模板键字典序决断。
- 命令词按标识升序遍历，与已有簇代表比较，取首个达到阈值的簇；合并保留较小命令词标识并重写成员归属。
- 单模板命令词输出 `degenerate=true`、`skip_cross_word_merge=true`，并免于比较/合并。

### 识别与矩阵

- 逐片段按 `(-相似度, 命令词标识)` 排序；首位低于拒识阈值则输出 `<REJECTED>`、原因和次位。
- 首位与次位差值不超过量化半步时标记混淆边界，并记录 `top_ahead/tie` 方向。
- 每个片段保留最近 `overlap_candidates` 个代表模板候选及各自相似度。
- 混淆矩阵覆盖所有真值行与所有命令词加 `<REJECTED>` 列；每个单元维护支持、边界、拒识片段计数及片段清单。
- 边界单元额外保存高一位宽档位的参考结论、参考相似度、差值方向和结论是否改变；高参考沿用低位宽代表模板拓扑，并在输出中显式标注。
- 矩阵输出 `support + unavailable = corpus`、各计数等于逐片段清单等不变量。

### 缓存与增量

缓存键由音频内容指纹、配置/特征指纹、投影版本、量化档位/小数位宽、阈值、模板整数指纹和比较对共同派生。缓存事件记录 `reuse/miss/computed` 及引用对象。

只重算内容键真正变化的阶段：

- 替换一个模板只使该模板声学、定点、相关模板对及其下游失效。
- 量化档位/小数位宽变化复用声学向量，但使全部定点与比较失效。
- 拒识阈值变化复用声学、定点、相似度和簇，只使逐片段结论、参考结论及矩阵单元失效。
- 矩阵阶段整体键变化时，未变化单元格仍从单元缓存复用。
- 代表模板、簇轨迹、簇归属和混淆仲裁对均有独立缓存键。

### 预算与快照

- 每个比较任务的 `stable_sort_key` 由阶段、命令词、模板/片段标识派生；调度器按稳定键排序并顺序消耗预算。
- 一个任务所需点数等于声学维数。逻辑任务即使命中缓存也参与预算选择，因此缓存状态不会改变增量/全量结论。
- 超预算任务保留在 `pending_queue`，结果和失败清单均带 `E_BUDGET_PENDING`。
- 每个阶段先写入快照临时目录，成功后整体重命名，并原子替换 `current.json` 指针。
- 快照含各阶段产物、待办队列、已完成命令词游标、配置版本/结构哈希/运行哈希。结构配置变化会清理不兼容内容缓存；同结构版本修改走内容键增量。

## 实际验证结果

环境：Python 3.11、NumPy 2.4.6、pytest 8.4.2。

```text
.venv/bin/python -m pytest -q
...........................                                              [100%]
27 passed
```

`sample/result.json` 的基线结果：

- 18 个注册模板、28 个片段、153 个模板对。
- 3 个命令词簇：`LIN+LING`、`MA+NA`、`SASA+SHASHA`。
- 81 个混淆候选对；45 个仲裁为 `merge_to_consensus`，36 个 `retain_both`。
- 28 条逐片段结论，4 个负样本被拒识，16 个混淆边界片段。
- 混淆矩阵 49 个单元格，所有计数不变量通过；16 个边界片段均有高一位宽参考。
- 首轮预算消耗 2364 点，无待办。

完整编辑序列实测：

```text
canonical_results_exactly_equal: true
incremental sha256 = full sha256 = a16bd1759812cff20a8c059341ce5cd0b08f0ab891da0f5fec64a764f72d904b
degenerate_words: ["NA"]
incremental reuse hits: 56
budget pending: 0
matrix consistent: true
```

低预算测试把预算降为 100 点、并发降为 2：增量与全量规范结果仍逐字段一致，产生 173 个稳定排序的待办，`E_BUDGET_PENDING` 同时进入失败清单，实测线程峰值不超过 2。

故障隔离测试覆盖：截断 RIFF/WAV、NaN float32 WAV、模板数低于下限、帧移为 0、特征维度为奇数、量化档位越界、拒识阈值非整数、重复命令词/模板标识。正常文件与片段在故障存在时继续处理。
