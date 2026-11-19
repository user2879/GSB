# 中文依存句法分析结果双版本编辑应用 — 技术说明

## 1. 目标与范围

应用面向中文依存句法解析结果，解决两种工程交换版本之间的互操作与版本兼容问题：

- **v1**：固定宽度字段拼接的纯文本，1 起始词序号，头词缺失用填充串 `*NA*`，
  千分比整数得分（0..1000），关系用短码（如 `SBV`、`HED`）。
- **v2**：JSON，浮点偏移头词（默认 0 起始，可配置），命名关系（如 `nsubj`、`root`），
  0..1 浮点得分，带精度标签，缺失头词为显式 `null`。

统一中间表示（IR）固定为 **1 起始词序号、头词 0 表示根缺失、命名关系、0..1 浮点得分**，
并保留来源版本与原始字段。

## 2. 环境

- Python 3.11.0rc1（`python3` / `python3.11` 均指向同一解释器）。
- NumPy 2.2.6（cp311 wheel，用户级安装）。
- pytest 9.2.0（仅测试用）。
- 不使用 Docker / 容器；所有样本本地确定性生成，不访问外部目标。

依赖见 `requirements.txt`。

## 3. 目录结构

```
depedit/
  errors.py        业务异常码（E_HEAD_STRUCTURE / E_SNAPSHOT_STALE 等 9 类）
  models.py        统一中间表示：Word / Arc / Sentence / Corpus
  config.py        定宽字段、编号起点、并发上限、注入配置
  fingerprints.py  SHA-256 分层指纹与稳定排序键
  validation.py    头词越界、环、单根校验
  codecs.py        v1 定宽与 v2 JSON 的解析 / 序列化、原子写盘
  relations.py     v1 短码 <-> v2 命名关系映射
  edits.py         四类编辑操作（失败不写入）
  injection.py     六类故障注入引擎（注入编号可追溯）
  concurrency.py   线程池执行器 + 稳定排序键重排
  cache.py         增量失效与精确重算范围
  snapshot.py      临时目录 + 原子整体替换快照、恢复一致性
  diagnostics.py   跨版本往返一致性诊断
  analysis.py      NumPy 模型权重统计与诊断聚合
  replay.py        编辑序列回放
  samplegen.py     固定种子样本生成
  pipeline.py      主编排
  cli.py           命令行入口
config/            并发、注入、编辑序列、应用配置样例
tests/             22 个 pytest 用例
scripts/           端到端验证脚本
```

## 4. 数据契约

### 4.1 v1 定宽字段（字符宽度，左对齐补空格）

| 字段 | sent_id | word_idx | form | pos | head | relation | score |
|---|---|---|---|---|---|---|---|
| 宽度 | 10 | 4 | 12 | 6 | 4 | 16 | 6 |

- `head` 为 `*NA*` 时还原为根缺失（IR 头词 0）。
- `score` 为千分比整数，合法区间 0..1000，超界抛 `E_SCORE_RANGE`。
- 关系短码不在映射表内抛 `E_RELATION_UNKNOWN`。
- 行无法按宽度切出全部字段抛 `E_FIXED_WIDTH_FIELD`。

### 4.2 v2 JSON

```json
{
  "S001": {
    "words": [{"form": "提升", "pos": "v"}],
    "arcs": [
      {"word_idx": 1, "head": 3.0, "relation": "nsubj",
       "score": 0.898, "precision": "permille"}
    ]
  }
}
```

- `head: null` 表示根缺失；浮点偏移按 `v2_head_origin` 折算为统一 1 起始
  （0 起始时 `head + 1`，1 起始时直接取整）。
- 得分为非有限数（NaN/Inf）抛 `E_SCORE_NON_FINITE`。
- 缺失头词写入旁路元数据 `parse_v2_sidecar.json`。

## 5. 编辑与结构校验

四类编辑（`depedit/edits.py`）：`set_head`、`replace_relation`、`shift_heads`、`mark_reparse`。

- 每次编辑在深拷贝上试写，校验通过后才提交；返回受影响词与受影响依存弧清单。
- 校验规则：头词序号不越界、无环、每句恰好一个根；违反抛 `E_HEAD_STRUCTURE`，原状态不变。
- `shift_heads` 跳过根缺失弧（头词 0），保证根不被平移破坏。

## 6. 双向转换与信息保留

- **v1 -> v2（升级）**：千分比 `/1000` 精确换算并生成精度标签；短码经映射表转命名关系；
  缺失头词转显式 `null` 并写入旁路元数据。
- **v2 -> v1（降级）**：浮点得分 `round(score*1000)` 还原千分比；命名关系折回短码，
  映射不上时保留原名（超列宽安全截断，原名完整写入降级清单）；
  降级损失清单记录三类信息：
  - `truncated_residual`：四舍五入小数残差与精度标签；
  - `unmapped_relation`：无法折回的关系原名与回退字段；
  - `origin_adjustment`：编号起点折算（仅在 v2 起点与统一起点不同且非根弧时记录）。

## 7. 指纹、缓存与精确失效

缓存键由 **词序列指纹 + 模型参数指纹 + 映射表指纹 + 版本号** 经 SHA-256 派生，
分句子键、依存弧键、序列化产物键三层；每个弧记录其引用词集合。

- 句子样本替换：仅该句全部弧及其下游产物失效。
- 映射表增删关系名：仅引用受影响关系的弧失效。
- 模型参数变更：模型指纹变化，全部句子失效。
- 词级变更：仅引用受影响词集合的弧失效。
- 未失效句子出现在 `reusable_sentences`，复用已持久化产物，不重新解析。

## 8. 快照与恢复

`SnapshotStore.commit` 先写同级临时目录，再 `os.rename` 原子整体替换目标目录；
旧目录先移到临时备份，提交成功后清理。清单 `manifest.json` 记录：
全局版本号、模型哈希、映射表哈希、统一表示状态指纹、降级损失指纹、各句各版本产物指纹。

恢复时比对清单版本号与当前状态/模型/映射指纹，任一不一致抛 `E_SNAPSHOT_STALE`，
旧快照保留不覆盖。

## 9. 异常注入与失败隔离

注入配置（`config/injections.json`）覆盖六类真实故障，命中时把注入编号写入失败清单：

| 注入故障 | 业务码 | 默认命中句 |
|---|---|---|
| fixed_width_missing | E_FIXED_WIDTH_FIELD | S006 |
| head_out_of_range | E_HEAD_OUT_OF_RANGE | S007 |
| score_over_1000 | E_SCORE_RANGE | S008 |
| relation_not_in_map | E_RELATION_UNKNOWN | S005 |
| v2_non_finite_score | E_SCORE_NON_FINITE | S004（转换阶段） |
| cycle_dependency | E_CYCLE_DETECTED | S003（编辑阶段） |

单句在解析 / 编辑 / 转换任一阶段抛异常，仅记录句 ID、阶段、业务码、异常类型与注入编号，
其余句子继续处理。数据破坏型注入通过线程局部上下文把注入编号传递给自然抛出的异常。

## 10. 并发一致性

`concurrency.py` 用 `ThreadPoolExecutor` 执行句子级解析、指纹构建与序列化，
最大并发数取配置并受 `workers_hard_limit` 钳制。每个任务携带
`stable_task_key = sent_id + 阶段名`，结果在收集前按该键稳定排序。
验证脚本在并发度 1/2/4/8/16 下比对句子顺序、弧顺序、失效清单与失败清单，逐字段完全一致。

## 11. 诊断与回放

- `diagnostics.roundtrip_compare`：对同一 IR 分别做 v1、v2 序列化-反序列化往返，
  输出逐句头词偏差直方图、最大得分偏差、关系还原失败清单。
- `replay.apply_edit_sequence`：从空白工作区重放编辑序列，最终 IR、降级损失与诊断摘要
  与逐步编辑结果一致（状态指纹比对）。报告经临时文件 + 原子替换写盘。

## 12. 操作手册

```bash
# 生成确定性样本
python3 -m depedit.cli --data-dir data/sample --out-dir out generate
# 编码与哈希检查（含 NumPy 模型统计）
python3 -m depedit.cli --data-dir data/sample --out-dir out fingerprints
# 解析（默认无注入；--injections 显式启用故障注入）
python3 -m depedit.cli --data-dir data/sample --out-dir out parse --source v1
python3 -m depedit.cli --data-dir data/sample --out-dir out --injections config/injections.json parse --source v2
# 双向转换
python3 -m depedit.cli --data-dir data/sample --out-dir out convert --direction up
python3 -m depedit.cli --data-dir data/sample --out-dir out convert --direction down
# 编辑与回放
python3 -m depedit.cli --data-dir data/sample --out-dir out edit
python3 -m depedit.cli --data-dir data/sample --out-dir out replay
# 测试与端到端验证
python3 -m pytest
python3 scripts/run_verification.py
```

## 13. 验证结果（实际执行）

- `python3 -m pytest`：22 passed。
- `scripts/run_verification.py`：13/13 检查通过，报告落盘
  `out/verification_report.json`（原子替换）。
- 六类注入全部命中并映射到对应业务码，注入编号齐全；失败句子被隔离，其余句子正常产出。
- 干净样本往返头词偏差直方图仅含 `0`，最大得分偏差 0.0，关系还原无失败。
- 降级样本（构造的非千分比得分与未映射关系）正确产生三类降级损失并保留原名。
