# 中文新闻摘要与重写组件 — 技术说明

## 1. 目标与范围

面向本地确定性样本的领域组件，完成五个阶段：句子切分、定点句重要度打分、覆盖约束下
摘要句选择、模板化风格重写、摘要汇总。所有产物由内容指纹派生缓存键，支持句级精确失效、
阶段化原子快照、中断恢复、预算分批并发调度，并保证增量运行与空白全量重建逐字段一致。

仅依赖 Python 3.11 标准库与 `numpy==1.26.4`（float64 参考分）。YAML 使用内置的确定性
子集解析器 `news_pipeline/yaml_lite.py`，不引入第三方 YAML 依赖。

## 2. 目录结构

- `news_pipeline/errors.py`：稳定业务码（E_INPUT_INVALID / E_REWRITE_RULE_INVALID /
  E_SNAPSHOT_STALE / E_SENTENCE_BUDGET_PENDING）。
- `news_pipeline/fixedpoint.py`：Qm.f 定点格式与确定性量化（四舍五入远离零）。
- `news_pipeline/hashing.py`：规范 JSON 与 sha256 内容指纹。
- `news_pipeline/yaml_lite.py`：受控 YAML 子集解析器。
- `news_pipeline/config.py`：配置与同义替换表加载、严格校验、业务/运行时指纹拆分。
- `news_pipeline/documents.py`：JSONL 新闻文档逐行校验，错误进失败清单且不中断其余文档。
- `news_pipeline/segmentation.py`：中文标点切句、原文偏移、受控词表最长匹配分词。
- `news_pipeline/scoring.py`：固定运算顺序的 float64 参考分与定点分、分项贡献。
- `news_pipeline/selection.py`：覆盖约束选择、定点/float 双路排序与边界句决断。
- `news_pipeline/rewrite.py`：模板替换、键长降序+条目序、重叠抑制与替换记录。
- `news_pipeline/cache.py`：内容寻址阶段缓存与复用/失效日志。
- `news_pipeline/snapshot.py`：临时目录写入 + 整体替换提交 + 清单版本比对。
- `news_pipeline/edits.py`：编辑命令流的确定性应用（文档/配置/替换表）。
- `news_pipeline/pipeline.py`：主编排、预算分批、线程池并发、诊断报告。
- `news_pipeline/sample_generator.py`：固定种子 `20261125` 的样本生成。
- `tests/test_pipeline.py`：14 个 unittest 用例。

## 3. 确定性设计

1. 切句按段落顺序与中文句末标点 `。！？!?` 切分，记录全局起止偏移与段落归属；句 ID
   采用 `doc_id#s000` 稳定编号。
2. 分词对受控词表按“词长降序、词形升序”做最长匹配，未登录字逐字成词，不依赖集合遍历。
3. 打分固定运算顺序：先 `词频权重*词频归一 + 实体权重*实体覆盖`，再乘位置因子；
   中间值全部为 numpy float64，最后按 Q 格式量化为 32 位整数。
4. 选择排序键固定为“得分降序 → 起始偏移升序 → 全局序号升序”，串行与并发结果同序。
5. 所有缓存键均为规范 JSON（dict 键排序）的 sha256，杜绝哈希种子/插入顺序影响。
6. 配置指纹拆分：业务指纹（进入缓存键，权重/覆盖/位宽/档位/模板/替换表版本）与运行时
   指纹（workers、预算、快照间隔）分离，调整并发参数不会使业务缓存失效。

## 4. 定点量化与边界决断

- 档位 `coarse/standard/fine/precision` 对应 4/8/12/16 个小数位，量化为
  `floor(abs(x*scale)+0.5)` 并恢复符号。
- 选择器分别用定点分与 float64 参考分各跑一次完整选择；两路入选集合的对称差即边界句，
  输出两者分值、差值与 `fixed_includes_float_excludes` 等判定，边界句绝不静默丢弃。
- 位置因子按“段落内相对位置”计算，使未改动段落的句子缓存键只绑定自身文本、实体与配置，
  前面段落增删不会让无关句子重新打分。

## 5. 句级与替换条目级精确失效

- 切分键派生自文档内容指纹；打分键派生自句文本、段落 PID、段内序号、实体集合、词表与
  业务配置指纹；重写键派生自句文本、模板 ID、命中的替换条目（含条目序）与替换表版本。
- 修改单段落：仅该段落句子的切分/打分/下游选择/重写失效，其它段落直接命中缓存。
- 修改单条替换规则：仅命中该条目的句子重写失效，未命中句子复用重写产物。
- 替换冲突：候选匹配按源串长度降序、再按规则条目序、再按位置排序，重叠区段首个匹配
- 胜出，其余写入 `suppressed_rules`；引用未声明模板或目标为空的规则在加载期输出
  E_REWRITE_RULE_INVALID 并保留原句。

## 6. 快照与中断恢复

- 五个阶段各自先写入同级临时目录，再通过 `rename` 整体替换；清单记录全局版本号、
  配置指纹、替换表版本/指纹、已完成文档游标与各阶段产物版本。
- 提交时合并历史阶段产物，保证任意时刻快照都包含游标内的完整文档集合。
- 启动时按配置指纹与替换表版本比对，不符则输出 E_SNAPSHOT_STALE 并全量重建，
  不做逐文件覆盖。
- SIGINT/SIGTERM 或显式停止回调只在“当前文档/当前批次完成并提交后”生效。

## 7. 预算与并发

- 调度器按文档 ID 稳定排序，以真实切句数消耗单批句级点数预算；超预算文档保留待办并
  计 E_SENTENCE_BUDGET_PENDING，超大文档独占下一批避免饿死，下一批优先处理。
- 线程池大小取 `min(配置 workers, 批内文档数)`，并用 BoundedSemaphore 硬性限制同时
  执行的文档任务数；结果按文档 ID 顺序回收，保证并发与串行逐字段一致。

## 8. 诊断报告

每次运行输出 JSON 报告，含每文档选中句数、实际实体覆盖率、边界句数、触界标记
（最大句数触界/覆盖下限触界）、逐级误差与首次超差环节、业务码计数、缓存复用统计、
预算待办、快照过期事件与失败清单。

## 9. 运行方式

```bash
python -m news_pipeline.cli generate-samples --out samples/data
python -m news_pipeline.cli run --samples samples/data \
    --snapshot runtime/snap --report runtime/report.json
python -m news_pipeline.cli run --samples samples/data \
    --snapshot runtime/snap --report runtime/incr.json \
    --edits samples/data/edit_stream.jsonl
python -m news_pipeline.cli run --samples samples/data \
    --snapshot runtime/snap --report runtime/partial.json \
    --stop-after-docs 2
python -m unittest tests.test_pipeline -v
```
