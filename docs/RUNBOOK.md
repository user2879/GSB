# 验收运行手册

## 环境准备

```bash
python --version            # Python 3.11.x
python -m pip install numpy==1.26.4
```

离线环境可用本地 wheel：`python -m pip install --no-index --find-links <wheel目录> numpy==1.26.4`。

## 一次性生成确定性样本

```bash
python -m news_pipeline.cli generate-samples --out samples/data
```

产物（固定种子 `20261125`，无网络依赖）：

- `documents.jsonl`：6 个合法文档 + 2 个非法文档（空段落、损坏三元组）。
- `config.yaml`：摘要/重写/定点/运行时参数。
- `rewrite_table.yaml`：formal/brief 模板，含 2 条非法规则（未声明模板、空目标）。
- `edit_stream.jsonl`：14 条编辑，覆盖文档替换、段落增删改、摘要长度、覆盖下限、
  权重、位置衰减、风格模板、替换规则增删改、量化档位切换与替换表版本升级。
- `sample_manifest.json`：种子与词表清单。

## 验收命令

```bash
# 全量运行（空白快照目录，组件自建临时/快照目录）
python -m news_pipeline.cli run --samples samples/data \
  --snapshot runtime/full --report runtime/full.json

# 编辑流增量运行（同一快照目录）
python -m news_pipeline.cli run --samples samples/data \
  --snapshot runtime/full --report runtime/incremental.json \
  --edits samples/data/edit_stream.jsonl

# 中断后恢复：先处理 2 个文档再提交退出，随后续跑
python -m news_pipeline.cli run --samples samples/data \
  --snapshot runtime/recover --report runtime/partial.json --stop-after-docs 2
python -m news_pipeline.cli run --samples samples/data \
  --snapshot runtime/recover --report runtime/resume.json

# 完整自动化验证（14 个用例）
python -m unittest tests.test_pipeline -v
```

## 业务码触发对照

- `E_INPUT_INVALID`：非法文档/配置进入失败清单，其余文档继续处理。
- `E_REWRITE_RULE_INVALID`：替换规则引用未声明模板或目标为空，原句保留并记录。
- `E_SNAPSHOT_STALE`：清单配置指纹或替换表版本与当前输入不符，全量重建。
- `E_SENTENCE_BUDGET_PENDING`：文档句数超单批预算，保留待办、下一批优先。

## 验证结论

以 `tests/test_pipeline.py` 为准，覆盖：加载校验与失败隔离、切句偏移与确定性分词、
定点量化与 float64 参考分、定点/float 边界反转标注、重叠替换抑制、单段落与单规则的
句级精确失效、增量与全量逐字段等价、串行/并发一致、并发硬上限、中断恢复等价、
快照过期重建、预算待办标记。运行结果见仓库根目录 `runtime/`（运行时生成，不随源码预置）。
