# Deterministic Cross-Lingual Retrieval Pipeline

本项目在本地确定性样本上实现一条可审计的中英跨语言相关性检索管线：归一化、语言识别、分语言 BM25 倒排检索、翻译桥接、定点打分融合、预算调度、阶段快照、异常注入隔离与评估报告。

所有数据由固定种子生成器从受控中英词表重建，不依赖账号、容器、线上服务或真实业务数据。

## Environment

- Python 3.10+
- NumPy 1.26.4

```bash
python --version
python -m pip install -r requirements.txt
```

## Quick Start

```bash
# 1. 用固定种子重建样本
python run_pipeline.py generate \
  --output samples/default \
  --seed 20261120 \
  --documents 48 \
  --queries 20 \
  --zh-ratio 0.5 \
  --mixed-ratio 0.25

# 2. 全量运行并生成 JSON 报告
python run_pipeline.py run \
  --sample-dir samples/default \
  --state-dir artifacts/state-main \
  --report artifacts/report.json \
  --fresh

# 3. 运行独立异常注入配置
python run_pipeline.py run \
  --sample-dir samples/default \
  --state-dir artifacts/state-injection \
  --report artifacts/injection-report.json \
  --injections samples/default/injections.json \
  --fresh

# 4. 自动验证并发/串行、全量/增量、预算与注入
python run_pipeline.py verify \
  --sample-dir samples/default \
  --config samples/default/config.json \
  --injections samples/default/injections.json \
  --output artifacts/verification.json

# 5. 单元测试
python -m unittest discover -s tests -v
```

主要产物：

- `artifacts/report.json`：全量运行审计报告。
- `artifacts/injection-report.json`：四类真实输入故障注入报告。
- `artifacts/verification.json`：串行/并行、全量/增量、预算和注入验证清单。
- `artifacts/state-main/manifest.json`：阶段快照清单、逻辑版本、哈希和已完成批次。

## Sample Schema

| File | Record fields |
| --- | --- |
| `documents.jsonl` | `doc_id`, `lang`, `title`, `body` |
| `queries.jsonl` | `query_id`, `lang`, `query` |
| `translations.jsonl` | `source_lang`, `source_term`, `target_lang`, `target_term`, `confidence` |
| `qrels.jsonl` | `query_id`, `doc_id`, `relevance` |
| `config.json` | 归一化、LID、翻译权重、BM25、召回、融合、定点、预算、线程和快照参数 |
| `injections.json` | 独立注入项：指定注入 ID、查询 ID、阶段、故障类型与参数 |
| `sample.json` | 生成种子、规模、语言比例与受控概念表 |

语言枚举为 `zh`、`en`、`mixed`；倒排索引语言为 `zh`、`en`。非法记录统一记录为 `E_RECORD_INVALID` 并进入失败清单，不中断其他记录。

## Determinism Contract

- 查询、候选、词项分解、翻译抑制、失败、预算序列都使用稳定键排序。
- 预算配额在查询 ID 字典序上一次性分配；线程只执行已获配额的查询。
- 并行和串行执行产生相同的 `determinism.sha256`，并逐字段比较查询结果、预算、评估、失败与快照进度。
- 快照使用逻辑版本号（已提交批次数），因此中断后续跑与一次性全量执行最终得到相同版本。
- 报告中排除时间戳、耗时、观测线程数和 `threads.max_workers` 后再计算确定性摘要。

## Exit Semantics

- 完整运行或可处理中断：退出码 `0`，报告 `status` 为 `complete` 或 `interrupted`。
- 快照身份、配置哈希、翻译哈希或输入哈希不一致：拒绝续跑，退出码 `2`，错误码 `E_CONFIG_MISMATCH`。
- SIGINT/SIGTERM 只设置停止标志；当前批次完成并提交快照后退出。

## Validation Status

当前样本实际验证结果：

- 9 个单元测试全部通过。
- `verification.json` 中 12 项检查全部通过。
- 默认样本跨语言扩展、翻译抑制、无翻译诊断和桥接结果均已命中。
- 四类注入均命中并被限制在目标查询/阶段，其他查询继续产出结果。

算法、字段、预算和快照细节见 `docs/TECHNICAL.md`。
