# 运行报告

## 环境与依赖

- 执行日期：2026-11-22
- 执行环境：WSL Ubuntu 22.04（Linux），未使用 Docker 或其他容器
- Python：3.11.0rc1
- numpy：1.26.4
- PyYAML：6.0.2
- pip：20.3.4（依赖已满足，无需升级）

实际执行：

```bash
python3 -m pip install -r requirements.txt
```

输出摘要：

```text
Requirement already satisfied: numpy>=1.24 ... (1.26.4)
Requirement already satisfied: PyYAML>=6.0 ... (6.0.2)
PIP_EXIT=0
```

## 样本生成

```bash
python3 scripts/generate_samples.py --output samples
```

固定种子为 `20261122`。生成 10 篇受控词表新闻，其中 `DOC-007` 被非法实体注入后按预期拒绝；配置最大并发数为 4。样本量化档位为 5、定点位宽为 6，量化步长 `0.2`。

## 主运行

```bash
python3 run.py --news samples --config samples/config.yaml \
  --injections samples/injections.json \
  --report reports/sample_report.json
```

并发运行退出码为 0，完成 9 篇有效文档，`DOC-007` 以 `E_CONFIG_INVALID` 单独进入失败清单。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

结果：`Ran 16 tests in 0.275s`，全部通过。覆盖：

- 配置、文档、triple 和模板版本校验；
- 中文句界与原文偏移；
- 定点分项、参考路径、舍入残差及重复运行一致性；
- 覆盖触界、固定/参考边界句、竞争排序；
- 词频与实体主导的扰动翻转归因；
- 未命中注入项判定；
- 长键优先替换与重叠抑制；
- 缺失模板保留区段；
- 阶段异常隔离；
- 线程池并发上限；
- 状态恢复与完整运行一致。

## 一致性验收

```bash
python3 scripts/verify_samples.py
```

实际结果：

```json
{
  "boundary_sentences": 2,
  "complete": true,
  "concurrent_equals_repeated": true,
  "concurrent_equals_serial": true,
  "coverage_boundary_hits": 8,
  "dependencies": {
    "numpy": "1.26.4",
    "python": "3.11.0rc1",
    "pyyaml": "6.0.2"
  },
  "documents": 9,
  "failure_codes": {"E_CONFIG_INVALID": 1},
  "failures": 1,
  "injection_selection_flip_dominant_components": {
    "entity_coverage": 1,
    "word_frequency": 2
  },
  "injections_applied": 8,
  "injections_not_applied": 2,
  "rewrite_errors": 3,
  "rewrite_table_warnings": 1,
  "suppressed_replacements": 5
}
```

并发、串行、重复运行三份报告的 SHA-256 完全相同：

```text
ad7eb8eb2ab3769325d93dfb702fcccfae5e3129b6c583109c69104583dbd83d
```

## 关键诊断结果

### 覆盖与边界句

- 9 篇有效文档均达到实际覆盖下限，覆盖触界事件合计 8 次。
- `DOC-002` 出现 2 个固定/参考路径边界句：
  - 句 1：固定 `0.400000`，参考 `0.378667`，固定选中而参考弃选；
  - 句 2：固定 `0.400000`，参考 `0.509333`，固定弃选而参考选中。

### 扰动归因

- `INJ-FREQ-002`：`DOC-002` 句 3 由 `0` 变为 `600000`，选中翻转，主导项 `word_frequency`。
- `INJ-ENT-003`：`DOC-002` 句 3 的复合实体扰动命中，主导项 `word_frequency`。
- `INJ-ENT-004`：`DOC-004` 句 2 由 `200000` 变为 `600000`，选中翻转，主导项 `entity_coverage`。
- `INJ-MISS-001`：`DOC-999` 不存在，列为未生效。
- `INJ-MISS-002`：`DOC-004` 句 99 越界，列为未生效。

### 错误链

- `DOC-007`：`INJ-ILLEGAL-001` 在段落 1 上下文注入空名称实体；加载阶段以 `E_CONFIG_INVALID`、`DocumentValidationError` 拒绝该文档，其他文档继续处理。
- `R002` 被 `INJ-TEMPLATE-001` 改绑到未声明模板 `ghost_v1`；表级输出 1 条 `E_TEMPLATE_MISSING` 警告。
- `DOC-001`、`DOC-005`、`DOC-009` 的选中句命中“表示”，该片段未被替换并输出 3 条保留记录。
- 重写冲突记录中，`R005` 压制 `R004`，`R006` 压制 `R007`，共 5 条被抑制替换。

## 并发与退出信号实测

使用 150 篇确定性文档、`max_workers=4` 启动 CLI，在运行中发送 `SIGINT`：

```text
exit_code = 0
report exists = true
state manifest exists = true
progress log exists = true
persisted documents = 125
processed document ids = 125
pending document ids = 25
report completion.complete = false
```

随后执行：

```bash
python3 run.py --news /tmp/signal_news.jsonl --config samples/config.yaml \
  --injections samples/injections.json --state-dir /tmp/signal_state_opt \
  --resume --report /tmp/signal_report_opt_resumed.json
```

恢复退出码为 0，150 篇全部完成。恢复报告与同输入全新串行报告的 SHA-256 完全一致：

```text
1cf774909c1fda1f5fd7f2ee927ef7b29de3c00fe1ddd197ed4529daf68a8c02
```

## 结论

定点打分、覆盖约束、边界翻转、模板缺失、替换冲突、扰动归因、线程池排序一致性和持久化恢复均已在本地确定性样本上验证。完整报告见 `reports/sample_report.json`。
