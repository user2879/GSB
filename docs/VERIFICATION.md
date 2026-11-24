# 实际验证记录

验证时间：2026-11-25，Asia/Shanghai。

## 环境

```text
Python 3.11.0rc1
numpy 1.26.4
pytest 8.2.0
```

## 已执行命令

```bash
python3 -m dialogue_rag.sample_data --out samples
python3 -m pytest tests/test_dialogue_rag.py -q --disable-warnings --no-header
python3 -m dialogue_rag.cli run \
  --config samples/config.json \
  --commands samples/commands.json
python3 -m dialogue_rag.cli run \
  --config samples/config.json \
  --commands samples/commands.json
```

实际结果：

- 样本生成成功：`samples/config.json`、`samples/commands.json`。
- 完整测试：`.............. [100%]`，退出码 `0`，共 14 项通过。
- 首次 CLI：输出 `artifacts/diagnostic_report.json`，快照全局版本 `8`（1 次基线 + 7 条命令），命令游标 `7`。
- 第二次 CLI：恢复模式 `resume`，清单与文件哈希校验通过；两次报告的 `steps` 与 `turn_results` 相等。
- 主样本失败清单数量：`0`。

## 验收点抽查

- 查询替换 `T3`：召回阶段仅 `T3` 未命中；`T1/T2` 评分直接复用，下游按轮次链检查/重算；`TP_T1`、`TP_T5` 话题复用，包含变更轮次的 `TP_T3` 重算。
- 证据权重修改 `E002`：证据级定点贡献中仅 `E002` 标记为 `forensic_weighting`，其他证据贡献复用，相关引用轮次重新求和。
- 通道权重将 `ch_sem.weight` 从 `1.15` 调到 `1.05` 后，`T1/E001` 权威通道由 `ch_sem` 变为 `ch_lex`。
- 样本中三对初始话题的 Jaccard 相似度恰为 `0.25`，与阈值相等；默认 `equal_rule=merge` 时记录为合并，切到 `split` 后归属真实变为六个单轮话题。
- 最终切到 `F8/G2` 后，`T1` 定点置信度为 `0.625/supported`，精确有理数参考为 `0.5866667/uncertain`；报告归类为 `quantization_rounding`，并输出阈值边际与判据。
- 低预算验证：预算 4 点时 `T1` 消耗 3 点，`T2` 起保留 `E_BUDGET_PENDING`；下一批高预算完成后与空白全量结果一致。
- 快照验证：有效清单返回 `I_SNAPSHOT_RESUME`；篡改阶段文件导致 SHA-256 不匹配时返回 `E_SNAPSHOT_REBUILD`。
- 并发验证：`workers=4` 与串行的完整步骤和轮次结果一致。
- 随机命令验证：8 个固定种子、每个 14 条命令（含未知命令、缺失目标、删除、权重、规则和位宽切换）的一次性运行、逐条重启恢复运行、最终状态空白全量重算三者一致；`steps`、`turn_results`、`failure_list` 满足对应逐字段相等断言。

## 产物

- 可运行包：`dialogue_rag/`
- 确定性样本：`samples/config.json`、`samples/commands.json`
- 自动化测试：`tests/test_dialogue_rag.py`
- 诊断报告样例：`artifacts/diagnostic_report.json`
