# 近重复探测组件验证报告

本报告由 `python -m near_dup.cli demo` 自动生成，所有数据来自本地固定种子样本。

## 环境与样本

- Python: `3.11.0rc1`
- NumPy: `2.4.6`
- 初始文本数: `14`
- 初始簇数: `9`
- 单元素簇: `T008, T009, T010, T011, T012, T013, T014`
- 边界文本对数: `1`

## 校验与故障注入

- 非法文本库命中码: `{'E_TEXT_INVALID': 8}`
- 非法编辑计划命中码: `{'E_PLAN_INVALID': 3, 'E_TEXT_INVALID': 1}`
- `I01-ZERO-DIM` / `zero_signature_dimension`: 期望 `E_CONFIG_INVALID`，实际 `E_CONFIG_INVALID`，命中=True
- `I02-NONFINITE-SIM` / `non_finite_similarity`: 期望 `E_CONFIG_INVALID`，实际 `E_CONFIG_INVALID`，命中=True
- `I03-BUCKET-OVERFLOW` / `bucket_bits_out_of_range`: 期望 `E_CONFIG_INVALID`，实际 `E_CONFIG_INVALID`，命中=True
- `I04-CLUSTER-COLLISION` / `cluster_id_conflict`: 期望 `E_CLUSTER_ID_CONFLICT`，实际 `E_CLUSTER_ID_CONFLICT`，命中=True

## 增量、预算与并发

- 阈值单改时签名复用: `True`（复用 15 条）
- 首轮预算计算: `['T001']`，剩余待办: `13`
- 待办优先: `True`；累计预算: `64`
- 并发/串行逐字段一致: `True`
- 增量与全量重建逐字段一致: `True`
- 投影哈希: `2a2687ef3a9b9e196480fc844778231113d835a5698be9777d374cab2b4b4c89`

## 快照恢复

- 模拟阶段中断: `simulated crash after stage 'signatures'`
- 中断快照: 版本 `6`，已完成 `3`，待办 `12`
- 恢复结果等于全量重建: `True`
- 版本不符码: `E_SNAPSHOT_VERSION_MISMATCH`
- 配置哈希不符码: `E_SNAPSHOT_VERSION_MISMATCH`

## 复现命令

```bash
.venv/bin/python scripts/generate_samples.py --out samples
.venv/bin/python -m near_dup.cli demo --samples samples --run-dir runs/demo --report-dir reports
.venv/bin/python -m unittest discover -s tests -v
```

机器可读明细见 `reports/diagnostic_report.json`。
