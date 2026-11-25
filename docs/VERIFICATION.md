# 实际验证结果

环境：Python 3.11、numpy 1.26.4、pytest 8.2.0（需求期望 8.2.2，本机已装 8.2.0，
API 与本工程用法兼容；`requirements.txt` 固定 8.2.2 供全新环境安装）。
无容器、无外部服务，样本由固定种子 `SEED=20261125` 生成。

## 测试
命令：`pytest`
结果：`36 passed`。覆盖加载校验、迷你 YAML 解析、三版回执裁决、分位标定、定点量化/饱和、
误差归因、蒸馏校正（含“不降反升”标注）、预算调度、缓存精确失效、
增量/全量逐字段一致、串行/并行一致、强制中断恢复、CLI 子进程运行。

## 全量账本（命令流最终配置）
`python -m qcalib.cli run ... --commands configs/commands.json`
输出四层，失败清单仅包含确定性的畸形回执 `E_RECEIPT_MALFORMED`
（`receipt_bad.json#entry0`，覆盖率千分比越界），其余层未中断。

| 层 | 标定点/组 | 饱和元素 | 蒸馏前 RMSE | 蒸馏后 RMSE | 首次超容差环节 |
| --- | --- | --- | --- | --- | --- |
| embedding_proj | 4 | 32 | 1.51720 | 1.19811 | saturation_truncation |
| fusion_block | 4 | 16 | 2.32000 | 1.58431 | saturation_truncation |
| retrieval_head | 8 | 32 | 2.69225 | 2.05373 | saturation_truncation |
| score_mixer | 1 | 8 | 1.52296 | 1.17627 | saturation_truncation |

四层蒸馏后偏差均下降，`increased=false`；“不降反升”分支由
`tests/test_distill.py::test_distillation_increase_is_flagged_with_judgement_numbers`
以高学习率确定性触发并校验判据数值。

## 预算与调度
- 默认配置消耗 17 个标定点，位于全局预算 128 内；单批量化点数 16 下
  `quant_batches` 分别为 2/1/2/1。
- `configs/pipeline_constrained.yaml`（全局预算 10、fusion_block 配额 4）
  确定性产生 `E_QUOTA_EXCEEDED` 与 `E_BUDGET_PENDING`，待办任务保留，
  由 `tests/test_budget_end_to_end.py` 校验。

## 增量/全量与并发一致
- 逐命令增量最终账本与一次性全量账本使用 `qcalib diff` 比对，
  忽略瞬态 `cache`/`reuse_ledger`/`canonical_invalidation_ledger` 视图后
  返回 `{"equal": true}`、退出码 0。
- 增量运行 `reuse_ledger` 出现命中，冷启动全量无命中；规范失效账本
  `canonical_invalidation_ledger` 在两种运行下逐字段相同。
- 串行（workers=1）与并行（workers=4）的层产物与复用账本逐字段一致，
  见 `tests/test_pipeline_parity.py`。

## 中断恢复
`--fault before_commit:ledger_aggregation` 触发后退出码 2，前四阶段目录已持久化、
最终账本目录不存在；`recover` 报告可恢复阶段且不接受未提交账本。
去除故障在同一状态目录重跑后，与空白全量 `diff` 退出码 0，
见 `tests/test_snapshot_recovery.py`。

## 子进程与临时目录
`tests/test_subprocess_cli.py` 通过 `subprocess` 启动 Python 解释器运行 CLI，
在 pytest 临时目录读写状态、账本与诊断报告，验证本地子进程与文件系统能力。
