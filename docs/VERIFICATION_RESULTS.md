# 实际验证结果

总体结果：通过

## 环境

- Python：`3.11.0rc1`
- 平台：`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- numpy：`1.26.4`
- PyYAML：`6.0.1`

## 检查项

- [PASS] `concurrency_primitive`
- [PASS] `serial_concurrent_equivalence`
- [PASS] `warm_cache_snapshot_recovery`
- [PASS] `incremental_edit_sequence`
- [PASS] `point_budget_pending_and_completion`
- [PASS] `injection_and_batch_faults`
- [PASS] `precision_attribution`
- [PASS] `boundary_and_flip_rules`

机器可读完整报告：`artifacts/verification_report.json`。

复跑命令：`python -m unittest discover -s tests` 与 `python -m vibpipe.verify`。
