# 实际验证结果

验证时间环境由本机执行产生；所有样本和随机序列均为固定种子，不依赖运行时时钟。

## 环境

```text
Python 3.11.0rc1 (满足 >= 3.10)
venv: .venv/bin/python
numpy 1.26.4 (/home/zsl/feedback/uidfc5909d3-default-408/.venv/lib/python3.11/site-packages/numpy)
```

## 复现命令

```bash
PYTHONPATH=. .venv/bin/python audio_fingerprint/acceptance.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v | tee artifacts/unittest_run.txt
PYTHONPATH=. .venv/bin/python -m audio_fingerprint.cli validate --workspace sample_workspace
PYTHONPATH=. .venv/bin/python -m audio_fingerprint.cli rebuild --workspace sample_workspace --state state/cli-smoke
```

## 单元测试

最近一次实际输出保存在 `artifacts/unittest_run.txt`：

```text
Ran 10 tests in 56.172s

OK
```

覆盖：

- 时间量化档位和取整方向；
- 桶容量淘汰和稳定顺序；
- 候选并列、边界差值；
- 偏移投票 tie-break；
- 多版本冲突裁决；
- 六类样例校验业务码；
- 增/删/改/格式切换/阈值/桶容量/量化档位序列与空白全量 deep-diff 为空；
- 采样点预算、片段配额和待办优先恢复；
- 串行/2/4 并发 canonical 指纹一致；
- 3 组确定性 fuzz 编辑序列；
- 原子快照恢复。

## 端到端验收

最近一次机器报告：`artifacts/verification_report.json`；简表：`artifacts/verification_summary.txt`。

```text
acceptance_passed=True
[PASS] validation_business_codes
[PASS] invalid_command_stream_continues
[PASS] baseline_build_completed
[PASS] format_normalization_framewise_identity
[PASS] voting_candidates_boundary_and_arbitration
[PASS] deterministic_tie_rules
[PASS] precise_invalidation_and_reuse
[PASS] incremental_equals_blank_full_rebuild
[PASS] budget_quota_todo_priority_and_recovery
[PASS] threadpool_serial_field_equivalence
[PASS] signal_completes_batch_and_snapshot_recovery
[PASS] forced_termination_committed_snapshot_only
[PASS] bucket_eviction_records_present
```

关键实测值：

- 校验业务码顺序：`E1001, E1002, E1003, E1004, E1005, E2002`。
- 非法命令流产生：`E3001, E3002, E3003, E2003, E2003`，且后续合法正阈值命令仍被接受。
- 基线有效产物：657 个谱图帧、515 个指纹对、105 个非空桶。
- `c-alpha-v1`（22050 Hz/16-bit）与 `c-alpha-v2-fmt`（44100 Hz/8-bit）的 95 帧 `energy_q/bands_q` 零差异；归一内容指纹相同，原始格式指纹不同。
- 综合编辑序列后的增量与空白全量 canonical 指纹相同：
  `canonical_965188c21177aa4cc242c70774ceed321da4d38cc43a2bd86845422e5ee84d89`，deep-diff 为 0。
- 单帧阈值步骤实测：1 个峰值集合重建，引用它的指纹对和桶重建，同时存在桶复用命中。
- 预算验收首批稳定选中 `frame:c-alpha-v1:000000` 至 `000003`，首个待办为 `frame:c-alpha-v1:000004`，第二批首单元确认为该帧；最终与无预算限制基线 deep-diff 为 0。
- 串行、2 线程、4 线程 tiny workspace canonical 指纹均为：
  `canonical_3f4e0533e91354cb868165e780380e895ef65380c25caa93569726e1d707eb7f`。
- 基线桶内共记录 161 条容量挤出条目。

CLI 校验实测有效片段 9 个、有效编辑命令 8 个；CLI 全量重建成功，输出见 `artifacts/cli_validate.json` 与 `artifacts/cli_rebuild.json`。
