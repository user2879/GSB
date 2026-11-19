# 验证记录

验证日期：2026-11-19

环境：Python 3.11.0rc1，NumPy 2.4.6，Linux 本地文件系统。

## 自动化测试

```text
$ .venv/bin/python -m unittest discover -s tests -t . -v
test_overflow_refuses_to_truncate ... ok
test_text_only_edit_invalidates_only_visible_blocks ... ok
test_visibility_background_mask_and_critical_marker ... ok
test_wrap_is_deterministic_and_rate_is_reported ... ok
test_block_error_is_reported_and_other_blocks_cancel ... ok
test_export_commit_pairs_frames_and_report ... ok
test_manifest_frame_count_mismatch_triggers_digest_repair ... ok
test_memory_budget_serializes_and_rejects_impossible_blocks ... ok
test_reuse_digest_repair_and_version_namespaces ... ok
test_business_codes ... ok
test_force_kill_reuses_only_valid_persisted_blocks ... ok
test_sigterm_exits_at_boundary_and_resumes ... ok
test_generator_is_deterministic ... ok

Ran 13 tests in 6.605s
OK
```

## 主样本增量复用

首次运行：

```json
{
  "frame_count": 80,
  "blocks": {"total": 16, "reused": 0, "recomputed": 16, "invalidated": 16},
  "business_codes": ["E_SPEED_LIMIT"]
}
```

相同版本第二次运行：

```json
{
  "frame_count": 80,
  "blocks": {"total": 16, "reused": 16, "recomputed": 0, "invalidated": 16},
  "version": "bb3ab7f4e2aabc669918fa5768dc7ea64ca54fca11e5dbd1cf72eac8dba306d1"
}
```

四步编辑脚本的失效块数：

```json
[
  [1, "set_text", 7],
  [2, "set_style", 16],
  [3, "set_time", 6],
  [4, "set_style", 16]
]
```

## 并发确定性

同一输入分别以 `concurrency=1` 和 `concurrency=3` 运行，输出形状均为 `[80, 180, 320]`：

```text
numpy.array_equal(serial_frames, parallel_frames) == true
```

## 摘要损坏修复

人为翻转 `blocks/000007.npz` 中一个字节后重启：

```json
{
  "business_codes": ["E_DIGEST", "E_SPEED_LIMIT"],
  "blocks": {"total": 16, "reused": 15, "recomputed": 1},
  "frame_count": 80
}
```

## 中断恢复

```text
$ .venv/bin/python scripts/interruption_test.py
{
  "ok": true,
  "results": [
    {
      "signal": "sigterm",
      "interrupted_exit_code": 130,
      "reused_blocks": 1,
      "recomputed_blocks": 23,
      "frame_count": 120,
      "status": "ok"
    },
    {
      "signal": "sigkill",
      "interrupted_exit_code": -9,
      "reused_blocks": 1,
      "recomputed_blocks": 23,
      "frame_count": 120,
      "status": "ok"
    }
  ]
}
```

## 非法输入无输出

`anchor=(2,2)` 位于安全区 `(16,16,288,144)` 外时：

```json
{
  "ok": false,
  "error": {
    "code": "E_ANCHOR",
    "message": "Screen anchor must lie inside the safe area"
  }
}
```

退出码为 `2`，对应输出目录中没有 `frames.npy`。
