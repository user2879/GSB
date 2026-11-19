# 验收记录

运行日期：2026-11-19  
环境：WSL2 Linux，Python 3.11.0rc1，`numpy==2.2.6`，`PySide6==6.8.3`；无 GPU、无容器、无运行时网络访问。

## 1. 自动化测试

命令：

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
```

结果：**20/20 通过**。

```text
test_budget_admission_waits_and_merges_by_index ... ok
test_budget_too_small_for_one_trace_fails_explicitly ... ok
test_tight_budget_waits_without_changing_results ... ok
test_worker_count_does_not_change_diagnostics ... ok
test_first_tolerance_exceedance_is_fixed_point_authoritative ... ok
test_repeated_runs_are_vertex_identical ... ok
test_window_reads_persisted_snapshots ... ok
test_cancel_at_boundary_and_resume_matches_uninterrupted ... ok
test_gui_view_marks_tampered_snapshot_invalid ... ok
test_tampered_snapshot_is_rejected_and_not_used ... ok
test_version_change_archives_previous_run ... ok
test_cli_invalid_input_writes_no_state_or_output ... ok
test_non_finite_coordinate_is_schema_error ... ok
test_out_of_bounds_index_is_index_error ... ok
test_unknown_operator_is_schema_error ... ok
test_non_positive_tolerance_is_schema_error ... ok
test_sigkill_resumes_only_from_valid_persisted_snapshots ... ok
test_sigterm_finishes_current_operator_then_resumes ... ok
test_defect_counts_and_rescan_consistency ... ok
test_repair_changes_are_deterministic_and_degenerate_free ... ok

Ran 20 tests in 16.911s
OK
```

覆盖点：

- 有限数、结构、算子类型、容差、越界索引校验；
- 非法输入不创建状态/输出目录；
- 拓扑缺陷计数与逐面重扫一致；
- 重复运行逐顶点定点/参考坐标完全一致；
- 并发 1 与并发 8 的诊断数值完全一致；
- 紧预算任务按索引等待、峰值驻留不超限，单任务超预算返回 `E_BUDGET`；
- 优雅停止在算子边界生效并退出 `130/143`；
- `SIGKILL` 后只从已持久化有效快照续算；
- 续算结果与不间断执行逐步骤、逐顶点一致；
- 摘要被篡改的快照被拒绝且不会被覆盖；
- GUI 将失效快照标红并拒绝绘制；
- 版本变化时旧运行进入归档目录。

## 2. 最终样例

生成并运行：

```bash
PYTHONPATH=src python3 -m mesh_audit generate --directory samples
PYTHONPATH=src python3 -m mesh_audit run \
  --mesh samples/sample_defective.mesh.json \
  --script samples/sample_repair_script.json \
  --state state/sample \
  --output output/sample
```

关键输出：

```json
{
  "ok": true,
  "exit_code": 0,
  "completed_steps": 4,
  "total_operators": 4,
  "verdict": "FAIL_FIXED_POINT_AUDIT",
  "first_tolerance_exceedance": {
    "max_abs_error": 5.89761087521623e-07,
    "operator_index": 4,
    "tolerance": 1e-07,
    "vertex": 0
  },
  "invalid_snapshot_count": 0
}
```

拓扑对照：

```text
输入: V=10 F=12 E=20 非流形边=1 边界边=6 边界环=1 开放链=1
      退化面=1 孤立点=1 方向冲突边=2 Euler=2 rescan=true
最终: V=11 F=15 E=23 非流形边=1 边界边=2 边界环=0 开放链=1
      退化面=0 孤立点=1 方向冲突边=0 Euler=3 rescan=true
```

算子计数：

| 步骤 | 算子 | 执行计数 | 细节 |
|---:|---|---:|---|
| 1 | hole_triangulation | 1 | 填充 1 环，新增 4 面/1 顶点，跳过 1 个开放链 |
| 2 | remove_degenerate_faces | 1 | 删除面索引 11，剩余 15 面 |
| 3 | reorient_normals | 1 | 翻转面索引 5，共 2 个面连通分量 |
| 4 | laplacian_smooth | 16 | 477 次坐标更新，λ=21/50 |

传播摘要：

```text
step 1 source=5  affected=10 max_layer=3 source_only=5
step 2 source=2  affected=10 max_layer=2 source_only=2
step 3 source=3  affected=10 max_layer=2 source_only=3
step 4 source=10 affected=10 max_layer=3 persists_above_tolerance=10
```

每步定点几何效应：

| 步骤 | 最大顶点增量 | bbox 变化范数 | 面积变化 | 体积变化 | 曲率变化 |
|---:|---:|---:|---:|---:|---:|
| 1 孔洞填充 | 0（新增 1 顶点） | 0 | +1.000000 | +0.333333 | -0.089739 |
| 2 退化面移除 | 0 | 0 | 0 | 0 | +0.010949 |
| 3 法线重定向 | 0 | 0 | 0 | +0.333333 | 0 |
| 4 拉普拉斯平滑 | 0.588019 | 0.932194 | -6.284858 | -0.999999 | -0.642934 |

最终判定：

```text
decision_basis = fixed_point_only
float64_reference_used_for_verdict = false
coordinate_tolerance_respected = false
topology_rescan_matches = true
parallel_face_scan_matches = true
all_fixed_coordinates_finite = true
```

## 3. 中断与强制终止实测

验收测试使用包含 4000 次平滑迭代的同一脚本分别执行：

1. 不间断基线；
2. 在 step 3 已持久化、step 4 执行期间发送 `SIGTERM`；
3. 在 step 3 已持久化、step 4 执行期间发送 `SIGKILL`。

结果：

- `SIGTERM` 后当前平滑算子完成并写入 step 4，进程退出 `143`；再次运行从 step 4 恢复并完成报告。
- `SIGKILL` 后仅能加载 step 0..3 的有效快照；再次运行从最后一个有效边界续算。
- 两种场景恢复后的 step 0..4 定点坐标、参考坐标、面数组、执行计数、逐顶点误差和传播记录均与不间断基线完全一致。

## 4. 失效快照拒绝

将 `step-0002.json` payload 中一个定点坐标加 1、但不更新外层摘要后：

- `mesh-audit run` 返回 `E_SNAPSHOT_INVALID` 和退出码 `3`；
- 原快照内容未被覆盖；
- `inspect` 显示最后连续有效步骤为 1；
- GUI 时间线将 step 02 标红为 `[INVALID]`，不绘制该步骤，不允许用于恢复。

## 5. 并发一致性

同一网格与算子链分别以 `concurrency=1` 和 `concurrency=8` 运行：

- 每步定点/参考坐标、面索引、执行计数、逐顶点误差、传播源、逐层顶点集合与逐层最大偏差完全一致；
- 仅调度指标（实际线程数、峰值驻留预算）变化；
- 每次运行的峰值驻留节点均未超过声明的 `memory_budget_nodes`。

## 6. 产物

- 最终报告：`output/sample/report.json`
- 快照：`state/sample/current/step-0000.json` … `step-0004.json`
- 状态清单：`state/sample/current/manifest.json`
- GUI 截图：`output/sample/workbench.png`
- CLI 摘要：`output/sample/run_stdout.json`
- 自动化测试日志：`output/sample/test_run.log`
