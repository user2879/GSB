# 实际验证结果

验证命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/acceptance.py --out artifacts/acceptance
```

## 环境

- Python `3.11.0rc1`
- NumPy `1.26.4`
- Pillow `11.3.0`
- PyYAML `6.0.4`
- Linux WSL2，无 Docker/容器，无网络服务

## 单元测试

`20` 项全部通过，覆盖校验、编码/融合重复一致性、引导/种子/步数真实变化、增量与全量一致、陈旧/最新区分、待办优先、并发一致、快照恢复和五类故障注入。完整日志：`artifacts/acceptance/unittest.log`。

## 端到端验收

`scripts/acceptance.py` 共执行 13 条编辑：提示词替换、图像替换、运动表替换、融合权重变更、去噪步数变更、分块行数变更、量化档位切换、图像删除、引导强度、种子和再次替换/重分块。每一条编辑后都与同最终状态的“空白缓存、一次全量重建”比较：

- 逐帧 PNG 解码像素：全部一致
- 帧内容/像素指纹：全部一致
- 全部去噪步摘要与配置指纹：全部一致
- 融合向量与量化残差账本：全部一致
- 分块行指纹与潜变量区域：全部一致

关键不变量均为 `true`：

```text
incremental_vs_full_pixel_and_fingerprint
stale_latest_distinction
pending_priority
concurrency_determinism
snapshot_restore_exact
fault_injection_hits
unit_tests
interaction_latency_within_budget
```

最大响应时延实测 `92.214 ms`，配置预算 `250 ms`，每批去噪点预算 `3`。

## 最终帧指纹

| frame | content fingerprint | pixel SHA-256 |
|---|---|---|
| 0 | `c94667c87a1299cf28eeb9ad060a0b098fc953ba8b2bf9e39b08f44b114e8168` | `914d19b8d9ec99c16d0aba940f04f0bbc5cd89a927353f6bcdc5d43a872b1d1c` |
| 1 | `96ebce3febdc004e3ebaf872d1ae1b33f4a8c868c498f3d7030f953120b06b8d` | `0a367edc2ed758be7665f5cb57a0d492cdd01d7c3ca1f3ba3b87f8710dab673d` |
| 2 | `6926aae67fafa121fdc575eda2b521a75bba3f58e45ee579119c7bf81c25fdab` | `6ffcb216644cd67ee7e109ac37ef18cf53a79d40e64c470da1c1987bb5a99958` |

最终运行目录：`artifacts/acceptance/runs/incremental-output/`。

## 陈旧摘要实测

触发第 14 个游标后先返回单个预算批：

```text
code=E_STALE_SUMMARY, stale=true, is_latest=false
requested_cursor=14, based_on_cursor=13
requested_config_fp != based_on_config_fp
pending=15, pending_codes=[E_BUDGET_PENDING]
```

继续排空待办后：

```text
code=OK, stale=false, is_latest=true
requested_cursor=14, based_on_cursor=14
requested_config_fp == based_on_config_fp
completed_frames=[0, 1, 2]
```

## 并发、恢复与预算

- `max_concurrency=1` 与 `max_concurrency=4` 的帧指纹、步骤摘要、行结果和真实像素完全一致。
- 第一批发只完成 `denoise f0/s0`；第二批首任务为上一批待办中的 `denoise f1/s0`，证明下一批优先处理待办。
- 首批快照版本为 `1`，重启后恢复游标、12 条待办并最终与空白全量重建完全一致；相同状态再次提交不创建新快照版本。

## 故障注入

| 注入编号 | 阶段/故障 | 结果 |
|---|---|---|
| `INJ-DIM` | fusion 条件维度不符 | `E_PARTIAL_FAILURE`，帧 0/2 完成，帧 1 隔离 |
| `INJ-NONFINITE` | denoise 非有限潜变量 | `E_PARTIAL_FAILURE`，帧 0/2 完成，帧 1 隔离 |
| `INJ-ROW-OOB` | 行号越界 | `E_PARTIAL_FAILURE`，帧 0 保留 3 行，帧 1/2 完成 |
| `INJ-QUANT-OOB` | 档位越界 | `E_CONFIG_INVALID`，注入编号进入报告 |
| `INJ-STALE-SNAP` | 陈旧快照版本 | `E_STALE_SNAPSHOT_VERSION`，报告命中 `INJ-STALE-SNAP` |

机器可读完整结果：`artifacts/acceptance/report.json`；摘要：`artifacts/acceptance/SUMMARY.md`。
