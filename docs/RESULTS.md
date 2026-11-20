# 实际验证结果

本文件由 `python tools/verify.py` 在本地确定性样本上自动生成。

## 环境

- Python `3.11.0rc1`
- NumPy `1.26.4`
- Platform `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`

## 增量编辑序列

编辑序列依次累积执行：追加分段、替换单段、删除分段、调整马氏距离候选参数、切换频带、切换定点量化档位、修改阈值。

| 步骤 | 增量=全量 | 段数 | 复用 | 重算 | 快照码 |
|---|---:|---:|---:|---:|---|
| initial | true | 18 | 0 | 162 | `-` |
| append | true | 19 | 132 | 39 | `E_SNAPSHOT_STALE` |
| replace | true | 19 | 127 | 44 | `E_SNAPSHOT_STALE` |
| delete | true | 18 | 147 | 15 | `E_SNAPSHOT_STALE` |
| algorithm_parameter | true | 18 | 108 | 54 | `E_SNAPSHOT_STALE` |
| band_configuration | true | 18 | 54 | 108 | `E_SNAPSHOT_STALE` |
| quantization | true | 18 | 90 | 72 | `E_SNAPSHOT_STALE` |
| threshold | true | 18 | 144 | 18 | `E_SNAPSHOT_STALE` |

逐段特征摘要、三算法原始分数、定点归一化整数、融合总分、异常标记、边界清单、失败清单、注入命中与波形指纹均逐字段相等；增量与空白全量的复用/重算清单描述执行路径，按定义允许不同，但均验证了稳定排序。

## 跨算法失效粒度

- `parameter_change_only_robust_mahalanobis_raw`: `true`
- `band_change_reuses_crest_raw`: `true`
- `band_change_recomputes_robust_and_spectral_raw`: `true`
- `quantization_change_reuses_all_raw`: `true`
- `threshold_change_reuses_raw_and_quantized`: `true`

## 注入与边界

- `all_configured_injections_hit`: `true`
- `nonfinite_failure_recorded`: `true`
- `band_failure_recorded`: `true`
- `zero_window_failure_recorded`: `true`
- `threshold_boundary_recorded`: `true`
- `three_algorithms_on_healthy_segments`: `true`
- 注入运行段数：`18`
- 注入后失败记录数：`5`（其余分段继续执行）
- 命中注入数：`4`

## 量化与并发

- 定点整数随量化档位变化：`true`
- 异常标记随量化档位变化：`true`；档位 1 标记 `['motor-b/2', 'motor-b/4', 'motor-b/5', 'pump-a/0', 'pump-a/3']`，档位 2 标记 `['fan-c/2', 'motor-b/2', 'motor-b/4', 'motor-b/5', 'pump-a/0', 'pump-a/3']`
- 串行与 4 线程业务结果一致：`true`
- 最大实际并发：串行 `1`，并发版 `4`

## 报告文件

- `artifacts/report.json`：带四类注入的完整运行报告
- `artifacts/incremental_report.json`：最终编辑状态下的增量运行报告
- `artifacts/verification_report.json`：机器可读验收汇总
