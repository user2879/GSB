# 实际验证结果

- Python: `3.11.0rc1` on Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
- NumPy: `1.26.4`
- pytest: returncode `0`
```text
..........                                                               [100%]
10 passed in 2.23s
```

## 命令流一致性
- incremental vs blank full serial: `True`
- serial vs threaded: `True`

| step | op | batches | position hit/miss | rotation hit/miss | dot hit/miss | norm hit/miss | summary hit/miss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `switch_scheme` | 3 | 18/18 | 18/18 | 18/18 | 18/18 | 0/9 |
| 2 | `switch_scheme` | 1 | 18/0 | 18/0 | 18/0 | 18/0 | 3/0 |
| 3 | `update_scheme` | 3 | 42/6 | 42/6 | 42/6 | 42/6 | 6/3 |
| 4 | `update_scheme` | 3 | 42/6 | 42/6 | 42/6 | 42/6 | 6/3 |
| 5 | `add_scheme` | 3 | 60/6 | 60/6 | 60/6 | 60/6 | 9/3 |
| 6 | `set_lengths` | 1 | 24/0 | 24/0 | 24/0 | 24/0 | 0/4 |
| 7 | `replace_sample` | 1 | 24/0 | 20/4 | 20/4 | 20/4 | 0/4 |
| 8 | `delete_sample` | 1 | 20/0 | 20/0 | 20/0 | 20/0 | 0/4 |
| 9 | `set_fixed_point` | 3 | 24/20 | 24/20 | 24/20 | 24/20 | 0/12 |
| 10 | `set_fixed_point` | 1 | 20/0 | 20/0 | 20/0 | 20/0 | 4/0 |

## 终态分层统计

| scheme | stratum | samples | keys | score mean | max abs score error | breach stages |
|---|---|---:|---:|---:|---:|---:|
| `rope_added` | training | 5 | 20 | -0.10245667 | 1.651e-05 | 0 |
| `rope_added` | extrapolation | 5 | 10 | -0.19305038 | 1.091e-05 | 0 |
| `rope_base` | training | 5 | 20 | -0.06381302 | 1.381e-05 | 0 |
| `rope_base` | extrapolation | 5 | 10 | -0.12193756 | 1.510e-05 | 0 |
| `rope_block4` | training | 5 | 20 | -0.09245377 | 1.174e-05 | 0 |
| `rope_block4` | extrapolation | 5 | 10 | -0.02211838 | 1.536e-05 | 0 |
| `rope_flash` | training | 5 | 20 | -0.11277351 | 1.241e-05 | 0 |
| `rope_flash` | extrapolation | 5 | 10 | 0.03160553 | 2.020e-05 | 0 |

- 外推位置交叉对照行数: `10`

## 粗量化档位首次超差（命令流第 9 步）

| scheme | sample | first stage | value | tolerance |
|---|---|---|---:|---:|
| `rope_added` | `s001` | `position` | 1.927663e-01 | 2.000000e-03 |
| `rope_added` | `s002` | `position` | 1.927663e-01 | 2.000000e-03 |
| `rope_base` | `s001` | `position` | 2.205845e-01 | 2.000000e-03 |
| `rope_base` | `s002` | `position` | 2.092234e-01 | 2.000000e-03 |
| `rope_block4` | `s001` | `position` | 1.737016e-01 | 2.000000e-03 |
| `rope_block4` | `s002` | `position` | 1.928221e-01 | 2.000000e-03 |
| `rope_flash` | `s001` | `position` | 1.970180e-01 | 2.000000e-03 |
| `rope_flash` | `s002` | `position` | 1.569526e-01 | 2.000000e-03 |

## 同一外推位置对照示例：`s001` / key_index `0` / distance `26`

| scheme | position err | rotation err | dot err | norm err | fixed score | fixed prob |
|---|---:|---:|---:|---:|---:|---:|
| `rope_added` | 6.309e-06 | 8.201e-06 | 4.882e-06 | 3.875e-06 | -0.356598 | 0.100052 |
| `rope_base` | 1.181e-05 | 8.890e-06 | 6.719e-06 | 2.894e-06 | -0.730949 | 0.064804 |
| `rope_block4` | 6.194e-06 | 8.479e-06 | 1.610e-06 | 4.708e-06 | 0.517067 | 0.227493 |
| `rope_flash` | 9.759e-06 | 6.042e-06 | 1.072e-06 | 1.098e-06 | -0.128105 | 0.127655 |

## 预算与故障隔离
- 首批待办: `['s003', 's004', 's005', 's006']`; consumed=`96`; codes=`['E_POINT_BUDGET_PENDING', 'E_POINT_BUDGET_PENDING', 'E_POINT_BUDGET_PENDING', 'E_POINT_BUDGET_PENDING']`
- 排空批次: `3`, pending=`[]`
- 配额超限失败数: `6`, codes=`['E_POINT_BUDGET_EXCEEDED']`
- 注入 stderr:
```text
INJECTION_HIT id=INJ-ROT-NAN-001 sample=s004 scheme=rope_base stage=rotation code=E_INJECTED_NONFINITE
```
- failure: `E_INJECTED_NONFINITE` sample=`s004` scheme=`rope_base` stage=`rotation`

## 快照恢复
- 未提交临时目录: `artifacts/snapshot-recovery-demo/tmp-stage-ds01k61m`; recover 前存在=`True`
- 崩溃恢复: `current_snapshot_ready`; 清理 orphan=`1`
- 删除 current pointer 后: `current_snapshot_ready`; global_version=`1`
- 真实 SIGKILL: returncode=`-9`, 临时目录 kill 后存在=`True`, 恢复状态=`current_snapshot_ready`, 清理=`1`, recover 后目录消失=`True`
