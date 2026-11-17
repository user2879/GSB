# 实际验证记录

验证环境：Linux/WSL2，Python 3.11.0rc1，NumPy 2.2.6，本地 `.venv`，
无容器；固定种子 20261118；日期 2026-11-18。

## 样本规模

- 24 个实体、6 个白名单特征、3 个来源（v1/v2/v3），合并输出 144 行。
- v3 分 3 个分区，每分区 8 实体。
- 缺失计划共 9 个来源级缺失坐标（三源共同缺失 `ent_0003/f_income`）。
- 来源偏置 v1=0、v2=0.004、v3=-0.011，制造可复现冲突。

## 解析与互转

- v1 定宽往返：千分位整数、47 列宽度、0/1 缺失位完整还原。
- v2：显式 null 与缺键均归一为缺失单元，且保留可区分的原始标记。
- v3：分区头、实体名表、偏移表、float32 块往返一致，NaN 槽位与原始字节保留。
- v2→v1 sidecar 记录截断残差、精度标签与 `v2_to_v1_downgrade` 来源标记；
  v3 降级时分区结构/偏移表进入降级损失清单。

## 合并与诊断

- 容差 0.02、`source_version` 优先级下检出 5 个超容差冲突，288 条被抑制项，
  144 条来源追踪记录。
- 互操作报告：实体数三源均为 24；特征集合、缺失位置一致；最大绝对偏差
  4.954974947485556e-4（v1 千分位量化），全部位于容差内；
  `consistency_judgement = CONSISTENT`。
- 逐特征偏差直方图桶计数之和等于样本数；降级损失逐条可审计。

## 增量失效

| 场景 | reused | recomputed | invalidated |
| --- | ---: | ---: | ---: |
| 首次运行 | 0 | 576 | 0 |
| 无变化再次运行 | 576 | 0 | 0 |
| 替换 v2 两个单元 | 572 | 4 | 4 |
| 容差/优先级变更 | 0 | 576 | 576 |
| 特征退出白名单 | 0 | 480 | 576（含 96 条白名单裁剪） |

## 并发一致性

- `max_workers=1/2/4/8` 四次独立运行的 `merged_table.json`、
  `conflicts.json`、`failures.json` 经 JSON 比对完全相同。
- 合并行按 `(entity_id, feature)` 排序，冲突与失败清单顺序与单线程一致。

## 快照、信号与恢复

- 每次运行生成新全局版本并原子切换 current 指针；旧版本号换入被拒绝。
- `tools/signal_demo.py` 自发 SIGTERM 后进程退出码为 0，当前批次与
  manifest 完整提交。
- 模拟硬 kill 留下 staging 残留时，恢复点仍指向上一完整快照
  （`recoverable=true, reason=complete_snapshot`），不读取半成品。

## 业务码

`tools/validate_error_codes.py` 实测输出：

```text
config_missing_field       -> E_CONFIG_INVALID
precision_out_of_range     -> E_PRECISION_INVALID
fixed_width_overflow       -> E_LINE_OVERFLOW
non_finite_value           -> E_NONFINITE_INPUT
partition_offset           -> E_OFFSET_OUT_OF_RANGE
all five business codes observed
```

## 自动化测试

```text
.venv/bin/python -m unittest discover -s tests
Ran 27 tests in 0.9s
OK
```
