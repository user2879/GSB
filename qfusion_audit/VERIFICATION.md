# 实际验证结果（VERIFICATION）

环境：Python 3.11（要求 3.10+），NumPy 1.26.4，无 Docker、无网络/账号依赖。
样本：固定种子 `SEED=20261123`，6 个请求、62 个候选物品、186 条候选流记录
（每物品 3 通道），特征维度 4，2 个分片。

## 测试套件

`python3 -m unittest discover -s tests`：**34 个用例全部通过**
（以 `-W error::ResourceWarning` 运行亦无告警）。

| 测试模块 | 覆盖要求 |
| --- | --- |
| test_fixedpoint.py | 定点舍入、步长随档位变化、残差/方向、确定性、裁剪、稳定键（要求 2/7） |
| test_loader_config.py | JSONL/配置校验、重复物品/名次、非法维度、单调非空集合、失败不中断（要求 1） |
| test_engine_audit.py | 归一化零分母、档位笛卡尔、逐阶段账本、环节归因、翻转/边界对单调性（要求 2/3/4/5） |
| test_scheduler_budget.py | 预算待办保留与优先、小批=足额结果、并发上限、来源记录（要求 6/7） |
| test_integration.py | 编辑序列增量=全量重建、串行=并发、快照失效/版本递增/清单、预算续跑字段一致（要求 7/8） |

## 关键实测数据（batch_budget=13，5 批续跑）

- 逐环节最大定点差值（公共单位 2^-12）：
  normalize=3104，channel_fusion=3840，feature_overlay=4096，clip=4096。
- 翻转候选累计 1488：同位宽跨档位 392、跨位宽 1096；
  对端候选全部可解析，并列决断触发 9 次。
- 边界对数量随档位收紧（gear 1→2→4）单调增加，且每个位宽都严格增长：

  | 位宽 | gear=1 | gear=2 | gear=4 |
  | --- | --- | --- | --- |
  | f=4  | 19 | 178 | 307 |
  | f=8  | 6  | 22  | 206 |
  | f=12 | 4  | 8   | 22  |

- 预算：总消耗点 62（=候选物品数），累计保留项 118，分 5 批 drain；
  续跑产物与一次性足额预算在 ledger/flips/boundary 上 canonical JSON 逐字节相同。
- 两次独立运行（清空快照后重跑）快照目录 `diff -r` 完全一致：字节级确定性确认。
- 阶段化提交：batch_budget=13 共 5 批、提交间隔 2 批，单次运行生成
  v1/v2/v3 三个阶段版本（提交于第 2/4/5 批），current 原子指向 v3，
  报告 `intermediate_manifests=[1,2]`。
- 切换到 `samples/config_versions/` 下不同配置哈希的副本时，启动校验输出
  `E_SNAPSHOT_STALE`，旧版本隔离至 `snapshot_commit/stale/` 并全量重建；
  同配置再次运行 global_version 在已有版本基础上继续递增。

## 复现命令

```bash
python3 -m qfusion.sample_generator
python3 -m unittest discover -s tests
python3 run_demo.py
python3 -m qfusion.cli --batch-budget 13 --report samples/audit_report.json
```

## 验收对照

- 跨档位量化账本：`samples/snapshot_commit/current/ledger/data.json`
- 翻转归因：`.../flips/data.json`
- 边界对诊断：`.../boundary/data.json`
- 审计报告：`samples/audit_report.json`（含逐环节最大定点差值、翻转数、
  边界对数、预算保留项、配置哈希、最高位宽参考名次）。
