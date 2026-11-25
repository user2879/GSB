# 实际验证报告

验证日期：2026-11-26（Asia/Shanghai）

## 环境

- Linux / WSL Ubuntu
- Python `3.11.0rc1`
- 虚拟环境：`virtualenv -p /usr/bin/python3.11 .venv`（系统缺少 `ensurepip`，未使用容器）
- `numpy 2.4.6`
- `pytest 9.2.0`
- 运行时仅使用 Python 标准库

## 执行命令

```bash
python tools/generate_sample.py
pytest -q
python -m log_sampler verify \
  --config samples/config.json \
  --batches samples/batches.json \
  --injections samples/injections.json \
  --commands samples/commands.json \
  --state artifacts/verification --fresh
```

## 测试结果

```text
28 passed
```

完整验收报告：`artifacts/verification/verification.json`，状态 `ok`，16 项检查全部为 `true`。

## 核心数值

| 指标 | 实际结果 |
|---|---:|
| 桶数量 | 9 |
| 故障隔离失败数 | 4 |
| 注入编号覆盖 | INJ-001、INJ-002、INJ-003、INJ-004 |
| 基线最终样本 | 8 |
| 全命令序列后最终样本 | 5 |
| 全局预算收紧探针 | 8 -> 5，最终集合真实改变 |
| 候选记录数 | 22 |
| 证据行数 | 32 |

命令步骤共生成 9 个已提交版本。`cmd-01` 替换批次、`cmd-02` 删除日志、`cmd-03` 调整单桶容量、`cmd-04` 调整单桶优先级均只使 1 个桶失效并复用其余 8 个桶；`cmd-07` 切换交换版本时全部桶、最终样本与已有三版序列化产物均命中缓存。

## 逐字段一致性

- 增量最终结果与空白缓存全量重放：差异数 `0`。
- 进程在 `bucket:2` 后以退出码 `137` 强制终止；staging 清单记录 2 个已完成桶和 7 个待办桶。重启后 2 个桶以 `stage_resume_hit` 复用，最终逻辑结果与未终止重放差异数 `0`。
- 恢复后的四阶段产物版本与未终止重放完全一致：

```text
validation    11baf1038f12c1f65fff201709ac28492231bca374c08cc114f3d078454fb97a
buckets       f4323a80bc44489a2b7ff6bfe96cb1a39e33d9ed0c64ea743940f2bc7af34ec8
truncation    5c0ccbf0396c5eda4470ce3fc8319fef361871ea150859ed95f581bed154c8b8
serialization 817155f0635493e9670ab716edb1f1f88abbb7d87dd83931fdc4c90c9cf1f2a6
```

## 三版本互转

| 链路 | 结果 |
|---|---|
| p3 -> p1 -> p3 | 32 行，逐字段相等，旁路无错误 |
| p3 -> p2 -> p3 | 32 行，逐字段相等，旁路无错误 |
| p3 -> p2 -> p1 -> p3 | 逐字段相等，无 `E_SAMPLE_METADATA` |

p1/p2 降级产物及旁路元数据位于 `artifacts/verification/evidence-roundtrip/`。

CLI 冒烟同样通过：p3 -> p1 输出 32 行、32 槽；独立 `validate-evidence` 无错误；p1 -> p3 升回 32 行、32 槽且无错误；`compare` 对增量与重放输出 `status=match`、`difference_count=0`。

## 诊断与探针

最终诊断业务码：

```text
E_D_FLIP_RANDOM_SOURCE   随机源定点化越界
E_D_FLIP_TRUNCATION_TIE  截断相邻权重在两档位间翻转
```

- 容量探针：`edge-a|warning` 从容量 2 调整为 3，槽位成员由 `log-0021, log-0039` 变为 `log-0039, log-0012, log-0021`；其他 8 个桶历史逐字段不变。
- 预算探针：相同输入和档位下，全局预算 8 的最终集合与预算 5 不同。
- 注入异常均只隔离目标项，其余 27 条基线记录仍正常进入分桶与账本。

## 产物

- `artifacts/verification/incremental/versions/v00000009/`：最终增量版本、四阶段产物、三版证据、账本、诊断、逻辑结果。
- `artifacts/verification/replay/`：空白缓存全量重放。
- `artifacts/verification/crash/`：强杀与恢复状态。
- `artifacts/verification/evidence-roundtrip/`：互转中间产物。
- `artifacts/verification/verification.json`：机器可读验收结论。
