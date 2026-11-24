# 输入输出速查

## 命令

```text
python -m routing_workbench generate-samples --samples samples
python -m routing_workbench run [--samples DIR] [--utterances JSONL] [--config YAML]
                                  [--commands JSON] [--vocabulary TXT] [--state DIR]
                                  [--report JSON] [--full] [--no-resume] [--expected-version N]
                                  [--batch-budget N] [--workers N] [--crash-after-commands N]
python -m routing_workbench verify [--samples DIR] [--workdir DIR] [--report JSON] [--skip-pytest]
python -m routing_workbench inject [--samples DIR] [--utterances JSONL] [--config YAML]
                                    [--injections YAML] [--vocabulary TXT] [--report JSON]
python -m routing_workbench snapshot-info --state DIR
```

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_ROUTE_INVALID` | 话语、配置、命令或单阶段输入不合法 |
| `E_ROUTE_BUDGET_PENDING` | 本批预算不足，话语保留待办 |
| `E_ROUTE_INJECTED_RUNTIME` | 指定话语阶段命中运行期异常注入 |
| `E_SNAPSHOT_STALE` | 快照版本/配置/前缀不一致，需全量重建 |
| `E_SNAPSHOT_IO` | 快照持久化 I/O 异常 |

## 稳定摘要范围

`business_digest` 覆盖：

- `results`；
- `failures`；
- `boundary_samples`；
- `pending_queue`；
- 本次运行命中的运行期注入编号。

不覆盖运行相对的 `cache` 状态、预算账本和快照版本；这些字段在报告中单独审计。

