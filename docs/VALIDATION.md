# 实际验证记录

## 环境

| 项目 | 值 |
| --- | --- |
| 操作系统 | Linux 6.18 / WSL2（代码与脚本仅使用跨平台标准库；Windows 命令见 README） |
| Python | 3.11.0rc1 |
| NumPy | 2.4.6（隔离安装于 `.deps`；系统默认 1.26.4 未参与验收） |
| 容器 | 未使用 |
| 网络服务/外部数据 | 未使用 |

NumPy 2.x 导入确认：

```text
2.4.6 /home/zsl/feedback/uidfc5909d3-default-253/.deps/numpy/__init__.py
```

## 自动化测试

```text
PYTHONPATH=src:.deps python -m pytest -q
....................                                                     [100%]
22 passed
```

覆盖项：配置/请求负例、三版本解析与显式缺失、字段宽度和非有限值、坏行继续、r3→r1→r3 旁路恢复、r1→r3 缺失说明、覆盖链裁决、参数版本裁决、位宽资源、七步增量/全量一致、快速/权威一致与差异字段、冷启动零预算短路、四类故障命中、快照续跑/游标/损坏、降级账本、真实回环子进程。

## 一键验收

```text
python scripts/run_acceptance.py
```

结果写入：

- `artifacts/verification/summary.json`
- `artifacts/verification/sample-ledger.json`
- `artifacts/verification/fault-ledger.json`
- `artifacts/verification/fault-diagnostics.json`
- `artifacts/verification/scenario/comparison.json`

关键结果（以最终生成文件为准）：

- 样本账本：9 条回执、5 个批次、6 个权威请求条目、0 失败；1 个旧版本批次失效、4 个批次复用。
- r-003：`coverage_ref`，权威批次 `b-alpha-002`，`b-alpha-001/r-003` 作为被覆盖投递保留。
- r-002：`param_version_desc_batch_id_asc`，权威批次 `b-alpha-001`，`b-alpha-legacy/r-002` 引用链保留。
- 七步场景：替换回执、两次新增批次、参数版本 4→5、批大小 3→2、归并规则修改；12 条回执、7 个批次、3 个失效批次、3 个复用批次。增量与空白全量账本版本完全相同，差异字段为空。
- 故障注入：4/4 命中，编号 `INJ-TRUNCATE-001`、`INJ-UNKNOWN-VERSION-001`、`INJ-NONFINITE-DURATION-001`、`INJ-DUPLICATE-RANK-001`；映射到 `E_RECEIPT_MALFORMED`、`E_UNKNOWN_PARAM_VERSION`、`E_DUPLICATE_RANK`。
- 预算：0 ms 返回 `partial`、`E_RESPONSE_BUDGET_EXCEEDED`、`0/6`；正常快速与权威视图共同范围字段完全一致。
- 转换：r3→r1 旁路保存 6 个字段；r1→r3 恢复 0 丢失；无旁路 r1→r3 明确列出 6 个缺失字段。
- 回环工作进程：启动 3 个配置内 `127.0.0.1` 子进程，烟雾调用返回 2 条 r3 回执；重复调用逐字段一致；停止成功。

## 定点位宽实测

批次 `b-alpha-001`（3 请求，32×64 模型）：

| 位宽 | fixed_macs | utilization_units | utilization_scale | 保留十进制位 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 15,826,944,000 | 88 | 10,000 | 4 |
| 24 | 4,051,697,664,000 | 8,832 | 1,000,000 | 6 |
| 32 | 1,037,234,601,984,000 | 883,200 | 100,000,000 | 8 |
| 48 | 67,976,206,875,623,424,000 | 8,832,000,000 | 1,000,000,000,000 | 12 |
| 64 | 4,454,888,693,800,856,715,264,000 | 88,320,000,000,000 | 10,000,000,000,000,000 | 16 |

每个位宽的整数 McA 与保留位数均真实变化，计算过程不含浮点。

## 限制说明

本次自动测试宿主是 WSL2，而不是原生 Windows 进程；代码避免 POSIX 专用路径逻辑，启停脚本使用 Python `subprocess` 和标准 socket。Windows 上应按 README 的 PowerShell 命令运行；回环端口由 `config.json` 的 `port_range` 限制，启动前不要求其他依赖。
