# 验收记录

运行环境：WSL2 Ubuntu（Linux 6.18，x86_64），Python 3.11.0rc1，
NumPy 2.4.6，pytest 9.2.0；纯本地 CPU，无网络依赖、无真实模型权重。
代码仅使用标准库 + NumPy 与跨平台文件 API，同样适用于 Windows 原生 Python 3.11。

## 1. 自动化测试

命令：`python3.11 -m pytest tests/ -q`

结果：**33 passed in ~26s**，按 8 条业务规则分布：

| 测试文件 | 覆盖规则 | 用例数 | 结果 |
| --- | --- | --- | --- |
| test_01_loader_validation.py | 规则 1 加载/校验/错误码 | 5 | 全通过 |
| test_02_forward_determinism.py | 规则 2 前向顺序与逐元素一致 | 3 | 全通过 |
| test_03_lora_reproducibility.py | 规则 3 训练/验证可复现 | 3 | 全通过 |
| test_04_invalidation_granularity.py | 规则 4 双层失效粒度 | 5 | 全通过 |
| test_05_snapshot.py | 规则 5 快照/原子提交/对账 | 5 | 全通过 |
| test_06_incremental_vs_rebuild.py | 规则 6 增量=全量逐字段 | 2 | 全通过 |
| test_07_concurrency.py | 规则 7 并发/串行一致 | 3 | 全通过 |
| test_08_injection.py | 规则 8 注入隔离/原子报告 | 3 | 全通过 |
| test_09_edge_cases.py | 提交节奏/严格对账/并发不失效 | 4 | 全通过 |

## 2. 端到端演示（scripts/run_demo.py）

命令：`python3.11 scripts/run_demo.py`，11 项断言全部 PASS：

- 无变更重跑：4 个任务全部命中复用，`trained=[]`，无重新训练；
- 单样本编辑：仅所属任务 `cls_mod4` 失效，其余 3 任务复用；
- 单任务秩/缩放变更：仅 `reg_cycle` 失效（原因 `rank`、`alpha`）；
- 目标层白名单变更：4 个任务全部失效（原因 `target_layers`）；
- 基座版本替换：4 个任务全部失效（原因 `base_version`）；
- 每一步增量结果（适配器数组、逐轮损失、指标、失效/复用清单、失败清单）
  与从空白工作区的全量重建 `compare_outcomes` 逐字段相等；
- 并发度 1 与 4 的验证汇总、失效清单与任务摘要完全一致；
- 6 个异常注入项全部命中预期业务码，INJ-06 证明单任务故障被隔离、其余任务完成。

## 3. CLI 实测

- `lora-bench generate`：生成样本 JSONL、任务清单与 `base_v1/base_v2` 两套基座；
- `lora-bench run`：首次 4 任务训练，二次运行 4 任务全部复用；
- `lora-bench rebuild`：退出码 0，`comparison.equal=true`，无差异字段；
- `lora-bench inject`：退出码 0，`hit 6/6`，报告经临时文件原子替换落盘；
- 严格续跑配置不一致时返回 `E_CONFIG_MISMATCH`（见 test_05_snapshot）。

## 4. 关键失效粒度证据

| 编辑类型 | 失效范围 | 失效原因 |
| --- | --- | --- |
| 任务样本增删改 | 单任务 | `task_samples` |
| 任务秩/缩放系数 | 单任务 | `rank` / `alpha` |
| 目标层白名单 | 全部任务 | `target_layers` |
| 基座参数版本 | 全部任务 | `base_version` |
| 学习率/批量/轮数/指标 | 单任务（身份内字段） | 对应字段名 |
| 仅并发数/提交间隔 | 无（不进配置哈希） | — |

## 5. 复现实验

```bash
python3.11 -m pip install numpy pytest
python3.11 -m pytest tests/ -q
python3.11 scripts/run_demo.py
```

生成脚本与种子保留在 `lora_bench/generator.py`（默认主种子 20240501，
基座种子 1001/2002），重复执行产生字节级一致的输入与产物。
