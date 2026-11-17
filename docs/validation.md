# 实际验证记录

验证日期：2026-11-18（Asia/Shanghai）。

环境：

```text
Python 3.11.0rc1
NumPy 1.26.4
Linux/WSL2 x86_64
virtualenv: .venv
```

## 端到端命令

```bash
python tools/generate_samples.py
python -m rope_diag \
  --config samples/position_config.json \
  --samples samples/attention_samples.jsonl \
  --injections samples/injections.json \
  --state-dir .workbench_state \
  --report artifacts/diagnostic_report.json
```

最终报告记录：

- 3 种方案 × 9 条样本。
- 8 条注入命中，覆盖全部 4 个注入业务码。
- RoPE、线性内插、NTK 感知缩放均生成外推追踪记录。
- 训练长度边界样本 `s-boundary-001` 被归入 `in_train` 并标记 `boundary_marked=true`。
- 敏感性扫描包含 `base_zero`、`scale_nonpositive`、`dimension_truncation`、`scaling_overflow` 四类边界记录。

## 自动化测试

```bash
python -m unittest discover -s tests -v
```

结果：`Ran 12 tests ... OK`。

| 测试 | 验收点 |
| --- | --- |
| 三方案接口确定性 | 同方案重复编码逐元素相等；统一 query/key 形状。 |
| 严格配置校验 | 缺失基频/缩放因子、奇数维度返回 `E_CONFIG_INVALID`。 |
| 数值业务码 | 基频 0 与缩放因子非正分别返回 `E_BASE_ZERO`、`E_SCALE_INVALID`。 |
| 边界分层 | 距离等于训练长度归训练层并标记边界。 |
| 三阶段追踪 | 首个超差阶段优先；矛盾判定下后续偏差保留。 |
| 注入报告 | 四类注入共 8 个方案-样本命中，ID 与业务码正确。 |
| 敏感性排序 | 边界值不进入排序，同分按参数值升序。 |
| 冷/热缓存 | 首次全量重算，第二次全量复用；诊断摘要一致。 |
| 增量失效 | 训练长度变化只重算分类受影响样本的 scoring/tracing；基频变化使三阶段失效；单样本变化只重算该样本。 |
| 串行/并发 | 并发与串行诊断字段一致；相同并发配置冷运行报告逐字段一致。 |
| 快照恢复 | `encoding` 快照后续跑成功；配置变化后 `--resume` 输出拒绝错误。 |
| CLI 非法配置 | 奇数维度返回退出码 2，并写入 `E_CONFIG_INVALID` 失败清单。 |

## 补充确定性实验

两次使用不同全新状态目录并加 `--no-cache` 运行，生成的完整 JSON 报告字节级相同：

```text
SHA-256 64cf7cec5076f5e6fe4c6567dcfdb806944d2cc4ed3b7ff8bb3bf4f2e7bef6c5
```

使用 1500 条临时样本运行 CLI，并在进程运行中发送真实 `SIGTERM`：

```text
signal_exit_code=0
manifest_status=interrupted
completed_stages=['encoding']
```

进程完成了当前 `encoding` 阶段与原子快照后才退出，满足可处理信号退出码 0 的要求。

## 产物

- `artifacts/diagnostic_report.json`：最终诊断报告。
- `artifacts/validation_summary.json`：报告计数、哈希与测试结果摘要。
