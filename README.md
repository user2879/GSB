# LLM 偏好对齐与奖励建模 · 本地确定性诊断工作台

面向本地确定性样本的偏好对齐与奖励建模可视化/诊断工作台，覆盖：偏好对与配置
加载校验、确定性特征哈希与定点成对打分、边界样本判定与批次汇总、基于内容指纹的
对级精确失效、预算/配额调度、阶段化快照恢复、并发与串行一致及异常注入隔离。

完整设计与验证结论见 [`docs/TECHNICAL_NOTES.md`](docs/TECHNICAL_NOTES.md)。

## 快速开始

```bash
pip install -r requirements.txt
python -m reward_workbench init --root .     # 生成 samples/ config/ commands/
python -m reward_workbench run               # 一次性全量运行
python -m reward_workbench replay            # 命令流回放 + 增量/全量逐字段等价
python -m reward_workbench verify            # 并发 vs 串行一致
python -m reward_workbench inject            # 四类异常注入
python -m unittest discover -s tests         # 17 个单元测试
python run_verification.py                  # 端到端验收脚本
```

## 输入与产物

- `samples/preference_pairs.jsonl`：固定种子、受控词表生成的偏好对（UTF-8、毫秒时间戳）。
- `samples/preference_pairs_invalid.jsonl`：用于 `E_PAIR_INVALID` 演示的非法样本。
- `config/reward_config.yaml`：特征/打分/量化/预算/并发/快照配置。
- `commands/command_stream.json`：替换/删除对、调权重/阈值、切档位/子词版本/批次/配额。
- `reports/diagnostics.json`、`reports/diagnostics.html`：可审计明细与可视化。
- `runtime/snapshots`、`runtime/tmp`：阶段快照与整体替换提交的临时目录。

## 业务错误码

`E_PAIR_INVALID`、`E_CONFIG_INVALID`、`E_TRAIN_BUDGET_EXHAUSTED`、
`E_SNAPSHOT_STALE`、`E_PAIR_NOT_FOUND`、`E_COMMAND_INVALID`，均进入确定性失败清单
而非中断整批。
