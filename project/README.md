# 本地合成工单升级仲裁工作台

这是一个完全离线、可确定性复现的工单升级仲裁项目。它从本地 JSONL 工单证据与规则包出发，完成输入校验、定点证据打分、时间衰减、互斥/支持关系扣减、升级裁决、三版本交换格式转换、增量缓存、并发执行、原子快照与诊断报告。

## 环境

- Python 3.11（实际验证为 3.11.0rc1）
- numpy 1.26.4（验收环境已离线可用；仲裁核心只使用 Python 标准库，避免浮点进入主路径）
- 无 Docker、无外部服务、无网络访问
- 实际验收运行在 Linux/WSL2。代码只使用 `ThreadPoolExecutor`、`json`、`struct`、`os.replace` 等跨平台标准库接口；Windows 专属行为需要在实际 Windows 主机上再次执行同一验收脚本。

## 快速开始

```bash
cd project
PYTHONPATH=. python3 -m arbitration_workbench generate --root .
PYTHONPATH=. python3 -m arbitration_workbench run --root . --fresh
PYTHONPATH=. python3 -m arbitration_workbench compare --root .
PYTHONPATH=. python3 scripts/run_acceptance.py
```

编辑示例（默认使用配置中的交互预算）：

```bash
PYTHONPATH=. python3 -m arbitration_workbench edit --root . \
  --edit '{"op":"set_threshold","value":"1.10"}'

PYTHONPATH=. python3 -m arbitration_workbench edit --root . \
  --edit '{"op":"replace_evidence","ticket_id":"T0010","evidence_seq":1,"record":{"timestamp":17600000001,"terms":["audit","replaced"]}}'
```

版本转换示例：

```bash
PYTHONPATH=. python3 -m arbitration_workbench convert --root . \
  --source samples/version_samples/arbitration.v2.json --source-version v2 \
  --target tmp/out.v1.txt --target-version v1
```

## 目录

```text
project/
  arbitration_workbench/      # 核心实现
  samples/                    # 固定种子确定性样本
    evidence.jsonl            # 48 条正常工单、211 条证据
    evidence_invalid.jsonl    # 独立坏样本集
    rules.json / config.json
    exception_injections.json
    version_samples/          # v1/v2/v3 初始交换产物
  outputs/                    # 三版本产物、诊断与验收结果
  snapshots/                  # 完整代快照与 CURRENT 原子指针
  tests/                      # 单元/集成测试
  scripts/run_acceptance.py   # 端到端验收
  TECHNICAL.md                # 算法与一致性协议
  VERIFICATION.md             # 最近一次实际验收记录
```

## 输入格式

证据 JSONL 每行：

```json
{"ticket_id":"T0001","evidence_seq":1,"evidence_type":"failed_login","source_role":"external","timestamp":1760010000,"terms":["otp","lockout"]}
```

规则包包含 `evidence_type_weights`、`role_trust`、`relations` 和 `priority_bands`。配置包含定点小数位、量化档位、半衰期、互斥扣减、融合方式、裁决阈值、交互预算、线程数、快照提交间隔和目标版本。

固定种子样本中 `T0047` 被构造成裁决分恰等于 `1.0` 阈值，用于验证边界决断。

## 输出

- `outputs/arbitration.v1.txt`：定宽文本行（票号、升级码、优先级档、千分比整数分、缺失位）
- `outputs/arbitration.v1.txt.meta.json`：v1 截断残差、精度、理由编码与不可表达字段旁路
- `outputs/arbitration.v2.json`：浮点裁决分、命名优先级、精度标签、理由编码
- `outputs/arbitration.v3.bin`：特征名表、绝对偏移表、int64 定点分块
- `outputs/version_conversion_report.json`：逐版本降级、`E_VERSION_LOSSY` 明细与往返一致性
- `outputs/diagnostic_report.json`：逐工单摘要、边界工单、失败清单、失效/复用、并发和快照信息
- `outputs/acceptance_results.json`：端到端验收账本
- `outputs/acceptance_diagnostic_report.json`：最终验收编辑态的独立诊断报告

## 错误码

输入失败码：`E_JSON_INVALID`、`E_EVIDENCE_TYPE_UNKNOWN`、`E_ROLE_UNKNOWN`、`E_TIMESTAMP_INVALID`、`E_DUPLICATE_EVIDENCE`。其他关键码：`E_STAGE_EXCEPTION`、`E_VERSION_LOSSY`、`E_SNAPSHOT_STALE`、`E_INTERACT_BUDGET_PENDING`、`E_SIDECAR_STALE`、`E_TEMP_VOLUME_MISMATCH`。

失败工单不会进入仲裁。单工单阶段异常会记录工单 ID、阶段、业务码与异常类型，其余工单继续执行。

## 验证

最近一次实际验收结果见 `VERIFICATION.md`。验收覆盖：

- 三版本精确往返、v3→v1 有损明细与旁路恢复
- 定点仲裁、互斥零下界、支持加成、阈值边界
- 单线程与四线程逐字段一致
- 四个阶段分别注入异常后的隔离
- 11 步编辑序列每次与空缓存全量重建一致
- 未受影响工单零重新计算
- 快照续跑全阶段复用、陈旧检测、预算等待与 SIGTERM 退出码 0
