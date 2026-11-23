# 实际验证记录

## 环境

```text
OS: Linux (WSL Ubuntu 22.04)
Python: 3.11.0rc1
依赖: 仅标准库
线程池: 4（默认样本配置）
```

## 自动化测试

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 20 tests in 20.000s
OK
```

覆盖内容：

- 语料重复 ID、长度不一致、非法标签、非法片段、类型跳变；
- 模型维度、嵌入长度、转移矩阵形状；
- 命令类型、目标样本、调整权重字段；
- 特征哈希与发射打分重复运行逐位一致；
- 量化残差和舍入方向；
- 零分对称维特比的全路径并列与标签编码决断；
- `I` 无 `B`、`B` 无延续、`E` 无 `B`、未闭合片段修复；
- 类型候选并列时 `LOC < ORG < PER < TIME`；
- 边界等值、高一位宽参考分与 `decode` 翻转；
- 非命中词表增删、类型替换、语料仅标签替换的精确复用；
- 发射/转移/位宽/量化调整后的阶段失效边界；
- 每个稳定命令点与空白缓存全量计算的业务投影一致；
- 1 线程与 4 线程业务投影一致；
- 0 预算首响 `E_LATENCY_BUDGET_EXCEEDED` 与最终稳定；
- `NaN` 参数只在发射阶段输出 `E_EMISSION_INVALID`；
- 快照清单、硬链接复用、正常恢复、损坏快照回退、`SIGINT` 恢复。

## 默认命令流与全量对照

```bash
python3 main.py --workspace . --run-id final-check --verify-full \
  --report artifacts/validation-report.json --quiet
```

实测：

```text
global_version: 12
最终样本数: 17（s018 被命令删除）
实体片段数: 76
类型分布: LOC=14, ORG=50, PER=7, TIME=5
失败清单: []
首响: ready, 322.738 ms（预算 600 ms）
最慢命令响应: 351.569 ms
增量投影哈希: fc5bbee3a02869adc4a40f12a128296db524bbb4d72f37a32208a198aa3d5d92
全量投影哈希: fc5bbee3a02869adc4a40f12a128296db524bbb4d72f37a32208a198aa3d5d92
equivalence.passed: true
```

边界样例：

```text
sample: s001, span: 8..9
active:    4.50006 (scale 100000)
reference: 4.50006 (scale 1000000)
difference: 0.000000000000
alternate quantization level: 2
flip_stage: null
```

## 低预算首响

```bash
python3 main.py --workspace . --run-id final-latency --budget-ms 0 \
  --verify-full --report artifacts/validation-latency.json --quiet
```

首条命令实测：

```text
status: stale
codes: ["E_LATENCY_BUDGET_EXCEEDED"]
summary_version: 0
progress: 0/18
command elapsed: 69.481 ms
summary response_elapsed_ms: 66.338 ms
```

0 ms 是强制超预算路径的压力测试；解释器与线程调度本身不可能保证墙钟时间为 0。该场景验证了“不等待最新结果、立即给旧摘要和业务码”的语义。最终 `settle()` 后：

```text
equivalence.passed: true
E_LATENCY_BUDGET_EXCEEDED 次数: 12（12 条命令各一次）
```

## 中断与恢复

自动化测试执行：

```text
启动默认 12 命令流 -> 0.8s 后发送 SIGINT
进程返回码: 130
报告 exit_code: 130
完成当前命令后提交快照，manifest 含 completed_samples 与各阶段版本
```

随后：

```bash
python3 main.py --workspace <tmp> --run-id signal-test --resume --verify-full --quiet
```

结果：

```text
resumed_from_snapshot: true
resume_failure: null
equivalence.passed: true
```

另外验证了：

- `--max-commands 3` 停止后从快照恢复，继续执行第 4..12 条命令；
- 故意损坏 `CURRENT` 指针时返回 `E_SNAPSHOT_INVALID`，从输入安全重建；
- 未变化阶段文件在新快照代次中使用硬链接，`changed_stages=[]`。

## 非法输入

`tests/test_snapshot_cli.py` 使用局部输入文件验证：

```text
合法 1 行 + 非法 1 行语料 + 2 条非法命令
exit_code: 0（合法样本继续运行）
code_counts.E_CORPUS_INVALID >= 1
code_counts.E_COMMAND_INVALID >= 2
```

结构非法模型：

```text
feature_dim = 0
exit_code: 2
fatal: true
code_counts.E_MODEL_PARAM_INVALID >= 1
```

运行期非有限参数：

```text
emission_weights[0][0] = NaN
每个样本:
  code = E_EMISSION_INVALID
  entities = []
失败顺序 = s001, s002, s003
```

## 复现

全部结果可由以下命令重建：

```bash
python3 tools/generate_samples.py
python3 -m unittest discover -s tests -v
python3 main.py --workspace . --run-id rerun --verify-full --report artifacts/report.json --quiet
```
