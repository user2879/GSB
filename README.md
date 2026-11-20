# 流式 ASR 领域数据编辑与操作应用

面向语音识别流式转写的确定性数据编辑应用，解决「交互延迟」与「多次编辑后计算结果一致性」
的冲突。纯 Python + NumPy 实现，不使用容器，仅依赖本地确定性合成样本。

## 环境要求

- Python 3.10+（开发验证于 3.11）
- 标准库 `wave`、`json`、`threading`、`hashlib`
- NumPy 1.24+（见 `requirements.txt`）

```bash
python -m pip install -r requirements.txt
```

## 目录

- `asr_stream_editor/`：应用包（加载校验、特征、维特比、流式合并、稳定前缀、预算、
  并发、快照、引擎、全量参考、CLI）
- `samplegen/generate_samples.py`：固定种子样本与配置生成器
- `samples/`：合成 WAV、音素映射、声学参数、流式配置、命令流、异常注入示例
- `tests/`：25 个 pytest 用例，覆盖八项业务要求与异常/恢复路径
- `verify_equality.py`：端到端「增量编辑最终态 vs 一次性全量解码」逐字段核对
- `docs/DESIGN.md`：技术设计、不变量与需求映射

## 快速开始

```bash
# 1. 生成确定性样本（仓库已附带，可重复执行）
python -m samplegen.generate_samples --output samples

# 2. 全量测试
python -m pytest tests/ -q

# 3. 端到端一致性验收（逐字、逐帧、逐分数、冲突、失败顺序）
python verify_equality.py samples artifacts/verify

# 4. 交互式命令流（含预算内/超预算响应、分片替换、回退/重叠/稳定帧调整）
python -m asr_stream_editor run --input samples --output artifacts/demo \
    --commands samples/command_stream.json

# 5. 独立一次性全量参考解码
python -m asr_stream_editor full --input samples --output artifacts/ref \
    --overlap-ms 120 --stable-frames 4 --backoff-scale 0.5
```

## 交互命令

`set_overlap`、`set_stable_frames`、`set_backoff`、`replace_chunk`、`resume`、
`finalize`、`initial_decode`，示例见 `samples/command_stream.json`。预算内无法完成时
返回业务码 `E_REDECODE_BUDGET_PENDING` 的中间态，并可继续推进至最终结果。

## 诊断报告

`run_report.json` 包含整段转写、各分片稳定前缀、预算内/超预算响应次数、对齐冲突清单、
失败清单（分片号/阶段/业务码/异常类型）、各阶段耗时、快照全局版本与恢复点分片序号。

设计细节与需求-测试映射见 `docs/DESIGN.md`。
