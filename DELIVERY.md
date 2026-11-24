# 交付说明

这是一个可在 Windows 10/11 + Python 3.11/3.12 上直接运行的稠密向量索引构建、增量调度与诊断示例。

## 快速验证

```powershell
py -3.11 -m pip install numpy PyYAML
py -3.11 run_validation.py --root sample_workspace
```

看到 `"overall_success": true` 即表示：

- 11 个自动化测试全部通过；
- 14 条业务命令的增量结果与空白全量结果逐字段一致；
- 9 个故障注入项全部命中预期业务码；
- 非法输入失败清单、三版本账本、预算待办和快照恢复均已验证。

详细操作见 `docs/USAGE.md`，需求对照见 `docs/REQUIREMENTS_TRACE.md`，实际运行结果见 `docs/VALIDATION_REPORT.md`。

## 主要入口

- `dense_index/sample.py`：固定种子样本生成器。
- `dense_index/corpus.py`：JSONL 语料与命令流校验。
- `dense_index/config.py`：YAML 配置校验和参数指纹。
- `dense_index/chunking.py`：确定性滑窗分块。
- `dense_index/features.py`：子词哈希与配置指纹投影矩阵。
- `dense_index/quant.py`：定点量化、符号分桶、相似度和边界诊断。
- `dense_index/receipts.py`：v1/v2/v3 回执、旁路转换和统一账本。
- `dense_index/engine.py`：缓存、失效判定、并发调度、召回和快照提交。
- `dense_index/replay.py`：命令流增量/全量等价对拍。
- `dense_index/diagnostics.py`：独立故障注入。
- `tests/test_pipeline.py`：自动化验收测试。
