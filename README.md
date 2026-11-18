# 中文文本校对与数值误差诊断应用

面向中文文本校对的离线、确定性、可交互专业应用：检测并修正错别字、重复词、
术语前后不一致与疑似漏字，同时诊断 float32/float64 打分链路的数值误差传播。
实现细节见 `TECHNICAL_NOTES.md`。

## 环境要求

- Python ≥ 3.10
- 第三方包：`numpy`、`pyyaml`
- 不使用容器，不访问网络或外部模型

## 快速开始

```bash
pip install numpy pyyaml
python3 scripts/generate_data.py
python3 -m proofread_app --base-dir . run --clean
```

输出：

- `output/corrected.jsonl`：逐样本修正文本与逐处编辑记录
- `output/diagnostics.json`：误差归因、双档位直方图、边界样本、复用/失效计数、
  指纹、快照版本与失败清单
- `state/snapshots/`：原子替换的阶段快照与清单
- `state/cache.json`：内容指纹缓存与规则引用索引

## 交互调参与回放

```bash
python3 -m proofread_app --base-dir . replay --commands config/commands.jsonl
python3 -m proofread_app --base-dir . recovery
python3 -m proofread_app --base-dir . clean
```

支持命令：`set_threshold`、`set_quant_levels`、`set_aggregation_weights`、
`add_confusion_pair`、`remove_confusion_pair`、`replace_sample`，命令流见
`config/commands.jsonl`，异常注入见 `config/fault_injection.json`。

## 测试

```bash
python3 -m unittest discover -s tests
```

21 个用例覆盖真值命中、定点量化、双档位分数、边界决策、重叠抑制、增量失效、
并发逐字段一致、快照恢复、配置/资源错误码与优雅退出。
