# 语音情感与对话理解定点校准组件

面向“对话演进中声学证据链的定点校准精度与增量失效边界”诊断的可集成 Python 组件，
纯本地确定性运行，不使用容器、不访问外部网络。

## 环境准备

- Python 3.11，依赖见 `requirements.txt`（NumPy、pytest）
- WAV/JSON 仅使用标准库 `wave` / `json`

```bash
pip install -r requirements.txt
```

## 一分钟上手

```bash
# 1. 固定种子生成 16kHz 单声道 16bit PCM 样本（WAV + 脚本 + 证据 JSONL + 配置 + 注入清单）
python generate_samples.py --out samples

# 2. 运行 22 个 pytest 用例
python -m pytest -q

# 3. 运行 8 类验收场景并落盘诊断报告
python run_demo.py
```

诊断报告：`output/demo/<场景>/diagnostic_report.json|md`，
验收汇总：`output/demo/acceptance_summary.json`。

## 验收场景（run_demo.py）

| 场景 | 验证点 |
| --- | --- |
| 01_baseline | 全量证据链校准、情感轨迹、说话人轨迹、预算批次 |
| 02_determinism | 单线程/多线程/重复运行逐字段一致 |
| 03_incremental | 替换 R6 音频：R0–R5 命中复用，R6 及下游重算 |
| 04_budget | 预算=2 时待办保留、下批优先、最终不丢轮次 |
| 05_delete | 删除 R4：E_AUDIO_MISSING，R0–R3 命中，下游级联 |
| 06_snapshot | 快照恢复逐字段一致；配置变更触发 E_SNAPSHOT_STALE |
| 07_injection | 四类故障注入命中且单轮隔离 |
| 08_validation | 畸形 RIFF、非有限证据、维度缺失的业务码映射 |

## 关键设计

- **定点模型**：位宽 `{8,12,16}` × 档位 `{64,256,1024}`，半偶舍入，
  校准全程整数运算，float64 仅作参考并记录量化残差。
- **确定性**：轮次升序、情感/说话人字典序、维度升序的固定运算顺序，
  无全局随机状态；并发结果按稳定排序键重排。
- **失效账本**：缓存键 = 上游内容指纹 + 阶段参数指纹 + 位宽档位；
  校准链失效向下游级联，证据各轮独立。
- **快照原子提交**：临时目录暂存后 `os.replace` 整体替换，
  清单含版本、配置哈希、注入清单哈希、游标、位宽档位。

详见 [TECH_NOTES.md](TECH_NOTES.md)。

## 集成接口

```python
from voicecalib import load_inputs, run_pipeline, build_report, write_report

inputs = load_inputs("samples")
result = run_pipeline(inputs, "output/run", resume=True)
write_report(build_report(result, inputs.config, "samples", "integration"),
             "output/run")
```
