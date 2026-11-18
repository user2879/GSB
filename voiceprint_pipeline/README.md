# voiceprint-pipeline

确定性声纹嵌入提取、定点相似度比对与数值可信度诊断工具。

## 快速开始

```bash
# 依赖（根目录虚拟环境）
.venv/bin/pip install -r voiceprint_pipeline/requirements.txt
export PYTHONPATH=voiceprint_pipeline

# 1) 合成固定种子数据集（16kHz/mono/16-bit PCM）
.venv/bin/python -m voiceprint.cli synth \
  --input-dir data/samples --injection configs/injection_manifest.json

# 2) 完整推理 + 诊断报告
.venv/bin/python -m voiceprint.cli run \
  --input-dir data/samples --config configs/voiceprint_config.json \
  --pairs configs/speaker_pairs.json --injection configs/injection_manifest.json \
  --work-dir data/work --report data/report.json

# 3) 续跑（缓存命中，逐字段一致）
.venv/bin/python -m voiceprint.cli run --resume [同上参数]

# 4) 增量替换单个音频
.venv/bin/python -m voiceprint.cli incremental \
  --replace-audio /tmp/replacement.wav spk_01_utt_02.wav [同上参数]

# 测试
.venv/bin/python -m pytest voiceprint_pipeline/tests -q
```

设计细节、定点链路、误差裁决与快照语义见 `DESIGN.md`。
