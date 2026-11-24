# 说话人日志可集成组件（diariz）

基于固定种子合成 16kHz/单声道/16bit PCM 长音频，完成重叠分窗、定点声纹嵌入、
整数域增量聚类与帧级说话人分段，支持窗级精确缓存失效、两档位误差归因、
快照原子提交、并发一致性与故障注入隔离。仅依赖 `numpy`，无网络/无容器。

## 快速开始

```bash
python --version            # 需要 Python 3.10+
python -m pip install -r requirements.txt
python demo.py              # 一键端到端示例
python -m unittest discover -s tests -v
```

- 合成样本与配置生成在 `samples/`（仓库不预置音频）。
- 全部报告与快照输出在 `output/`，每次运行自动重建。
- 设计细节、缓存失效矩阵与验证结果见 `TECH_NOTES.md`。

## 命令行

```bash
python run_diariz.py synth --input samples --speakers 3
python run_diariz.py diarize --wav samples/synthetic_3spk.wav \
    --config samples/diar_config.json --output output/run
python run_diariz.py visualize --summary output/summary.json \
    --html output/timeline.html
```

## 模块职责

- `src/diariz/audioio.py` RIFF/WAV 严格校验与确定性读写
- `src/diariz/features.py` 重叠分窗、帧特征与确定性 float64 嵌入
- `src/diariz/quantize.py` 定点量化、残差与纯整数 Q16 相似度
- `src/diariz/clustering.py` 字典序增量聚类与簇合并
- `src/diariz/segmentation.py` 重叠帧余量仲裁与边界规则
- `src/diariz/cache.py` 内容指纹缓存键与复用/重算清单
- `src/diariz/errorprop.py` 两档位误差传播与翻转归因
- `src/diariz/snapshot.py` / `concurrency.py` / `faults.py` 快照、并发、故障注入
