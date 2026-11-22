# 本地确定性说话人分离管线与结果核验工具

面向**本地确定性合成语音样本**的说话人分离（speaker diarization）数据处理管线。
仅依赖 Python 3.10+ 标准库与 NumPy，无需 GPU、网络或外部音频文件；所有样本、
配置、真值与异常注入清单均由固定种子程序在本地生成。代码同时兼容 Windows /
Linux / macOS（路径与原子替换均使用跨平台 API）。

## 目录结构

```
speaker_diarization/
├── diar/
│   ├── codes.py         # 稳定业务码与管线异常
│   ├── util.py          # 稳定 JSON、内容指纹、定点量化原语
│   ├── config.py        # 配置 dataclass 与分阶段配置指纹
│   ├── synth.py         # 固定种子合成 WAV / 真值 / 配置 / 注入清单
│   ├── audio.py         # RIFF 容器严格校验与 WAV 读取
│   ├── injection.py     # 异常注入注册表（真实输入边界故障）
│   ├── features.py      # 切帧、能量、谱平坦度、倒谱系数与定点化
│   ├── segments.py      # 双门限迟滞切分、最短段过滤、重叠候选标注
│   ├── clusters.py      # 确定性凝聚聚类、重叠裁决、定点置信度
│   ├── verification.py  # 与真值的定量核验与分歧诊断
│   ├── cache.py         # 内容寻址缓存、原子提交、快照清单
│   ├── pipeline.py      # 四阶段编排、线程池、失败隔离与续跑
│   └── run_demo.py      # 端到端演示（全量/增量/恢复/注入）
└── tests/               # unittest 测试套件（18 个用例）
```

## 快速开始

```bash
# 1. 生成数据集并运行完整演示（合成→全量→编辑序列→基准全量→一致性比对）
python -m diar.run_demo demo_workspace

# 2. 运行测试
python -m unittest discover -s tests -v
```

运行后 `demo_workspace/` 下产生：

- `input/`：合成 WAV、`config.json`、`truth.json`、`injections.json`
- `output/cache/objects/`：内容寻址的阶段产物（先写 `tmp/` 再 `os.replace` 原子替换）
- `output/snapshot/manifest.json`：全局版本、配置/注入哈希、音频游标、阶段版本、复用台账
- `reports/01..10_*.json`：每次运行的完整诊断报告与一致性摘要

## 五阶段管线与缓存键

| 阶段 | 产物 | 缓存键派生来源 |
| --- | --- | --- |
| features | 逐帧能量/谱平坦度/倒谱（含定点整数与残差） | 音频内容指纹 + 特征阶段配置指纹 + 量化档位 |
| segments | 语音/静音区间、重叠候选、边界清单 | features 键 + 切分阶段配置指纹（含能量/谱平坦度门限） |
| clusters | 说话人簇与段→簇映射 | segments 键 + 聚类阶段配置指纹（含聚类阈值） |
| attribution | 重叠逐帧归属、说话人区间、定点置信度 | clusters 键 + 归属阶段配置指纹（含 overlap_rule） |
| verification | 边界/数量/重叠/校准指标与分歧诊断 | attribution 键 + 核验阶段配置指纹 + 真值指纹 |

缓存键为内容寻址哈希；未失效的产物直接从磁盘读取并在台账记录 `reuse`，
失效或新增产物才记录 `compute`，因此“只重算真正受影响的片段与下游”可被报告验证。

## 定点量化与阈值

- 倒谱特征按 `frac_bits` 档位量化：`q = round(x / 2^-frac_bits)`，并保留步长与残差。
- 聚类在整数空间用 L1 距离；阈值按浮点语义反量化：
  `threshold_q = round(cluster_distance_threshold * 2^frac_bits)`，
  因而切换档位时裁决语义保持一致，而内部整数与置信度数值真实改变。
- 区间置信度为定点整数，记录 `quant_step`、`frac_bits` 与残差累计。
- 距离**恰等于阈值**时按非重叠决断并写入 `boundary_records`。

## 重叠裁决规则

- `nearest`：每帧归到 L1 距离最近的候选簇，距离相同按簇标识字典序决断。
- `max_margin`：重叠帧优先归属进入重叠前的前一说话人簇（确定性、可与 nearest 区分）。
- 聚类同分时按簇内最小帧序号决断；裁决同距时按簇标识字典序决断。

## 业务码

| 业务码 | 触发阶段 | 注入故障 |
| --- | --- | --- |
| `E_AUDIO_FORMAT` | audio | RIFF/采样率/声道/位深不合规 |
| `E_AUDIO_TRUNCATED` | audio | 截断 WAV 字节，data 块长度不符 |
| `E_NONFINITE_SAMPLE` | features | 非有限采样值（NaN/Inf） |
| `E_THRESHOLD_ZERO` | clusters | 聚类阈值被置零 |
| `E_RANGE_VIOLATION` | segments | 替换区间超出音频帧范围 |
| `E_TRUTH_MISMATCH` | verification | 真值说话人数与音轨实际不符 |
| `E_SNAPSHOT_STALE` | snapshot | 快照配置/注入/目录哈希不一致且未授权编辑 |

单个音频失败只记录音频名、阶段、业务码与注入编号，不中断其余音频。

## 并发一致性

`max_workers` 控制线程池上限；任务携带由文件名、帧序号、区间序号派生的稳定
排序键，最终音频顺序、区间顺序与失败清单顺序与单线程按文件名执行完全一致
（`tests/test_end_to_end.py::ConcurrencyDeterminism` 以 1/2/4/8 线程逐字段比对）。
