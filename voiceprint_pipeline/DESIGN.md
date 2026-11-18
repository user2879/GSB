# 确定性声纹编码推理管线 — 技术说明

面向音色克隆场景的本地、确定性、可集成声纹编码与验证管线。无真实账号、无线上服务、
无外部模型权重；全部随机量由配置指纹与内容指纹派生的固定种子生成。

## 1. 环境与目录

- Python 3.11 + NumPy 2.2.6 + pytest 8.3.5，固定于 `voiceprint_pipeline/requirements.txt`
- 虚拟环境位于仓库根 `.venv`（系统无 ensurepip，使用 get-pip.py 引导后 `python -m venv --without-pip`）
- `voiceprint/` 为包源码，`tests/` 为 pytest 套件，`configs/` 为输入 JSON，`scripts/run_demo.sh` 一键演示

## 2. 数据流与模块映射

| 业务要求 | 模块 | 关键产物 |
| --- | --- | --- |
| 1 音频校验/逐帧特征 | `audio.py`, `features.py` | F0、梅尔能量、倒谱、能量 dB、尾帧标记、特征指纹 |
| 2 说话人统计汇聚 | `aggregation.py` | 均值/总体方差/分位数、有效帧集合、汇聚指纹 |
| 3 定点嵌入 | `fixedpoint.py`, `embedding.py` | 特征码、Qn 整数、整数投影累加、整数 L2 嵌入 |
| 4 相似度判定 | `similarity.py` | 整数点积/范数、Q30 余弦、阈值决断、边界/翻转 |
| 5 误差传播 | `error_tracking.py` | 四环节逐项差值、首次超差、上下游矛盾裁决 |
| 6 增量失效 | `cache.py`, `pipeline.py` | 分作用域缓存键、引用键、复用/重算清单 |
| 7 快照恢复 | `cache.py`, `pipeline.py` | 版本化目录、临时文件原子替换、清单指针 |
| 8 并发/隔离/报告 | `pipeline.py`, `reporting.py`, `cli.py` | 线程池、稳定排序键、失败隔离、JSON 报告 |

## 3. 定点与量化链路

1. 统计向量先经**确定性逐维标定**（`calibration_scales`，规则按特征名物理量纲 +
   配置种子抖动，不依赖运行时数据）归一化到 [-1,1]。
2. `quantize_symmetric` 以零为锚点映射到 `[-(L-1), L-1]` 整数码，再按 Qn 位宽放大；
   `quant_levels` 改变码集合，`fixed_point_bits` 改变整数尺度，二者真实影响中间整数。
3. 投影矩阵由 `(特征维度, embedding/quant 作用域指纹)` 派生的 PCG64 种子生成，行归一化。
4. 整数矩阵乘（int64）累加后按确定性 half-away-from-zero 舍入回 Qn，再做**整数 L2 归一化**，
   使不同位宽嵌入处于同一量纲。
5. 同时输出：基准位宽定点嵌入、高一位宽参考嵌入、纯浮点参考嵌入（不归一化量化路径，结果必然不同）。
6. 余弦相似度全程整数：保留范数平方、`isqrt` 合流后单次舍入到 Q30，保证结果落在 [-1,1]。

## 4. 误差传播与裁决

- 四个环节：`feature_quantization`、`statistic_aggregation`、`embedding_projection`、`similarity`
- 上游 float64 统计与位宽无关，故汇聚环节差值严格为 0；首次超差通常出现在特征量化（1 LSB 量级）
- 每环节给出逐元素绝对差值、max/mean/rms，避免单一标量掩盖结构差异
- 当嵌入级与相似度级超差结论矛盾时，显式记录 `adjudication`，以更下游的 similarity 为准并附理由

## 5. 缓存键与增量失效

- 特征键 = f(样本, PCM 容器指纹, feature 作用域哈希)
- 汇聚键 = f(特征键, stats 作用域)；嵌入键额外含 quant/embedding 作用域、位宽、量化档位
- 相似度键 = f(两侧嵌入键, similarity 作用域, 位宽)，并记录每对引用的样本键
- 替换音频 → 仅该样本三阶段 + 引用它的对失效；改阈值 → 仅相似度；改档位/位宽 → 嵌入+相似度
- 未失效产物直接复用，复用/重算清单按稳定键排序写入报告

## 6. 快照与恢复

- 每批产物先写入全新版本目录 `versions/vXXXXXXXX/<stage>/<key>.json`，再原子替换根清单
- 清单记录全局版本、配置哈希、注入清单哈希、输入指纹、已完成阶段、游标、各产物版本指针
- 恢复时校验配置/注入哈希，不一致输出 `E_CONFIG_MISMATCH`（退出码 2）并拒绝续跑
- 输入内容变化属合法增量，按缓存键自然失效；`--allow-config-change` 用于阈值/位宽/档位调整
- 旧版本回收只删除无清单引用的目录，杜绝悬空指针

## 7. 并发与异常隔离

- 线程池并发度取 `max_workers`；任务带 `(阶段|标识)` 稳定排序键
- 单样本/单对异常记录 item_id、stage、业务码、异常类型后继续其余项，失败清单按 sort_key 稳定排序
- SIGTERM/SIGINT 在阶段提交点生效，完成当前快照后退出码 0，可无损续跑
- 报告与所有快照均经临时文件 + `os.replace` 原子写盘

## 8. 业务码

| 码 | 触发条件 |
| --- | --- |
| E_AUDIO_FORMAT | 容器头/采样率/声道/位深/数据长度不合规 |
| E_INSUFFICIENT_FRAMES | 有效帧低于 min_valid_frames，保留部分统计摘要 |
| E_BOUNDARY_SENSITIVE | 同一对在两种位宽下判定翻转 |
| E_CONFIG_MISMATCH | 续跑时配置或注入清单哈希不一致 |
| E_PAIR_DEPENDENCY | 对依赖的样本嵌入失败 |
| E_SIGNAL_SHUTDOWN | 优雅退出信号（内部） |

## 9. 复现实验

```bash
pip install -r voiceprint_pipeline/requirements.txt
bash scripts/run_demo.sh
.venv/bin/python -m pytest voiceprint_pipeline/tests -q
```
