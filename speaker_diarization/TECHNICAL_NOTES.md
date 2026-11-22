# 技术说明与实测结果
## 运行环境
- Python 3.11.0rc1，NumPy 1.26.4，无 GPU/网络依赖。
- 默认样本 6 段 4 秒 WAV（5 段分别注入故障，1 段干净），演示另增 1 段干净音频。
- 管线划分为五个可缓存阶段：features / segments / clusters / attribution / verification。

## 一、分离与真值核验（最终全量基准）
| 音频 | 说话人偏差 | 重叠帧正确率 | 边界偏差(ms) | 置信度校准偏差 | 容差内 | 分歧诊断 |
| --- | --- | --- | --- | --- | --- | --- |
| mix_006.wav | 0 | 1.0 | 10.5 | 0.039696 | True | - |
| mix_007.wav | 0 | 0.875 | 29.5 | 0.060279 | True | - |

## 二、精确失效与复用（编辑序列）
| 编辑动作 | 重算产物数 | 复用产物数 |
| --- | --- | --- |
| 无改动续跑 | 0 | 12 |
| 替换单段音频 | 5 | 7 |
| 调整能量门限 | 8 | 4 |
| 调整聚类阈值 | 8 | 4 |
| 切换重叠规则 | 3 | 9 |
| 切换量化档位 10bit→8bit | 12 | 0 |
| 新增一段音频 | 5 | 12 |

关键结论：
- 无改动续跑零重算，全部命中持久化缓存。
- 替换单段音频仅重算该音频五阶段，其余全部复用。
- 能量/聚类阈值变更只影响切分及下游，特征阶段复用。
- 重叠规则变更只令 attribution 与 verification 失效，clusters 复用。
- 量化档位切换使全部整数阶段失效重算，复用为 0。
- 编辑序列增量结果与空白工作区一次性全量结果逐字段一致：`True`（连续多次运行稳定）。

## 三、重叠规则切换
- `mix_006.wav` 在 nearest 与 max_margin 间有 130 个重叠帧归属确定性改变。

## 四、异常注入业务码命中
| 音频 | 阶段 | 业务码 | 注入编号 |
| --- | --- | --- | --- |
| mix_001.wav | audio | E_AUDIO_TRUNCATED | INJ-001 |
| mix_002.wav | features | E_NONFINITE_SAMPLE | INJ-002 |
| mix_003.wav | clusters | E_THRESHOLD_ZERO | INJ-003 |
| mix_004.wav | segments | E_RANGE_VIOLATION | INJ-004 |
| mix_005.wav | verification | E_TRUTH_MISMATCH | INJ-005 |

## 五、快照与并发
- 清单全局版本号 12，阶段版本 {'attribution': 1, 'clusters': 1, 'features': 1, 'segments': 1, 'verification': 1}，完成游标 6 段；产物经临时目录 `os.replace` 原子提交。
- 配置/注入/目录哈希不一致且未授权编辑时启动即报 `E_SNAPSHOT_STALE`；授权编辑后仅重算失效部分。
- 缓存与台账均加锁；1/2/4/8 线程的音频顺序、区间顺序、失败清单与业务结果逐字段一致（测试覆盖）。

## 六、测试
`python -m unittest discover -s tests -v` 共 18 个用例，覆盖 RIFF 校验、非有限值识别、双门限切分、等距边界、聚类同分、阈值为零、定点档位、缓存键派生、选择性失效、快照 stale、注入业务码、增量全量一致与并发确定性。
