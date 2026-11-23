# 语音情感证据链定点校准组件 — 技术说明

## 1. 目标与范围

组件面向“对话演进中声学证据链的定点校准精度与增量失效边界”诊断，提供可集成的
Python 3.11 包 `voicecalib`，覆盖：

1. 音频/脚本/证据摘要/配置的加载校验与业务码映射
2. 逐帧声学特征提取与初始证据强度构建（定点整数 + float64 参考值）
3. 逐轮定点校准核心（显式衔接权重、固定运算顺序、逐位可复现）
4. 说话人归属追踪与轨迹重建（字典序同分决断、空档断开、证据回溯）
5. 校准账本的精确失效与复用（内容指纹 + 参数指纹 + 位宽档位派生缓存键）
6. 单批片段预算与待办保留（`E_CALIB_BUDGET_PENDING`）
7. 线程池并发提取与串行结果逐字段一致
8. 阶段快照原子提交、恢复判定、异常注入隔离与诊断报告

不使用 Docker；仅依赖 NumPy、pytest，WAV/JSON 全部使用标准库 `wave`/`json`。

## 2. 目录结构

```
src/voicecalib/
  errors.py        业务错误码与失败清单（固定排序，与线程调度无关）
  fixedpoint.py    定点原语：半偶舍入、量化残差、饱和余量、定点乘加
  hashing.py       规范化 JSON + sha256 指纹
  config.py        校准配置/注入清单加载与枚举校验
  audio.py         RIFF/WAV 校验、16kHz/单声道/16bit PCM 读写
  features.py      25ms/10ms 分帧、自相关基频、能量包络、语速节奏
  evidence.py      三维特征归一化、加权原型距离、稳定 softmax、定点量化
  calibration.py   逐轮定点校准（gain/decay/衔接权重，整数运算）
  speakers.py      锚点距离归属、字典序同分决断、轨迹断开规则
  ledger.py        缓存键账本、分阶段失效判定、删除级联、持久化
  scheduler.py     单批预算、待办队列、批次账本
  concurrency.py   线程池封装与稳定排序键重排
  injection.py     四类故障注入与命中记录
  snapshot.py      临时目录暂存 + os.replace 原子提交 + 陈旧判定
  pipeline.py      主编排
  report.py        JSON/Markdown 诊断报告
generate_samples.py  固定种子样本生成器
run_demo.py          8 类验收场景
tests/               pytest 套件（22 用例）
```

## 3. 定点模型

- 位宽枚举 `frac_bits ∈ {8,12,16}`，量化档位 `quant_gear ∈ {64,256,1024}`。
- 档位决定饱和余量（headroom 1/2/4 个 1.0 单位），位宽决定标度 `2^frac_bits`。
- 舍入固定为半偶舍入（ROUND_HALF_EVEN），量化残差以 float64 逐分量保留；
  float64 仅作参考，不参与后续运算。
- 校准递推（全部为整数）：
  `cal_q = clamp(gain_q * evidence_q + link_q * decay_q * prev_cal_q)`，
  情感分量按情感标识字典序逐个计算；`link_q` 在同说话人/切换/首轮分别取
  `same_speaker_weight_q`、`transition_weight_q`、切换权重（首轮历史为零）。
- 无任何全局随机状态，不依赖数组归约默认顺序。

## 4. 缓存键与失效边界

缓存键由上游内容指纹、阶段参数指纹、定点位宽/档位派生：

- evidence：音频 sha256 + 证据参数指纹（初始权重、归一化、帧长帧移、位宽档位）
- calibration：evidence 缓存键 + 校准参数指纹（gain、decay、衔接权重、位宽档位）
  + 前驱校准键
- attribution：calibration 缓存键 + 锚点/容差指纹

失效规则（账本 `plan_round_status` 输出每轮每阶段动作与判定依据）：

- 替换某轮音频：该轮 evidence 及其下游 calibration/attribution 重算，上游全部命中
- 删除某轮音频：该轮及其下游缓存条目级联移除，上游保持命中
- 修改增益/位宽档位：evidence 仍命中，calibration 与 attribution 全量重算
- 配置整体哈希变化：阶段快照判定 `E_SNAPSHOT_STALE`，拒绝旧快照后冷启动

## 5. 预算与并发

- 调度器按轮次升序在失效轮次间分配 `batch_segment_budget`；
  预算耗尽返回 `pending`，待办轮次进入下一批且优先处理，错误码
  `E_CALIB_BUDGET_PENDING`；批次账本记录消耗、剩余、复用命中。
- 线程池并发执行帧特征与证据强度，`max_workers` 取自配置且不超过任务数；
  每个任务携带 `阶段|文件名|帧序号` 稳定排序键，结果收集后按键重排，
  因此并发度（1 vs N）与线程调度顺序不影响最终任何字段顺序。

## 6. 故障注入

独立 `injections.json`，四类真实故障，命中后写入 `injection_id`：

| 类型 | 阶段 | 业务码 |
| --- | --- | --- |
| nonfinite_sample | evidence | E_NONFINITE_SAMPLE |
| evidence_dim_mismatch | evidence | E_INJECTION_DIM_MISMATCH |
| zero_gain | calibration | E_ZERO_GAIN |
| replace_round_out_of_range | calibration | E_ROUND_OUT_OF_RANGE |

单轮失败进入失败清单并标记 blocked，其余轮次以最后成功前驱继续，结果互不污染。

## 7. 复现实验

```bash
python generate_samples.py --out samples          # 固定种子样本
python -m pytest -q                                # 22 个测试
python run_demo.py                                 # 8 场景验收 + 诊断报告
```

报告位于 `output/demo/<场景>/diagnostic_report.json|md`，
汇总位于 `output/demo/acceptance_summary.json`。
