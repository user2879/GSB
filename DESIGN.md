# 说话人音色分析组件技术说明

## 1. 环境与安装

目标环境为 Windows + Python 3.11（本仓库在 Linux/Python 3.11.0rc1 下完成等价验证，
代码仅使用标准库与 numpy，路径全部使用 `os.path`，可直接在 Windows 运行）。

```powershell
python --version
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install numpy==1.26.4
.venv\Scripts\python examples\selfcheck.py
```

若 Windows 上 `ensurepip` 不可用，可先安装 `virtualenv` 后执行
`virtualenv .venv`，其余步骤不变。

## 2. 工程结构

```
timbre_pipeline/
  errors.py       业务错误码与可隔离 PipelineError
  utils.py        规范化 JSON/SHA256、定点舍入、固定规则分位数、稳定排序键
  config.py       配置默认值、校验、全量/阶段配置指纹
  synthesize.py   固定种子样本、WAV/配置/注入清单/真值生成
  audio.py        RIFF 严格校验、帧能量、双门限迟滞切分
  features.py     自相关 F0（段锚点+八度过滤）、Burg LPC 共振峰、音节节奏
  quantize.py     逐维自适应量化模式、整数/重构向量、残差与冲突
  attach.py       加权相似度、阈值相等决断、标注冲突、连续段溢出
  aggregate.py    有效帧加权定点聚合、退化标记、差分摘要
  cache.py        缓存键、上游引用账本、split/merge/modify_feature 命令
  budget.py       首响预算执行器、ThreadPoolExecutor 并发上限、待办降级
  snapshot.py     临时文件 + os.replace 原子提交、版本清单与恢复判定
  inject.py       注入清单加载、段范围校验、按阶段路由
  pipeline.py     五阶段编排、精确失效、预算调度、快照与诊断报告
examples/
  selfcheck.py    任务前自检
  run_demo.py     端到端验收示例
  cli.py          命令行入口
tests/test_pipeline.py  20 个单元/集成测试
```

## 3. 五阶段契约

| 阶段 | 输入 | 产物 | 指纹/缓存键要素 |
| --- | --- | --- | --- |
| segment | WAV 采样 + 切分配置 | 起止采样、帧范围、内容指纹、切分轨迹 | 段 PCM SHA256 + segment 阶段配置哈希 |
| feature | 段采样 + feature 配置 | F0/共振峰/节奏统计、有效帧数、帧表 | 内容指纹 + feature 阶段配置哈希 |
| quantize | 段特征 + 量化模式 | 整数向量、重构向量、档位、步长、残差 | 特征指纹 + schema 指纹 |
| attach | 同音频量化向量 + 阈值 | 说话人归属、相似度、合并/切分、冲突 | 量化指纹列表 + attach 阶段配置哈希 |
| aggregate | 归属 + 有效帧权重 | 说话人整数向量、权重明细、退化/差分 | 归属指纹 + aggregate 阶段配置哈希 |

阶段配置指纹保证：改阈值不会使 feature 缓存失效，改位宽不会使 segment 缓存失效。

## 4. 确定性策略

- 样本合成只用固定种子 `numpy.random.default_rng(20240517)`，逐次运行位级一致。
- 分帧起点、帧序号累加、分位数插值、字典序列化全部固定顺序。
- F0 先收集段内候选周期，以强度加权直方图选最长周期主峰为锚点，
  再把偏离锚点 ±25% 的八度帧判为清辅音，消除 116/380Hz 翻转。
- 共振峰使用 Burg LPC + companion 矩阵求根，频率升序固定取前三。
- 聚合采用整数加权与半数向上舍入，不依赖浮点求和顺序。
- 并发只用于无共享写入的段任务，结果按稳定排序键回收；
  测试已验证线程池与单线程模式描述向量逐字段相同。

## 5. 量化与归属

- 每维在音频范围内构造 schema：自然步长吸收同说话人抖动，
  档位容量随动态范围自适应；显式加宽档位时步长细化，整数向量真实变化。
- 相似度为整数向量的归一化 L1 距离，音色五维权重 0.96，节奏三维 0.04。
- 相似度恰等于阈值：`merge` 规则合并，`split` 规则切分，均写 `boundary_sample`。
- 预测归属与真值标注冲突时保留预测并输出 conflict 记录，不静默覆盖。
- 连续同归属段超过 `max_consecutive_segments` 抛 `E_SEGMENT_OVERFLOW`。

## 6. 增量失效规则

- `split/merge`：失效受影响音频的 feature、quantize，再级联 attach、aggregate；
  段 ID 按音频内序号重排，内容指纹重新计算。
- `modify_feature`：只失效目标段 feature 及其 quantize，
  再通过引用图级联该音频 attach/aggregate；其他音频指纹保持不变。
- 位宽切换：只失效 quantize 及下游；阈值切换：只失效 attach/aggregate。
- 账本事件按 `seq` 记录 `invalidate/recompute/reuse`，
  未命中内容的段必须复用，测试断言复用与重算同时存在且顺序单调。

## 7. 首响预算、快照与注入

- `BudgetExecutor` 严格使用配置 `max_workers`，任务键为 `audio|s000000|stage`。
- 预算耗尽返回最近摘要与 `E_LATENCY_FALLBACK`，待办排序键持久化在失败清单。
- 每阶段先写同目录临时文件，再 `os.replace` 整体替换；
  manifest 记录全局版本、配置哈希、档位、命令游标、阶段版本。
- 启动时配置哈希变化 => cold_start；缺阶段文件 => resume_rebuild；
  一致 => resume，并恢复缓存账本与命令游标。
- 注入四类故障：非有限采样、维度不匹配、阈值越界、注入段越界，
  注入项编号 `inj-01..inj-04` 全部进入报告；单音频故障不阻断其他音频。

## 8. 验收命令

```powershell
.venv\Scripts\python examples\selfcheck.py
.venv\Scripts\python examples\run_demo.py
.venv\Scripts\python -m unittest tests.test_pipeline -v
```

## 9. 已验证结果（实际运行环境）

- 19+1 个 unittest 全部通过，覆盖切分、特征、量化、归属、聚合、
  增量逐字段一致、预算降级、并发上限、快照恢复与四类注入。
- 5 条正常 WAV 各稳定切出 5 个发音段；1 条坏 WAV 命中 `E_AUDIO_FORMAT`。
- 三位说话人 F0 中位数分别稳定在约 118/165/228Hz，
  同音频同说话人段聚合为 1 个说话人且无标注冲突。
- 增量重放与独立全量重放 `audio_reports` 规范化 JSON 完全相等。
- 极小预算（0.01ms）稳定产生 `E_LATENCY_FALLBACK` 与待办清单。
