# 语音转换数据管线技术说明

## 1. 目标与范围

管线在单机、纯 CPU、仅依赖 `numpy==1.26.4` 的条件下，解决语音转换（Voice
Conversion）分段结果在 v1 定宽文本与 v2 JSON 两种交换版本之间流转时的：

- 格式互操作（统一解析、双向转换、旁路元数据可追溯）；
- 精度损失量化（千分比整数与浮点、整数帧与浮点帧边界）；
- 增量失效判定（内容指纹驱动的段级缓存键与复用账本）；
- 确定性（串行/并发、增量/全量逐字段、逐采样一致）。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `voice_conversion/config.py` | 声学配置、规范化 JSON、配置/PCM 指纹 |
| `voice_conversion/errors.py` | 稳定错误码与按排序键输出的失败清单 |
| `voice_conversion/model.py` | v1/v2 统一的 `Segment` 中间表示 |
| `voice_conversion/v1.py` | 定宽文本解析/渲染，`E_FIELD_TRUNCATED` |
| `voice_conversion/v2.py` | JSON 解析/渲染，码表校验与旁路元数据 |
| `voice_conversion/convert.py` | 双向转换、旁路 sidecar、精度损失统计 |
| `voice_conversion/dsp.py` | 切帧、F0/倒谱/梅尔/能量、定点映射、逐帧合成 |
| `voice_conversion/assembly.py` | 段间确定性线性交叉淡化与拼接 |
| `voice_conversion/ledger.py` | 段内容指纹与四阶段缓存键、复用账本 |
| `voice_conversion/snapshot.py` | 临时目录暂存 + 整阶段原子替换 + 清单门控 |
| `voice_conversion/executor.py` | 串行/受限线程池，按稳定排序键回收结果 |
| `voice_conversion/editor.py` | 编辑命令（版本切换、边界、删除/追加等） |
| `voice_conversion/pipeline.py` | 阶段编排、失效判定、诊断报告 |
| `voice_conversion/cli.py` | `run` / `convert` / `restore` 命令行 |
| `generate_samples.py` | 固定种子合成 WAV、两版交换文件与配置 |
| `run_demo.py` | 验收场景总装并落盘验证报告 |

## 3. 统一中间表示与解析

v1 字段宽度固定为 `6/8/8/6/6/2`（段号、起帧、止帧、千分比强度、音色码、
缺失位），合计 36 字符。解析按物理行号逐行处理：

- 行宽不足、数值字段非数字、千分比越界或缺失位非空 -> `E_FIELD_TRUNCATED`，
  失败项携带行号与字段名；
- 音色码不在配置码表 -> `E_UNKNOWN_TIMBRE`；
- v2 对象缺字段或类型不符 -> `E_V2_SCHEMA`，命名音色不在码表 ->
  `E_UNKNOWN_TIMBRE`。

失败不中断整体运行，进入 `FailureLog`，最终按
`(段号, 阶段, 行号, 字段, 错误码)` 稳定排序输出，与线程调度顺序无关。

## 4. 双向转换与精度可追溯

- **v1 -> v2**：`intensity = permille / 1000`，可精确表示；整数音色码按码表
  映射为命名标签；精度标签写 `exact-from-v1-permille`。
- **v2 -> v1**：浮点强度四舍五入回千分比，浮点帧边界取整回整数帧；截断残差、
  精度标签、被丢弃的 `passthrough` 全部写入 sidecar（`vc-sidecar-v1`）。
- **还原**：`restore_v1_with_sidecar` 用 sidecar 还原降级前的浮点边界、浮点
  强度、命名音色与旁路元数据，逐字段相等。
- 精度损失由 `precision_loss` 给出强度/起止帧的最大与平均绝对误差。

## 5. 定点映射与确定性合成

- 定点化：`to_fixed` 采用“半值远离零”舍入，`snap_grid` 按 `quant_levels`
  做对称网格量化，正负对称、无全局随机状态。
- 声学分析：自相关 F0（配置搜索区间与强度阈值）、加汉宁窗的倒谱、HTK 梅尔
  滤波器组能量、帧 RMS；统计量按帧序号升序显式累加。
- 音色映射：F0 比例平移、倒谱平移与随阶数弯曲、增益，全部在 Q 定点整数域
  乘加后再网格量化，保证可复算、可审计。
- 合成：每段仅读取自身源帧与自身映射参数；浊音段用谐波叠加，清音段用由
  段号播种的 PCG 白噪声；汉宁窗重叠相加后饱和到 int16。
- 总装：段在帧移网格上放置，衔接处按 `crossfade_ms` 做确定性线性交叉淡化。

## 6. 指纹、缓存键与失效判定

段内容指纹只包含影响计算的规范化字段（不含 `passthrough`，也不含仅作包装的
交换版本号）。四阶段缓存键分别为：

1. analysis：段内容指纹 + 源音频指纹 + 配置指纹 + 量化档位；
2. mapping：analysis 键 + 配置指纹 + 量化档位；
3. synthesis：mapping 键 + 配置指纹 + 量化档位；
4. assembly：有序段指纹集合 + synthesis 键集合 + 配置指纹。

因此：

- 改单段边界/强度：仅该段 analysis 起的下游键变化，其余段直接复用；
- 删段/追加：账本 `prune` 删除消失段，新段计算，存活段复用；
- 版本切换：规范化内容不变则键不变、全量复用；v2->v1 取整若改变数值，则
  仅受影响段失效；
- 量化档位变化：键内嵌档位，全部段的映射/合成精确失效并重新计算。

账本 `ledger.json` 记录每段三阶段键、引用的源帧集合与参数集合，并原子落盘；
`snapshot_commit_interval` 控制长任务的周期性检查点。

## 7. 快照与恢复门控

恢复门控只比较全局结构版本号、**结构配置哈希**（剔除 `quant_levels`）与源
音频指纹；任一不一致输出 `E_SNAPSHOT_MISMATCH` 并整体重建。量化档位属于计算
参数而非结构参数，其变化通过缓存键精确失效，不触发硬重建。交换文件内容变化
是常规编辑，通过段级键解析，不清空快照。

各阶段产物先写入 `snapshot/tmp/<stage>/`，再以整目录替换发布；`manifest.json`
以原子替换作为提交标记，记录全局版本、配置哈希、交换指纹、音频指纹与各阶段
产物版本，替换前比对版本而非逐文件覆盖。

## 8. 并发一致性

`executor.run_tasks` 将任务按 `(段号, 阶段名)` 排序，线程池大小取
`min(config.workers, 任务数)`，结果按同一稳定顺序回收。工作线程不共享可变
累加器，浮点累加顺序在任务内部固定，因此并发度、调度顺序、失败发生顺序都不
影响段顺序、PCM、缓存键与失败清单。`workers` 不参与内容指纹，调整执行宽度
不会改变任何产物哈希。

## 9. 验证与结果

- `python -m unittest discover -s tests`：18 个用例全部通过，覆盖解析错误码、
  双向转换与 sidecar 还原、定点舍入、串行/并发一致、增量/全量一致、单段失效、
  样本字节级可复现、快照失配重建。
- `python run_demo.py`：验收场景全部 `PASS`，报告落盘
  `artifacts/validation_report.json`。

## 10. Windows 运行说明

全部路径使用 `pathlib`，WAV 用标准库 `wave`，无符号链接/容器/GPU/网络依赖；
在 Windows PowerShell 下执行 `pip install numpy==1.26.4` 后，直接
`python generate_samples.py`、`python run_demo.py` 即可。
