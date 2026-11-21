# 特征漂移监控组件技术说明

## 1. 定位与边界

`drift_monitor` 是一个可集成的表格模型线上分窗漂移诊断组件。输入为本地 YAML
监控配置与确定性生成的参考/当前时段数据，输出定点判定结果、量化翻转诊断、
增量复用统计、失败清单与 JSON/可视化报告。全程不依赖网络、容器或外部数据集。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `config.py` | YAML 加载、数据结构定义、字段级校验，非法配置抛 `E_CONFIG_INVALID` |
| `datagen.py` | 固定种子派生独立 RNG 流，生成数值/类别/高基数列与显式缺失掩码 |
| `windows.py` | 固定 base + 余数末窗切片，短尾并入或单独成窗并打标 |
| `metrics.py` | 均值偏移、直方图密度、近似/精确分位数、频率差、卡方，确定性累加 |
| `strategy.py` | 预置四档策略表，按规模指标评分并真实改变计算路径 |
| `fixedpoint.py` | 定点整数量化、阈值量化、显著性判定、档位重缩放 |
| `flip.py` | 相邻档位逐环节（量化/阈值/判定）翻转诊断与高一位宽参考分 |
| `cache.py` | 由上游内容指纹 + 参数指纹派生的分阶段缓存键 |
| `snapshot.py` | 临时目录写入 + 整体替换提交、版本清单、续跑/陈旧判定 |
| `engine.py` | 线程池并发、规范排序键、预算扣减、失效/复用集合 |
| `runner.py` | 端到端编排与 JSON 报告组装 |
| `visualize.py` | 零依赖确定性 SVG 热力图 |
| `interactive.py` | 交互式 REPL 与脚本化编辑序列 |

## 3. 关键设计决策

### 3.1 确定性数据生成
- 主种子 + 周期标签 + 列序号派生每列独立 `numpy.random.Generator`，列增减不改变其他列。
- 缺失值以独立缺失 RNG 生成布尔掩码，数据槽位置为 `None`，报告区分观测数与缺失数。
- 规模档通过 `feature_multiplier` 复制非高基数特征，样本量与乘法器均要求单调。

### 3.2 分窗与度量
- 前 `k-1` 窗使用固定 `floor(n/k)`，余数集中末窗；余数小于 base 一半时并入前窗
  （`merged_short`），其余短尾标 `standalone_short`。
- 数值：标准化均值偏移、共享参考网格的直方图密度差、分位数距离（近似/精确两条路径）。
- 类别：频率总变差（TV）距离、按自由度归一化的卡方统计。
- 累加顺序固定为窗序号升序、样本序号升序；类别桶按取值字典序输出。

### 3.3 定点判定与翻转
- 每个档位含小数位宽与整数 clamp，`quantize` 使用远离零的四舍五入，判定只比较整数。
- 切换档位真实改变整数值与保留位宽，而非显示格式。
- 相邻档位记录量化整数差、容差、首次超差环节、翻转标记，并额外保留高两位宽参考分。
- 判定翻转优先记录 `judgement` 环节，其次 `quantize`、`threshold`。

### 3.4 自适应策略
- 指标：样本量、列数、高基数比例、缺失率、请求窗数。
- 四档策略 `tiny_exact / standard_mixed / large_approx / huge_stream`，分段带分 +
  连续压力项评分，平局按预置表顺序决断。
- 策略输出窗数上限与近似分位开关，直接改变 `metrics` 调用路径；近似与精确分位的
  中间值在偏态数据上可观测差异（测试覆盖）。

### 3.5 指纹、缓存与增量
- 所有键为 `sha256(canonical(上游指纹 + 参数指纹))` 截断，canonical 序列化强制排序键。
- 阶段：`load / slice / metric / decision / report / visual`。
- 报告同时给出：
  - `required_keys`：当前配置逻辑上必需的键，与运行历史无关，增量与全量必然一致；
  - `expected_invalidated_keys / expected_reusable_keys`：相对当前物理缓存的失效判定；
  - `reused_keys / recomputed_keys`：本次真实命中与重算清单。
- 修改单列度量只改变该列各窗的 metric 键及其下游，未受影响列直接命中缓存。

### 3.6 并发一致性
- 线程池大小取 `min(配置并发, 任务数)`，绝不超过配置值。
- 每个列窗任务带 `列名::w窗序号` 稳定排序键，结果按提交前的规范顺序（列名字典序、
  窗序号升序）回收；预算扣减序列与失败清单顺序在任意并发度下与单线程一致。

### 3.7 快照整体替换
- 产物先写入 `snapshot/tmp/snap_*`，再 `os.replace` 整体替换 `current`，旧版本备份后清理。
- 清单记录全局版本、配置哈希、数据指纹与各阶段版本。
- 提交前比较配置哈希、数据指纹与阶段版本，三者均不变则不递增版本（版本门控）。
- 续跑校验配置/数据指纹，不匹配返回 `E_SNAPSHOT_STALE`，清单损坏返回
  `E_SNAPSHOT_CORRUPT`。

## 4. 错误码

| 代码 | 触发场景 |
| --- | --- |
| `E_CONFIG_INVALID` | YAML 字段缺失、列名重复、目标列混入特征、档位非单调、窗数非法、量化位宽非法等 |
| `E_DATA_GEN_FAILED` | 数据生成异常或展开列名冲突 |
| `E_SNAPSHOT_STALE` | 续跑时配置哈希或数据指纹不匹配 |
| `E_SNAPSHOT_CORRUPT` | 快照清单无法解析 |
| `E_METRIC_FAILED` | 列窗度量任务失败（含注入失败），进入失败清单而非中断整体运行 |

## 5. Windows 可执行性
- 仅使用标准库 + NumPy + PyYAML，路径全部通过 `pathlib` 处理。
- 原子替换使用 `os.replace`（Windows/Linux 均支持目录原子替换语义）。
- 提供 `run_windows.ps1` 引导脚本，在 PowerShell 下创建 venv、安装依赖并运行测试与演示。

## 6. 交互命令
`run`、`resume`、`set-windows`、`set-threshold`、`set-tier`、`set-metric`、
`unset-metric`、`set-size-tier`、`set-concurrency`、`set-approx`、`set-window-cap`、
`set-fail-keys`、`show`、`history`、`exit`。
