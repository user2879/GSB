# fablock：分块注意力算子本地可复现验证与异常注入实验环境

面向 FlashAttention 风格分块注意力算子的纯 CPU 验证环境：在无 GPU 的
Windows 工作站上对分块注意力核进行数值正确性验证、定点误差诊断与故障隔离。
核在**整数域**完成分块乘加与在线 softmax 归约，并与 float64 参考路径逐块
逐环节比对；支持突发查询块局部重算、内容指纹精确失效、预算调度、异常注入
隔离与阶段化快照恢复。全部样本由固定种子本地程序生成，不访问外部资源。

## 环境准备

- Windows 10 及以上或 Windows Server（本环境同样可在 Linux/macOS 运行，纯 Python 无平台依赖）
- Python 3.11 或 3.12
- `python -m pip install numpy pytest`（或 `python -m pip install -r requirements.txt`）
- 不使用 Docker 或其他容器环境

## 快速开始

```bash
# 1. 生成示例配置、注入清单与固定种子张量包（本地确定性样本）
python scripts/generate_samples.py

# 2. 端到端实验：3 个批次，输出可复现 JSON 报告与阶段化快照
python scripts/run_experiment.py --batches 3

# 3. 带异常注入清单的实验（独立快照目录避免续跑判定干扰）
python scripts/run_experiment.py --batches 2 \
    --injections configs/injection_manifest.json \
    --snapshots /tmp/snap_inj --reports /tmp/rep_inj

# 4. 测试套件（43 项）
python -m pytest tests/ -q
```

## 目录结构

```
configs/operator_config.json      示例算子配置（含快照提交间隔）
configs/injection_manifest.json   示例异常注入清单（7 项）
data/sample_pack.fatp             固定种子小端二进制张量包（Q/K/V）
fablock/                          核心包
  config.py                       配置加载与校验（允许枚举/范围）
  tensor_pack.py                  张量包解析与加载校验层
  fixedpoint.py                   定点量化、定点 exp、饱和
  kernel.py                       整数域分块乘加 + 在线 softmax 内核
  reference.py                    float64 参考路径（逐块三环节）
  compare.py                      逐块逐环节误差比对与超差标注
  cache.py                        内容指纹缓存键与精确失效判定
  scheduler.py                    预算/配额调度与线程池
  injection.py                    异常注入清单与真实故障施加
  snapshot.py                     阶段化快照（临时目录+整体替换）
  dataset.py                      固定种子样本与突发块逐批变化
  pipeline.py                     批次编排、故障隔离、报告产出
scripts/generate_samples.py       样本生成脚本
scripts/run_experiment.py         端到端实验入口
tests/                            pytest 测试套件（43 项）
reports/                          每批可复现 JSON 报告（运行时产出）
snapshots/                        阶段化快照（运行时产出）
```

## 关键设计

### 张量包格式（小端）
`magic "FATP" | u32 version | u32 meta_len | meta JSON | u32 n_tensors |`
逐张量：`u16 name_len | name | u32 ndim | ndim×u64 shape | u32 dtype | u64 nbytes | data`。
meta 记录种子、`pad_rule=zero` 补齐规则与各张量原始形状。形状描述与数据
长度不一致抛 `E_SHAPE_DATA_MISMATCH`。

### 整数域内核
- Q/K/V 按量化档位（8/16 位枚举）与定点小数位宽量化为整数；
- 打分按查询块×键块切块，键维度升序在声明位宽（32/64）整数缓冲中累加，
  溢出前饱和并记录溢出块对与累加位置（`E_ACCUMULATOR_OVERFLOW` 事件）；
- 在线 softmax 以数值稳定偏移逐块更新运行最大值与分母，定点 exp 为纯整数
  泰勒展开（范围约化到 [0, ln2)），同一输入与配置多次运行逐元素一致；
- 中间量标尺：打分/softmax 状态为 2^(2f)，最终输出为 2^f（f 为小数位宽）。

### float64 参考与逐块比对
参考路径以 float64 复算块内打分、运行最大值与分母更新、输出累加三个环节，
与定点主路径逐块逐元素求差，按配置容差标注超差块与首次超差环节。切换定点
位宽或量化档位会真实改变中间定点整数与超差统计（有测试锁定）。

### 内容指纹精确失效
三级缓存键 = 上游块内容指纹 + 算子参数指纹（含块长/位宽/容差/稳定偏移）+
量化档位：键块乘积（score）、在线 softmax 状态（state）、输出累加（out）。
输出块记录其引用的键块与状态集合。修改块长、定点位宽、容差或替换单块张量
数据时，仅受影响块及其下游失效，报告输出复用与重算清单。

### 突发查询块局部重算
突发查询块的查询数据按 `[seed, batch, block]` 确定性逐批变化；批次推进后仅
突发块及其键块乘积进入重算集合，稳定块产物直接复用并记录命中。突发清单变
更时，旧突发块保留最后一批数据并恢复稳定（产物复用），仅新增突发块失效。

### 预算调度
单批打分点数预算在查询块间稳定均分形成单块配额（向上取整）；块对打分任务
经线程池并发执行（峰值并发记录在报告且不超过配置上限）。单块需求超配额记
`E_NODE_BUDGET_EXHAUSTED`；全局预算耗尽时剩余任务保留待办并记
`E_POINT_BUDGET_PENDING`，下一批优先处理待办。

### 异常注入隔离
注入项对指定块施加真实输入故障（删张量、改头维度、置 NaN、压预算、越位宽、
块级故障），注入项编号写入报告并映射业务码。单块在加载/乘加/softmax 阶段
失败时记录块号、阶段名与业务码并继续处理其余块；失败块不计入输出汇总，
同批次其余块结果与无注入时逐元素一致（有测试锁定）。

### 阶段化快照
四个阶段（load_validate / block_score / softmax_reduce / output_summary）
产物先写 `snapshots/tmp/` 再 `os.replace` 整体替换到 `snapshots/current/`，
最后落清单（全局版本号、配置哈希、张量包指纹、量化档位、已完成批次游标、
各产物版本）。启动时按版本号与配置哈希判定续跑，不一致输出
`E_SNAPSHOT_VERSION_MISMATCH` 并重新初始化。退出信号（SIGINT/SIGTERM）
置位停止标记，当前批次完成并落盘后才停止。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| E_TENSOR_MISSING | 缺失必需张量 |
| E_SHAPE_DATA_MISMATCH | 形状描述与数据长度不一致 |
| E_PAD_RULE_VIOLATION | 序列长度未按块长补齐规则标注 |
| E_NONFINITE | 张量含非有限元素 |
| E_HEAD_DIM_MISMATCH | 头维度非正偶数或与配置不符 |
| E_FRAC_BITS_OUT_OF_RANGE | 定点小数位宽越界 |
| E_QUANT_LEVEL_INVALID | 量化档位/累加位宽不在允许枚举 |
| E_WORKERS_OUT_OF_RANGE | 线程并发数越界 |
| E_ACCUMULATOR_OVERFLOW | 整数累加缓冲溢出（饱和并记录） |
| E_NODE_BUDGET_EXHAUSTED | 单查询块超配额中止 |
| E_POINT_BUDGET_PENDING | 全局预算耗尽，任务转入待办 |
| E_INJECTED_FAULT | 注入的块级故障命中 |
| E_SNAPSHOT_VERSION_MISMATCH | 快照版本号或配置哈希不一致 |

## 实际验证结果（本仓库记录）

- `python -m pytest tests/ -q`：**43 passed**（覆盖加载校验、内核确定性、
  溢出饱和、参考比对、突发复用、缓存失效、预算调度、注入隔离、快照恢复、
  并发串行等价）。
- 默认样本（seed=20240501，Q/K/V 为 24×8，块长 8，frac=12，int16 档位）：
  定点主路径与 float64 参考逐块最大绝对差 score≈1.8e-4、
  softmax_state≈1.7e-3、output≈9.3e-4，均低于容差 0.05，超差块为 0。
- 突发块复用：批次 0 全量重算 15 项；批次 1 起仅突发查询块 1 重算 5 项
  （3 个键块乘积 + 状态 + 输出），稳定块 10 项全部命中复用。
- 异常注入：7 项注入全部命中并映射业务码；块级故障隔离后其余块输出与
  无注入运行逐元素一致。
- 快照恢复：游标续跑只执行剩余批次；篡改版本号或变更配置哈希均输出
  E_SNAPSHOT_VERSION_MISMATCH；停止请求在当前批次落盘后生效。
- 并发串行等价：workers=1 与 workers=4 的输出、比对统计与溢出记录完全一致。
