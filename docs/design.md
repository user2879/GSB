# 技术设计

## 1. 总体结构

工作台分为六层，层间只通过不可变数据对象和显式结果对象交互：

| 模块 | 职责 |
|---|---|
| `model.py` | 基元、时间轴片段、组合层、效果图；确定性波形求值 |
| `quantization.py` | 振幅/频率编码、解码、误差上界、量化等价判断 |
| `codec.py` / `frames.py` | v1/v2 帧结构、CRC、迁移、内容哈希、批次摘要 |
| `compiler.py` | 微秒时间轴离散化、通道混合、帧记录和误差报告 |
| `dependency.py` / `engine.py` | 依赖图、量化感知更新、哈希裁剪失效闭包、账本 |
| `transport.py` / `checker.py` | 回环设备、预算降级、并发批次、逐帧回执核对 |
| `persistence.py` | generation 快照、原子 `CURRENT`、孤儿回收与恢复 |

`cli.py` 只编排这些模块，不包含协议或失效算法。

## 2. 效果图模型

`Primitive` 支持 `sine`、`pulse_train`、`envelope`，字段包括：

- `channels`: 一个或多个逻辑通道。
- `start_us`, `duration_us`: 微秒时间区间。
- `amplitude`, `frequency_hz`: 参数化振幅和频率。
- `phase_deg`, `duty_cycle`, `attack_us`, `release_us`: 波形形状参数。

`TimeSlice` 是时间轴片段，包含基元 ID 和混合模式；`CompositionLayer` 组合多个片段并按优先级覆盖。求值顺序固定：

1. 片段内按声明的基元顺序构造贡献。
2. `max` 取振幅最大者，`replace` 取最后贡献，`sum` 振幅截断到 1 并按振幅加权频率。
3. 层间按 `(priority, declaration_order)` 取最高层。

固定遍历顺序和 `round` 规则保证相同输入得到相同帧。

## 3. 确定性样本

`configs/samples.json` 保存固定种子列表和四个场景参数。每个样本使用 `random.Random(seed)` 的局部实例，不读取系统时间、环境随机源或网络。生成器输出 canonical JSON（键排序、紧凑分隔符、UTF-8、LF），图与真值分别提交到 `samples/effect_graphs.json` 和 `samples/truth_manifest.json`。

真值包含每个基元的 `channels`、`channel_count`、`start_us`、`end_us`、`duration_us`，以及期望帧数。

## 4. 帧格式

两种版本共享 17 字节头：

```text
magic[2] | version[1] | flags[1] | seq[u32] | timestamp_us[u64] | mask[u8]
```

随后按逻辑通道顺序排列：

- v2：每个活动通道 `amplitude_code[u16] + frequency_code[u16]`。
- v1：每个活动通道仅 `amplitude_code[u16]`。

尾部为对“头 + 通道体”的 CRC32。v2 通道位为 `0x01/0x02/0x04/0x08`；v1 为 `0x01/0x04/0x10/0x40`。所有多字节整数为大端。

`decode_frame` 对 magic、version、mask、长度、CRC 做逐项检查，失败抛出 `ProtocolFormatError`，包含：

```text
code = E_PROTO_FORMAT
frame_seq
field
detail
```

## 5. 协议迁移

固定迁移表：

| 逻辑通道 | v1 位 | v2 位 | v1→v2 频率 |
|---|---:|---:|---:|
| `left_grip` | `0x01` | `0x01` | 100 Hz |
| `right_grip` | `0x04` | `0x02` | 120 Hz |
| `left_trigger` | `0x10` | `0x04` | 140 Hz |
| `right_trigger` | `0x40` | `0x08` | 160 Hz |

v1→v2 解码后按上表补齐频率并设置 migrated flag；对已是 v2 的帧直接返回原对象，因此迁移幂等。v2→v1 必然丢弃频率字段，这是旧协议不可表达的信息；回程保留时间戳、振幅和序列号。诊断器不会把频率丢失误判为传输失败，而是报告协议字段差异。

## 6. 量化与编译

```text
amplitude_step = 1 / 65535
frequency_step = 2000 / 65535
```

编码采用 `floor(value / step + 0.5)`，避免 Python 银行家舍入带来的边界不确定。声明误差界为步长的一半。编译器对每帧、每通道记录源值、编码值、误差和批次最大误差；编译结果断言实际误差不超过声明界。

每帧时间戳为 `seq * frame_period_us`。时间轴从 0 到图时长，固定生成全部 tick；即使某帧通道掩码为 0 也保留，因此参数变化不会导致重新编号或序号缺口。

## 7. 依赖图

节点键：

- `primitive:<id>`
- `slice:<id>`
- `frame:<seq>`
- `batch:<first>-<last>`
- `protocol:current`

每条边包含 `trigger_keys`、`channels` 和可选 `protocol_key`。基元到片段/帧的边声明 `amplitude`、`frequency_hz`、`duration_us` 等参数键；协议节点到帧/摘要的边声明 `protocol.version`；帧到批次摘要的边声明 `content_hash`。

## 8. 增量更新算法

输入：当前 `EngineState`、单基元参数补丁、可选新协议版本。

1. 生成新图并比较原始参数。
2. 对 `amplitude`、`frequency_hz` 先按目标协议比较编码值；相同则记为 ignored change，不进入下游闭包。v1 下频率变化直接忽略，因为线上格式没有该字段。
3. 有效变更取基元旧/新时间区间覆盖的帧作为候选。协议切换候选为全部帧。
4. 对每个候选帧重新求值并生成内容哈希；只有哈希或协议版本改变才进入失效集合。无变化候选保留原帧记录。
5. 包含有效变更基元的片段进入失效账本；仅当帧哈希变化时对应批次摘要失效。
6. 构造记录：失效节点、复用节点、复用来源、版本戳、ignored changes。
7. `UpdateResult` 构造时断言 `invalidated_frames == recompiled_frames`。
8. 协议切换归档当前账本并新建目标版本账本；归档账本 `reusable=false`，恢复时也不会重新激活。

这种方法同时满足“不能整图重编”和“闭包清单等于实际重编帧集合”。依赖图仍用于结构检查和通用闭包查询，最终失效集合由依赖时间窗与内容哈希共同裁剪。

## 9. 并发与预算

固定降级优先级：

1. 内存预算：根据最大帧的 base64 消息估算缩小批次。
2. 时间预算：根据每帧估计耗时和批次开销缩小批次；预算低于一个开销时固定批次 1、工作线程 1。
3. 线程上限：工作线程不超过 `max_workers` 和批次数。

每个线程只接收唯一批次，帧序号在计划阶段一次性分配。结果完成顺序不参与拼接，所有回执按 `seq` 放入字典后排序。运行中若某批超过时间预算，后续波次降为单线程并记录 `time_budget_overrun`，不重发已经完成的批次。

## 10. 回执核对

核对器按期望帧序号升序遍历：

- 没有回执：`transport_missing`，加入 `pending_sequences`。
- 协议版本、时间戳、掩码或通道集合不同：`protocol_field_difference`。
- 编码值不同：`quantization_difference`，并标记是否在一个编码步内。

首个不一致按序号和字段稳定排序。批次摘要使用线上字节计算；仅摘要不一致而逐帧全部通过时结果仍为 `confirmed`，并设置 `digest_mismatch_only=true`。批次分组不同导致摘要不可直接比较时 `digest_consistent=null`，不会误报。

## 11. 持久化状态机

```text
UNREFERENCED generation -> write files -> fsync -> write COMMITTED.json
  -> atomic rename -> atomic CURRENT replacement -> COMMITTED
```

未完成 generation 永远不被 `CURRENT` 引用。恢复流程：

1. 读取 `CURRENT` 中的 generation 名。
2. 校验 `COMMITTED.json`、每个文件哈希、完整状态哈希和协议戳。
3. 只有全部通过才反序列化为 `EngineState`。
4. 比 `CURRENT` 旧的完整目录移入 `history/`，记录为 superseded。
5. 比 `CURRENT` 新但未被引用的完整目录、无效 CURRENT 目录、不完整目录和临时目录移入 `orphans/`，记录为 orphan。
6. 恢复日志原子写入 `state/recovery_log.json`。

`CURRENT` 缺失或损坏时不猜测历史目录；严格返回“无可恢复稳定状态”。这保证崩溃时只使用最近一次真正提交的快照。

## 12. Windows 兼容性

- 路径全部使用 `pathlib.Path`，不依赖 POSIX 特殊文件。
- 原子替换使用 `os.replace`；目录 fsync 在不支持时安全降级。
- 回环服务绑定 `127.0.0.1`，使用标准库 `socket`/`socketserver`。
- PowerShell 和 BAT 脚本负责 3.11 虚拟环境、NumPy 安装、样本生成、测试和验证。
- 强制终止入口使用 `os._exit(137)`，不依赖信号语义。
