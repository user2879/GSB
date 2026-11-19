# 触觉反馈指令流合成引擎 — 技术设计

## 1. 输入模型（单个确定性 JSON）

- `devices`：`device_id`、`sample_rate_hz`、`frame_duration_ms`、强度范围
  `min_intensity/max_intensity`、量化档位 `levels`、每帧最大指令数
  `max_commands_per_frame`。
- `events`：`event_id`、`event_type`（`envelope|texture|tap`）、`start_ms`、
  `duration_ms`、`target_device`、`curve` 控制点（毫秒时间 + 强度）。
- `sync_anchors`：锚点毫秒、参与设备、参考设备、允许偏差（采样点数）。
- `global`：`concurrency` 并发上限、`inflight_limit` 在途指令缓冲上限。
- `state_dir`、`output_file`：状态目录与正式指令流输出路径。

三类事件共用同一条“采样点域包络 → 量化 → 装箱”管线，差异仅在曲线形状。

## 2. 模块划分

- `parser.py`：读取并严格校验，非法输入抛 `HapticError`（`E_xxx`），
  校验在创建状态目录之前完成，因此非法输入不产生任何状态文件。
- `envelope.py`：毫秒到采样点的确定性映射、分段线性插值、确定量化、
  事件叠加。
- `packing.py`：按帧装箱、同强度连续采样点 RLE 合并、超限溢出拆分、
  逐采样等价展开与校验。
- `scheduler.py`：固定顺序准入的预算调度器，串行/并行共用同一规则。
- `state.py`：版本摘要、原子写（同目录临时文件 + `os.replace`）、
  旧版本归档、块摘要校验、中断标记。
- `sync.py`：同步锚点偏差诊断（采样点 + 毫秒双重表示）。
- `engine.py`：编排恢复、调度、最终摘要补算、原子导出与 JSON 报告。
- `cli.py`：命令行入口与退出码约定。

## 3. 采样点域合成（需求 2）

- 毫秒转采样点：`round_half_up(t_ms * rate / 1000)`，采用
  `floor(x + 0.5)` 而非 NumPy 的银行家舍入，保证跨平台确定。
- 事件区间为半开区间 `[start_sample, end_sample)`，`end >= start + 1`。
- 控制点之间使用 `np.interp` 分段线性；解析器保证曲线覆盖整个事件区间。
- 量化先把连续强度归一化到 `[0, levels-1]`，再半向上取整并夹取。
- 同设备重叠事件：连续强度相加（以设备最小值为零点），夹取到设备范围后
  再量化，与事件顺序无关。空闲采样点为最低级 0。
- 任意毫秒映射与真实采样点的误差不超过 0.5 个采样点，即满足
  “多设备时间轴对齐误差不超过 1 个该设备采样点”。

## 4. 帧装箱与溢出拆分（需求 3）

- 帧长（采样点数）= `round(rate * frame_ms / 1000)`，至少为 1。
- 先在整段逐采样量化数组上做 RLE，再在帧边界切开，得到绝对采样区间
  `(start_sample, end_sample)` 与 `duration_samples`。
- 每个物理帧槽最多容纳 `max_commands_per_frame` 条；超出的指令按
  FIFO 顺序进入后续“溢出帧块”，累计 `overflow_count`。
- 拆分只改变指令所在的帧槽，绝不改变任何指令的采样区间与强度，因此
  `expand_blocks` 重建的逐采样序列与逐点单独下发逐元素相同。
- 校验同时检查：每帧指令上限、持续时长一致、采样点全覆盖、逐采样相等。

## 5. 全局预算并行（需求 4）

- 调度单位是“设备的一个帧块”，准入键固定为 `(device_id 升序, block_index)`。
- 块启动前预留其指令条数，持久化完成后释放；`inflight_usage` 永不超过
  `inflight_limit`。预算不足时设备标识靠后的设备等待，禁止降精度、
  截短或丢指令。
- 解析器要求 `inflight_limit >= max(max_commands_per_frame)`，保证系统
  始终可推进、无死锁。
- 串行（`--serial` 或 `concurrency=1`）与并行使用同一份计划和同一准入
  规则，输出逐条相同；归并按设备标识与采样点起点排序。

## 6. 同步锚点诊断（需求 5）

- 锚点在每个设备网格上映射为理想采样点，定位覆盖该采样点的指令起点，
  偏差 = 指令起点 − 理想采样点，并换算毫秒。
- 任一设备偏差绝对值超过 `allowed_samples` 即抛 `E_SYNC`，但引擎仍导出
  全部合法指令流，并在报告 `sync.violations` 中指名设备与锚点。

## 7. 持久化、版本与恢复（需求 6/7/8）

- 版本标识 = 设备/事件/锚点规范化 JSON 的 SHA-256（与全局预算、路径无关）。
- 每个块写入 `state/blocks/<device>/block_xxxxxx.json`，同目录临时文件
  写后 `fsync` 再 `os.replace`，崩溃不会留下伪完成块。
- 启动时版本不一致：旧快照整体移动到 `state/archive/<version>/` 后重做；
  版本一致：仅补算缺失或失效块，复用已通过摘要校验的块。
- 块摘要同样为规范化 JSON 的 SHA-256；声明指令条数不符、摘要不符或
  文件损坏都判失效，删除并补算，报告 `digest_events` 记录 `E_DIGEST`
  补算事件；若补算后仍校验失败则抛 `E_DIGEST`（退出码 4）。
- 可处理信号 `SIGINT/SIGTERM` 只在帧块边界生效，写 `interrupt.json`
  （含累计次数与 `pending_resume`），分别以 130/143 退出；重启时清除
  续跑标记但保留累计次数。强制终止（如 SIGKILL）无中断标记，重启仅依赖
  已持久化且摘要有效的块恢复，结果与一次跑完逐条相同。

## 8. 输出

- `commands.json`：`version_id` 与按设备排序的指令流，每条含绝对采样
  区间、持续采样点数、量化级与反量化强度，原子替换写出。
- `commands.report.json` / stdout：各设备指令条数、块数、溢出计数、
  摘要失效与补算、对齐偏差、中断次数、在途预算峰值与等待次数。

## 9. 错误码

| 码 | 含义 | 退出码 |
| --- | --- | --- |
| `E_JSON` | 文件不可读或非合法 JSON | 2 |
| `E_FORMAT` | 结构/锚点/路径字段非法 | 2 |
| `E_DEVICE` | 设备字段非法或标识重复 | 2 |
| `E_EVENT` | 事件字段或目标设备非法 | 2 |
| `E_CURVE` | 控制点时序/范围/覆盖非法 | 2 |
| `E_GLOBAL` | 并发或预算配置非法 | 2 |
| `E_SYNC` | 同步锚点偏差越限 | 3 |
| `E_DIGEST` | 块摘要失效且无法修复 | 4 |
| `E_STATE` | 状态目录不可读等状态错误 | 5 |
