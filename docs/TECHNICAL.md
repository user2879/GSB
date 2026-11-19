# 技术说明

## 1. 场景与通道模型

房间为轴对齐长方体，默认 `6 m × 5 m × 3 m`，声速 `343 m/s`，采样率固定 48 kHz，RIR 时长 0.45 s（21600 帧）。4 个通道是带方向支撑范围的接收胶囊：

| 通道 | 声源 | 方位 | 仰角 | 路径接收孔径 | 球谐/瓦片支撑半径 |
|---|---|---|---|---|---|
| 0 | s0 | 0° | +10° | 68° | 72° |
| 1 | s0 | 90° | +10° | 68° | 72° |
| 2 | s1 | 180° | -10° | 68° | 72° |
| 3 | s1 | 270° | -10° | 68° | 72° |

因此 `s0` 只改变通道 0/1，`s1` 只改变通道 2/3。每通道包含直达路径与可接收的一阶镜像反射路径；路径元数据记录墙面、距离、时延、到达方向、反射点、量化幅值、峰值和到达时间。固定噪声种子产生确定性衰减尾音。PCM 量化为 int16；所有计算和量化顺序固定，因此相同场景的不同构建逐样本一致。

## 2. 图数据结构

状态文件 `state.json` 中每个节点至少包含：

```json
{
  "id": "energy:r0c0",
  "type": "energy",
  "artifact_hash": "sha256(binary artifact)",
  "content_hash": "sha256(canonical payload + artifact)",
  "payload": {"tile": {}, "shape": [32, 32], "support_channels": []},
  "dependencies": ["sh:c0", "sh:c2"],
  "input_hashes": {"sh:c0": "...", "sh:c2": "..."},
  "param_stamps": {"grid.width": 1, "...": 1},
  "trigger_keys": ["receiver.sample_density", "reconstruction.order", "..."],
  "computed_revision": 1,
  "last_reuse_source_revision": null
}
```

边：

```json
{"source": "sh:c0", "target": "energy:r0c0", "kind": "artifact", "trigger_keys": ["..."]}
{"source": "param:wall.xmax.alpha", "target": "rir:c0", "kind": "parameter", "trigger_keys": ["wall.xmax.alpha"]}
```

`artifact` 边参与失效闭包传播；`parameter` 边只用于诊断视图和参数影响解释。

## 3. 精确失效算法

输入为当前 `scene`、`versions` 与旧 `state`。

1. 对每个预期节点，比较每个 `param_stamps[key]` 与 `versions[key]`。
2. 若节点缺失或版本不同，加入种子集合。
3. 若旧状态有 `scene`，同时比较参数字典值；这使外部只修改参数值而漏写版本戳时仍能防御性失效。
4. 依据旧图的 `artifact` 邻接表做 BFS；闭包之外不再访问。
5. 按固定层序重算：RIR→SH→能量块→瓦片。
6. 未失效节点验证产物哈希与上游输入哈希后直接复用，记录 `reused_from_revision`。
7. 新产物写成不可变 blob；全部完成并校验后，单次原子替换 `state.json`。

闭包报告逐节点给出 `parameter_change`、`upstream_invalidation`、`missing_cached_node`、`initial_build` 或 `cache_hit` 原因。

## 4. RIR 合成

直达路径：

```text
delay = distance / 343
amplitude = 0.90 / (1 + distance)
```

一阶反射：

```text
image_source = mirror(source, wall)
reflection_point = line(image_source, listener) ∩ wall plane
delay = |listener - image_source| / 343
amplitude = 0.62 / (1 + distance) * sqrt(1 - alpha_wall)
```

反射点必须落在墙矩形内，且接收方向与通道前向夹角不超过孔径。墙面吸声只进入通过该墙的通道路径，这是 `RIR→SH` 定向失效的物理依据。

所有路径以整数采样索引叠加，尾音由固定 `default_rng(seed * 1009 + channel * 9176 + 17)` 生成。峰值经统一缩放后量化为 int16，因此增量复用通道与全量重建通道字节一致。最终装配为 4 通道交错 WAV，并在写出后解码逐样本校验。

## 5. 球谐拟合

实球谐基使用伴随 Legendre 多项式与标准归一化，阶数 `l`、秩 `m` 共 `(l+1)^2` 个系数。采样方向由 Fibonacci 球面生成，覆盖声明为 `full sphere, deterministic Fibonacci sampling`。

对路径 `p`，观测能量为：

```text
E_p = q_p^2 * exp(-2 * delay_p / 0.28)
```

其中 `q_p` 是通道 PCM 在路径时延处的量化幅值。球面方向 `d` 的观测：

```text
y(d) = gate_channel(d) * [1e-7 + Σ_p E_p * lobe(d, arrival_direction_p)]
```

`lobe` 在 82° 内平滑衰减、之外为零。系数由 `numpy.linalg.lstsq` 求解。快照中的 SH 节点声明采样数、实际截断阶数、采样密度允许的阶数上界、支撑半径与相对拟合残差。

能量求值：

```text
F_c(d) = max(0, B(d) c_c) * support_mask_c(d)
```

支撑掩码外严格为零，因此未变化的通道不会影响支撑外能量块或瓦片。

## 6. 能量块与瓦片

等距柱状像素中心映射到经度/纬度方向。固定瓦片边界由整数比例计算并写入 `payload.tile.pixel_bounds` 与 `payload.tile.lon_lat_bounds`。默认 `256×128`、4 行 8 列，每片 `32×32` 像素。

能量块依赖所有支撑范围与该块相交的 SH 节点；瓦片只依赖对应能量块。色调映射使用固定绝对 dB 标尺：

```text
db = 10 log10(E + 1e-12)
t = clip((db + 100) / 90, 0, 1)
```

避免使用会随全局最大值变化的归一化，从而保证复用瓦片字节稳定。内容哈希覆盖瓦片 ID、固定边界和 RGB 字节。

## 7. 哈希与原子写

规范化 JSON 使用 UTF-8、键排序、紧凑分隔符和末尾换行。节点内容哈希：

```text
sha256(canonical_json(payload) || 0x00 "artifact" 0x00 || artifact bytes)
```

写 blob：同目录临时文件→写入→`fsync`→`os.replace`。状态发布：完整新状态写临时文件→`fsync`→单次替换。旧 blob 永不原地修改，因此任何读取者只能看到完整旧状态或完整新状态。快照使用内容相关文件名并设为只读。

启动校验顺序：blob 存在性/哈希→节点重算内容哈希→逐依赖输入哈希。瓦片节点专用 `E_TILE_HASH`，其余用 `E_DEPENDENCY_STALE`。

## 8. 并发提交

提交在 `locks/commit.lock` 上先取进程内线程锁，再取平台文件锁（Windows `msvcrt.locking`，POSIX `flock`）。锁内重新读取 `scene.json`，比较 `expected_revision`。比较失败立即返回 `E_SNAPSHOT_CONFLICT`，不写状态。参数校验、构建和发布都在同一临界区中；若构建在任何状态发布之前失败，恢复旧场景文档。

## 9. 诊断快照

快照包含：

- 当前场景、全部参数版本戳和构建报告。
- 72 个节点的类型、内容哈希、产物哈希、输入哈希、局部版本戳、依赖、状态与失效原因。
- 参数边和 artifact 边，均带触发参数键。
- 32 个固定瓦片的像素/经纬边界、瓦片哈希和复用/重算状态。
- 投影预览 PNG（以 base64 嵌入）、RIR 四通道导出信息和声明字段。

`DiagnosticService` 只持有递归冻结的映射和规范 JSON 字节，不持有 `IncrementalEngine`，不提供 `submit`/`rebuild`。HTTP 仅 GET/HEAD 读取；`POST/PUT/PATCH/DELETE` 返回 405 与 `E_SNAPSHOT_CONFLICT`。

## 10. 确定性验收

验收脚本从两个独立空工作区运行完全相同的五步变更序列。清单只包含可复现内容：补丁、revision、闭包/复用节点 ID、节点内容哈希和瓦片哈希；不包含耗时或绝对路径。两次清单按规范 JSON 比较，要求逐字节相同。

随后从最终场景和最终版本戳创建第三个空工作区并全量构建。比较：

- 最终逐节点清单字节。
- 所有节点内容哈希。
- 所有瓦片哈希。
- 4 通道 WAV 逐样本。
- SH 系数和能量 float64 最大绝对差。

当前实测所有哈希一致，三种数值差均 `0.0`；声明容差为 0。
