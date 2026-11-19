# 格式规范

## 1. 内部约定

- 时间单位：秒；浮点数为 IEEE-754 binary64。
- 四元数内部统一为 `wxyz`，即 `[w, x, y, z]`。
- 所有外部四元数在进入内部前按声明顺序重排，归一化后以首关键帧 `w >= 0`、后续关键帧与上一帧点积非负的规则统一符号。
- 骨骼关节按父节点先于子节点的拓扑顺序保存；静止旋转为局部旋转，静止位移为父关节局部坐标下的偏移。

## 2. JSON 动画格式

### 2.1 schema v3（最新）

```json
{
  "schema_version": 3,
  "joint_map": {"root": 0, "pelvis": 1},
  "quat_order": "wxyz",
  "interpolation": "slerp",
  "fps": 30.0,
  "keyframes": [
    {
      "time": 0.0,
      "rotations": [[1.0, 0.0, 0.0, 0.0], [0.98, 0.1, 0.0, 0.06]],
      "root_translation": [0.0, 0.0, 0.0]
    }
  ]
}
```

- `joint_map`：关节名到旋转数组下标的映射，下标必须从 0 开始连续且唯一。
- `quat_order`：`wxyz` 或 `xyzw`。
- `interpolation`：`slerp`、`linear`（归一化线性插值）或 `step`。
- `keyframes[*].rotations`：按 `joint_map` 下标排列的四元数。
- `keyframes[*].root_translation`：世界空间根位移。
- `fps`：可选名义帧率，不代替关键帧时间戳。

### 2.2 schema v2

v2 使用 `frames` 而非 `keyframes`，每帧不含 `root_translation`。v2 可以有
`joint_map`、`quat_order` 与 `interpolation`；若缺省，按迁移表补齐。

### 2.3 schema v1

v1 使用：

```json
{
  "schema_version": 1,
  "joints": ["root", "pelvis"],
  "frames": [
    {"time": 0.0, "rotations": [[0.0, 0.0, 0.0, 1.0]]}
  ]
}
```

v1 不携带 `joint_map`、`quat_order`、`interpolation` 和根位移通道。v1 的四元数
固定解释为 `xyzw`。

### 2.4 固定迁移表

| 缺失字段 | 来源版本 | 固定补齐值 |
|---|---|---|
| `joint_map` | v1，或 v2 同时有 `joints` | 按 `joints` 数组顺序生成 `{name: index}` |
| `quat_order` | v1、v2 | `xyzw` |
| `interpolation` | v1、v2 | `slerp` |
| `root_translation` | v1、v2 | 每帧 `[0.0, 0.0, 0.0]` |
| `frames` → `keyframes` | v1、v2 | 深拷贝为 `keyframes`，保留原字段 |

迁移只增加/转换已定义语义，不修改已有时间戳、旋转值或已有字段的值。对 v3 文档
再次迁移不改变内容；规范化 JSON 内容哈希保持不变。未知版本返回
`E_SCHEMA_VERSION`，不会猜测字段。

## 3. 二进制轨迹格式 BTRJ

### 3.1 24 字节头

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 4 | magic：ASCII `BTRJ` |
| 4 | 1 | 二进制格式版本，支持 1、2 |
| 5 | 1 | 声明端序：ASCII `L`（little）或 `B`（big） |
| 6 | 4 | 端序标记：整数 `0x01020304` 按声明端序写入 |
| 10 | 1 | 索引宽度：2 或 4 字节 |
| 11 | 1 | 四元数顺序：0=`wxyz`，1=`xyzw` |
| 12 | 1 | 插值：0=`slerp`，1=`linear`，2=`step` |
| 13 | 1 | flags；bit0 表示存在根位移通道 |
| 14 | 2 | 保留，写 0 |
| 16 | 4 | 关节数 `N` |
| 20 | 4 | 帧数 `F` |

头内整数与标记均使用声明端序。解析器先根据 4 字节标记独立探测实际端序，再与
偏移 5 的声明比较；不一致或标记非法返回 `E_FORMAT_ENDIAN`。

### 3.2 负载顺序

1. 索引表：`N * index_width` 字节。第 `r` 项表示序列化通道行 `r` 对应的规范关节
   下标，必须构成 `0..N-1` 的排列。
2. 关节名：`N` 个 `[u16 字节长度][UTF-8 bytes]`，按规范关节顺序。
3. 时间戳：`F * float64`，严格递增。
4. 根位移（v2 且 flags bit0）：`F * 3 * float64`。
5. 旋转：`F * N * 4 * float64`，帧优先，每帧内按索引表通道行排列；四元数分量按
   头中声明顺序解释。

v1 没有根位移块，插值缺省为 `slerp`；解析结果仍记录实际端序、索引宽度、四元数
顺序与插值方式。

### 3.3 结构错误

- 头长度、名字表、时间轴、根位移截断：`E_FIELD_MISSING`。
- 索引值越界或重复：`E_INDEX_OVERFLOW`。
- 旋转分量数不足以覆盖声明帧数/关节数、四元数为零范数或含 NaN：`E_QUAT_INVALID`。
- 时间戳非严格递增：`E_TIME_NONMONOTONIC`。
- 端序声明与标记冲突：`E_FORMAT_ENDIAN`。
- 不支持的格式版本：`E_SCHEMA_VERSION`。

## 4. 四元数与采样

1. 转成内部 `wxyz`。
2. 检查 4 分量、有限值和范数；零范数/NaN/Inf 返回 `E_QUAT_INVALID`，携带关节名与帧号。
3. 归一化。
4. 首关键帧使用确定性半球；后续帧若与上一帧点积为负则取反。
5. 相邻关键帧先完成符号对齐，再按 `slerp`、`linear` 或 `step` 采样；采样后再次归一化。

因此对整段轨迹逐关节乘以 `-1` 不会改变采样后的姿态。

## 5. 时间轴重采样

给定源时间轴 `[t0, tn]` 和目标帧率 `fps`：

```text
tk = t0 + k / fps,  k = 0, 1, ...
只保留 tk <= tn（边界 epsilon = 1e-12）。
```

不生成早于 `t0` 或晚于 `tn` 的帧，不做外插。根位移在允许区间内线性插值；
`step` 模式取左关键帧。时间戳相等或倒序返回 `E_TIME_NONMONOTONIC`。
