# 技术设计

## 模块边界

| 模块 | 职责 |
|---|---|
| `pickworkbench/generator.py` | SplitMix64 确定性随机流、三类对象、固定相机、射线/点选/拖拽查询 |
| `pickworkbench/geometry.py` | 向量与旋转、三类求交、投影、容差候选、拖拽平面、栅格吸附 |
| `pickworkbench/engine.py` | 查询调度、错误隔离、首个失败、最小反例、案例报告 |
| `pickworkbench/injection.py` | 注入配置校验、深拷贝应用、操作日志 |
| `pickworkbench/demo_cases.py` | 固定异常样本集 |
| `pickworkbench/demo.py` | 双次构建校验、完整产物与顶层哈希 |
| `pickworkbench/canonical.py` | 规范化 JSON、浮点清理、SHA-256 |

## 确定性约束

1. 场景随机数来自自实现 SplitMix64，不依赖 Python `hash()`、系统时间或进程状态。
2. 对象 ID 为 `obj-000` 到 `obj-011`；查询序号单调递增。
3. 所有集合在输出前显式排序：对象按 `(object_id, object_index)`，查询按 `(query_index, query_id)`。
4. 射线的存储方向是单位向量；求交距离以世界距离表示。
5. JSON 数值在序列化边界统一为 12 位小数，负零写为 `0.0`。
6. 记录哈希不包含 `content_hash` 自身；报告哈希同样排除自身字段。
7. 报告不写运行时间、机器路径、PID 或临时目录。

## 场景与包围体

每个对象包含：

```json
{
  "id": "obj-002",
  "object_index": 2,
  "kind": "convex_polyhedron",
  "transform": {"translation": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]},
  "geometry": {"vertices": [], "planes": []},
  "pick_priority": 20,
  "bounding_volume": {
    "aabb": {"min": [], "max": []},
    "sphere": {"center": [], "radius": 1.0}
  }
}
```

球体和凸多面体支持刚体旋转；AABB 保持世界轴对齐。凸多面体由局部半空间 `n·x <= d` 表示，顶点由平面三元组求交、约束过滤和去重得到。世界平面变换为 `n_w=R·n`、`d_w=d+n_w·t`。

## 求交与决胜

- 球体：将射线变换到对象局部坐标，解二次方程并取最小非负根。
- AABB：三轴 slab 相交，入口参数必须非负。
- 凸多面体：逐平面更新入/出参数；射线起点在内部时以起点为命中点。

每条候选命中记录对象 ID、优先级、距离、命中点和 `intersection_method`。排序键为：

```text
(-pick_priority, hit_distance, object_id)
```

因此数值更大的优先级优先；同优先级取更近命中；距离和优先级均相同时取 ID 字典序更小者。

## 点选一致性

对象投影形状：

- 球体：中心投影加 `radius * focal / depth` 的像素半径。
- AABB：8 个角点投影后的屏幕 AABB。
- 凸多面体：所有世界顶点的屏幕 AABB。

候选判断为“点击点到圆距离 <= 半径+容差”或“点击点在扩张后的屏幕 AABB 内”。候选对象再用点击像素对应的射线做精确求交。因为射线与对象相交时命中点必然投影在轮廓内，射线路径的赢家一定在候选集中，所以两条路径的赢家对象 ID、距离和命中点一致。容差只扩张候选成员；`tolerance_probe` 会证明赢家不变。

## 拖拽与吸附

1. `screen -> world ray`：使用相机位置、目标、上向量、竖直视场和规律化像素。
2. 起点/终点射线分别与拖拽平面求交。
3. `unsnapped = start_world + (plane_end - plane_start)`。
4. 对齐轴坐标变为 `round(value/spacing)*spacing`。
5. `orientation_before` 与 `orientation_after` 分开记录并要求完全相等。
6. 关闭吸附重算必须与 `unsnapped` 相等。

合法间距是一个有限正标量，或三个完全相等的有限正数。逐轴非等距值被拒绝并映射为 `E_SNAP_SPACING`。

## 首个失败与最小反例

批处理按查询序号升序执行。对象级异常只在异常对象的包围体与该查询的射线/屏幕候选相关时使该查询失败。失败查询被独立记录，后续查询继续执行。

查找顺序：

```text
query_index asc
  -> object_id asc
  -> object_index asc
  -> fixed anomaly code order
```

反例从有效注入后场景中只保留触发异常及该查询所需对象。构造后重新运行验证器；随后逐个移除查询或对象并再次验证。所有删除都必须使目标错误消失，才写出 `minimal=true`。反例记录原始场景/查询/注入哈希，并包含可直接重跑的有效场景和查询切片。

## 注入格式

```json
{
  "schema_version": 1,
  "name": "duplicate_object_id",
  "description": "...",
  "degenerate_rays": [{"query_id": "q-000-ray"}],
  "zero_normals": [{"object_index": 2, "plane_index": 0}],
  "duplicate_object_ids": [{"object_index": 1, "new_id": "obj-000"}],
  "priority_overrides": [{"object_index": 0, "priority": 999}],
  "snap_spacing_override": [0.25, 0.20, 0.25]
}
```

所有修改都发生在输入深拷贝上。报告记录原始注入、规范化哈希、应用后的场景/查询哈希和逐项操作日志。注入不会修改源码，也不会写入隐含全局状态。

## 不变量清单

- 同一 `GeneratorConfig` 的场景和查询逐字节相同。
- 同一求交输入产生相同方法和相同浮点输出。
- 点选赢家与同像素等价射线赢家一致。
- 容差变化不改变任何点选赢家或命中点。
- 吸附只改变启用轴的位置，不改变朝向。
- 关闭吸附可重建未吸附位置。
- 每个失败查询都有错误码和消息，且不计入通过数。
- 第一个受影响查询由固定遍历顺序产生。
- 每个最小反例可独立复现；删除任一元素后不复现。
- 同一输入与注入配置的所有演示文件、查询哈希、记录哈希和总哈希逐字节相同。
