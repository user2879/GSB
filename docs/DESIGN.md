# 技术设计与不变量

## 1. 状态模型

`IncrementalEngine` 持有：

- 每层源图定点数组、掩码、当前层输出；
- 每层已提交瓦片的 `TileMeta`（缓存键、输入/引导/参数指纹、输出指纹、下层引用、版本）；
- `valid_tiles` 与 `dirty_tiles` 两个互斥集合；
- 静态父子瓦片映射；
- 当前生效参数、命令游标、预算账本和失败清单。

失效时瓦片立即从 `valid_tiles` 移除。旧元数据可保留用于审计，但不会写入快照的 `valid_tiles`，也不会被上层当作有效依赖。

## 2. 精确失效

`mask` 是替换语义，所以 `Change.rects` 同时包含旧矩形和新矩形。失效种子：

```text
seed = union(tiles_intersecting(old_rects + new_rects))
invalid = upward_dependency_closure(seed)
```

参数变化则把目标层所有瓦片当作种子。向上闭包只沿 `lower_refs` 的逆向边，所以永远不会影响更低层。

## 3. 预算与依赖

预算分配在并发执行前完成：

```text
for tile in sort(dirty_tiles, level,row,col):
    if budget exhausted: break
    if all(lower_ref in valid or lower_ref in planned): plan(tile)
```

上层任务只有在同批下层依赖计划成功后才执行。若下层任务失败，上层保持待办。这样预算扣减序列和任务顺序不受线程完成时序影响。

## 4. 一致性的比较字段

`verify` 与 `compare_engines` 比较：

- 每层像素数组；
- 每层内容指纹；
- 每个瓦片缓存键；
- 每个瓦片输出指纹；
- 合成像素与合成指纹；
- 每层贡献数组与贡献指纹；
- 待办集合。

默认低预算样本中，待办未清空时不会把中间状态伪报为“已一致”，而是记录延后；最终 drain 后必须全等。

## 5. 快照不变量

- `CURRENT.manifest_sha256 == sha256(manifest.json)`；
- `COMPLETE.manifest_sha256 == CURRENT.manifest_sha256`；
- `manifest.state_hash == state.json.state_hash`；
- 快照内每个 `valid_tiles` 项都有对应 `.npy`，且加载后指纹等于 `output_fingerprint`；
- `pending_tiles` 不与 `valid_tiles` 相交；
- 只有 `CURRENT` 指向的目录可用于恢复。

## 6. 复现性

- 样本生成只使用固定种子和 NumPy 确定性操作；
- 瓦片锁键、预算排序和结果收集均为 `(level,row,col)` 稳定顺序；
- 不依赖线程完成顺序；
- 浮点配置在进入像素计算前统一转换为定点整数；
- 上采样与所有除法舍入规则固定；
- 快照目录名由版本和状态哈希派生。

