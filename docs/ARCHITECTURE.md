# 架构说明

## 数据流

```text
WAV scan/validate
  -> frame + hysteresis segmentation
  -> content fingerprint
  -> feature cache key / feature aggregation
  -> dynamic range + per-dimension tier plan
  -> description cache key / adaptive quantization
  -> adjacent-pair assignment cache key / annotation conflict check
  -> speaker aggregation cache key / fixed-point weighted vector
  -> dependency manifest + cache manifest + report
  -> staged media/manifest -> snapshot directory -> atomic CURRENT pointer
```

每个阶段只读上游不可变记录。并发任务返回局部值，主线程按稳定键归并，因此完成顺序不影响结果。

## 缓存键

- `content_fingerprint = H(segment PCM, audio+segmentation+features config, algorithm version)`。
- `feature_key = H(content_fingerprint, feature config fingerprint, sample rate)`。
- `description_key = H(feature_key, quantization plan signature, quantization config)`。
- `assignment_key = H(previous description key, current description key, assignment config)`；文件首段使用 root key。
- `aggregation_key = H(speaker_id, ordered description keys, frame weights, aggregation config)`。

量化计划签名包含每维最小值、最大值、位宽、档位、步长与模式。动态范围或强制档位改变会自动改变下游键，未变化的键继续命中。

## 命令状态机

Workbench 状态由有效配置、量化覆盖、媒体数组、当前段区间、四阶段缓存、失败清单、版本与命令游标组成。命令先完成全部校验再改动状态；失败只追加稳定错误记录。每次命令后，协调线程启动完整依赖图计算，缓存命中项不提交任务。

## 增量与全量等价

`semantic_projection(report)` 包含配置与标签哈希、量化计划、媒体清单、段、说话人、冲突、失败、依赖、当前缓存键清单、路径无关的失效/复用资格清单与规范摘要。增量与全量必须在该投影上字节级相等。

实际命中/重算键是本次执行轨迹，单独放在 `diagnostics.cache_events`。从空白运行不可能与增量运行拥有相同的历史命中集合，因此历史执行轨迹不参与最终语义等价；其规范对应物 `invalidation_reuse_manifest` 参与比较。

## 快照提交协议

1. 在临时目录完整写入媒体和 manifest。
2. `os.replace` 将临时目录重命名为唯一快照目录。
3. 原子写入 `CURRENT.json` 指针，切换生效版本。
4. 旧指针在切换前始终有效；崩溃最多留下未被引用的孤儿目录。

恢复前验证引擎 schema、基础配置哈希、标签哈希、可选全局版本、命令前缀哈希、快照媒体 SHA。任一失败即返回 stale 并从输入目录重建。

