# 实际验证记录

环境：Python 3.11、numpy 1.26.4，单机 CPU，无网络/GPU/容器。

## 复现步骤

```bash
pip install numpy==1.26.4
python generate_samples.py
python -m unittest discover -s tests
python run_demo.py
```

## 单元测试

`python -m unittest discover -s tests` 共 18 个用例，结果 `OK`：

- `test_exchange`：定宽解析、截断错误码（行号+字段名）、未知音色码、v2
  schema 校验、v1->v2 精确升级、v2->v1 sidecar 还原、超宽字段报错；
- `test_pipeline`：定点舍入与网格量化、重复运行 PCM 一致、串行/并发键与 PCM
  一致、编辑序列后增量与全量逐字段一致、单段失效边界、并发数上限；
- `test_samples_and_snapshot`：样本生成字节级可复现、16kHz/单声道/16bit、
  快照续跑全复用、结构配置变更触发 `E_SNAPSHOT_MISMATCH` 重建。

## 验收场景（run_demo.py）

报告写入 `artifacts/validation_report.json`，各场景结论：

1. **双版本互转**：v1/v2 各 5 段；v2->v1 后 sidecar 完整保留 5 段起止帧残差
   与被丢弃旁路元数据，可无损还原；强度最大绝对误差 0.0（千分步长可精确表示），
   起帧最大绝对误差 0.37、止帧 0.21（来自样本中的半帧浮点边界）。
2. **编辑序列增量一致性**：9 条命令（v1/v2 切换、边界修改、强度调整、删除、
   追加、量化档位切换）后，增量运行与空白全量重建的逐段映射参数、PCM、
   assembly 键、失败清单顺序逐字段一致；过程中持续出现段级复用（如边界编辑
   仅重算 1 段、复用 4 段）。
3. **并发一致**：`workers=1` 与 `workers=4` 的拼接 PCM、逐段缓存键、失败清单
   顺序完全相同。
4. **段隔离**：仅修改 2 号段边界时，0/1/3/4 段三阶段键不变并直接复用，2 号段
   失效重算。
5. **快照门控**：结构配置（梅尔带数/帧移）变更触发 `E_SNAPSHOT_MISMATCH` 并
   全量重建；仅量化档位变更不触发失配，而通过缓存键使全部段精确重算。
6. **解析失败**：故意截断的 v1 行产出 `E_FIELD_TRUNCATED` 并携带行号与字段名。

## 确定性保证

- 所有随机量来自固定种子/段号播种的本地 PCG 与 LCG，不读取 numpy 全局随机
  状态；
- 累加按帧序号升序、Q 定点乘加顺序固定，浮点统计使用 float64；
- 线程池仅并行无共享状态的段任务，结果按 `(段号, 阶段)` 稳定键重排；
- 同配置重复全量运行的输出 WAV 指纹相同。
