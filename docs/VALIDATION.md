# 实际验证

验证环境：Python 3.11.0rc1、NumPy 1.26.4、WSL2 Ubuntu 22.04。未使用容器、网络或外部服务。

## 复现命令

```bash
python3 -m unittest discover -s tests -v
python3 -m aqeval demo --workspace validation_workspace
python3 -m aqeval generate --workspace sample_workspace
python3 -m aqeval run --workspace sample_workspace
```

Windows 本地命令入口为 `scripts/run_validation.bat` 或 `scripts/run_validation.ps1`。当前执行环境未开启 `cmd.exe` WSL 互操作，因此实际日志由同一命令的 Bash 入口生成。

## 自动化测试

35 项测试全部通过，覆盖：

- 样本三次生成逐字节一致。
- 三版本解析与 46 字符定宽布局。
- v3 校验和、分区偏移和帧摘要。
- 格式错误、重复标识、未映射等级业务码。
- 六种双向转换、残差、旁路和定点位宽损失。
- v3→v2/v3→v1 的帧摘要丢弃降级与目标格式能力对齐。
- 噪声底仅使用低能量帧、连续零阈值、边界阈值与量化。
- 千分比截断和定点位宽导致的等级翻转归因。
- 等级映射缺失的跨版本翻转归因。
- 权重/阈值/音频/版本/等级表的精确缓存失效。
- 串行一致性校验只复算本次失效文件，不重算缓存复用文件。
- 快照中断续跑与 `E_SNAPSHOT_STALE`。
- 串行/并发逐字段产出一致。
- 增量与冷启动全量重建的产品字段和账本顺序一致。

完整日志见仓库根目录 `validation_tests.txt`。演示与运行日志见 `validation_output.txt`，机器可读汇总见 `validation_workspace/demo_summary.json`。

## 实际结果摘要

```text
Ran 35 tests in 22.421s
OK

initial: files=6 parsed=18 reuse=0 recompute=24
v1->v2   records=6 failures=0 degradations=0
v2->v1   records=6 failures=0 degradations=12
v1->v3   records=6 failures=0 degradations=6
v3->v1   records=6 failures=0 degradations=18
v2->v3   records=6 failures=0 degradations=6
v3->v2   records=6 failures=0 degradations=6
edits: operations=12 runs=12
final: failures=0 degradations=36 boundaries=30 reuse=24 recompute=0
snapshot_resume: interrupted=True parse_resumed=True resumed_stages=parse
full_rebuild_equivalence: equal=True
serial_vs_concurrent: equal=True
```

逐步编辑账本中的实际失效范围：

```text
set_weight               reuse 12 recompute 12
set_threshold            reuse  6 recompute 18
set_quantization_levels  reuse  6 recompute 18
set_fixed_bits(v3 only)  reuse 24 recompute  0
replace_audio            reuse 20 recompute  4
add_level                reuse 24 recompute  0
set_level_threshold      reuse 21 recompute  3
remove_level             reuse 24 recompute  0
set_concurrency          reuse 24 recompute  0
switch_version(->v3)     reuse 18 recompute  6
switch_version(->v1)     reuse 18 recompute  6
switch_version(->v2)     reuse 24 recompute  0
```

最后一次 v2 切换全部复用，是因为该最终输入与配置下的 v2 序列化缓存此前已经生成；版本切换本身只使目标版本序列化成为候选，不会使帧特征、维度和总分失效。`set_threshold` 同时改变帧摘要依赖的静噪/削波/连续零参数，因此六文件帧特征及其下游均精确失效。

最终诊断包含 `DEG_V1_TRUNCATION`、`DEG_V3_FIXED_WIDTH`、边界样本、`E_SNAPSHOT_STALE` 等业务码；完整统计见 `validation_workspace/reports/diagnostic.json`。
