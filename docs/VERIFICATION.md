# 实际验证结果

验证环境：Python 3.11.0rc1、NumPy 1.26.4、sqlite 3.37.2；该沙箱无 Tkinter，GUI 自动降级为控制台视图（目标 Windows 标准 Python 自带 Tkinter）。

## 单元/集成测试

命令：`python3 -W ignore -m unittest discover -s tests`

结果：**31 个测试全部通过（OK）**，覆盖：

- 强校验：唯一 id、正/非负整数（含拒绝布尔）、份额常量、显存预算、分片环、缺失引用（8 项）。
- 三重约束：全局 GPU/份额、池 GPU/显存/份额/并发、任务显存预算逐轮不越界。
- 确定性：池/任务/分片乱序输入下轮次序列逐字节一致；抢占序列重复执行逐字节一致。
- 抢占：触发优先级严格高于受害者、释放量为正、受害者可恢复并最终成功。
- 依赖传播：任务内下游与跨任务下游按稳定拓扑失效，成功分片版本/摘要保持。
- 定点：向零截断/向下取整、单值界 1、比较合成界 2、聚合界 1、稳定决胜与可重复整数结果（8 项）。
- 持久化：账本链式摘要校验、篡改报 `E_LEDGER`、崩溃 UNKNOWN 标注与重放一致、人为破坏参考时 `E_REPLAY` 定位、版本漂移整体替换（5 项）。
- 注入：`memory/quota/cycle/ledger` 分别映射 `E_MEMORY/E_QUOTA/E_CYCLE/E_LEDGER`（2 项）。

## 验收场景手工复核（demo.py）

- 三重约束调度：样本在 `global_gpu_cap=8`、池并发上限与任务显存预算下并行推进，逐轮占用均在上限内。
- 确定性抢占：`train-alpha`（优先级 5）于 t=100 抢占 `train-omega`（优先级 1），释放 4 GPU / 16384 MiB，omega 随后恢复并成功。
- 定点误差界：缩放因子 10000，聚合 GPU 占用 4121、显存占用 1458，均带 `abs_error_bound=1`，差值比较界为 2。
- 快照整体替换：改动任务声明后重开，旧快照进入只读 `backup/snapshot_backup_1/`，新状态整体重建。
- 强制终止重放：第 3 轮提交前崩溃，`infer-beta#S00` 标 UNKNOWN 并重跑，恢复后缀的分片结局、GPU/显存占用与抢占记录与不中断参考一致，账本摘要校验通过。

## 静态检查

- `python3 -m compileall gpu_scheduler tests` 通过。
- `python3 -m pyflakes gpu_scheduler tests` 无告警。
