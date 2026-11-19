# 实际验证记录

## 环境

- Python：`3.11.0rc1`
- NumPy：`1.26.4`，从 `wheelhouse/` 本地安装到 `.venv`，未访问 Python 包索引
- 验证宿主：WSL2 Linux x86_64
- Windows wheel：`wheelhouse/numpy-1.26.4-cp311-cp311-win_amd64.whl`
- Linux wheel：`wheelhouse/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

当前执行宿主没有 `powershell.exe` 或 `cmd.exe` 互操作入口，因此无法在本会话中直接运行 Windows 二进制。已提供 `setup_offline.ps1`、`run_acceptance.ps1`、`run_demo.ps1` 和 `run_acceptance.cmd`；Windows 验证应在安装 CPython 3.11 x64 后执行这些脚本。下文数值来自同一套 Python 代码在 Linux/WSL2 上的实际运行。

## 自动化验收

命令：

```bash
.venv/bin/python scripts/run_acceptance.py
```

结果：

```text
Ran 11 tests
OK
```

机器可读摘要：`artifacts/acceptance_summary.json`。

覆盖项：

- v2 仅改变声明层；未声明层指纹完全一致。
- 基座非法时仍继续解析任务并输出 `E_BASE_INVALID`；非法任务不阻断合法任务。
- 非法目标层输出 `E_TASK_INVALID`；非法秩输出 `E_ADAPTER_CONFIG`。
- 合并按任务名排序、量化后可逆；预算不足保留待处理队列且不写半任务。
- 不可逆时输出 `E_MERGE_NOT_REVERSIBLE`，旧合并二进制哈希不变。
- 并发度 1 与 4 的适配器矩阵、损失、兼容/复用/重训/失败清单逐字段一致。
- v2 增量运行只重训 `layer_1` 任务；四个适配器结果与 v2 空白工作区全量训练一致。
- `SIGTERM` 在合并阶段被接收后完成阶段、提交并以 `0` 退出。
- 训练检查点后被强制终止，重启从检查点继续；合并阶段被强制终止后重启不会发布半成品。

## 基座指纹

```text
base_v1  sha256:ad154e0d8392a58ed616a1f158293af60862965b0c8b3c0e368f7313b914bb93
base_v2  sha256:e74407a65b2ad9908d323a582fb6d504ecab5a031740b95a12bde1148f71918f

layer_0  v1 = v2 = sha256:9880590e2b2b741a1b2b1aca00fa6efa8817abfbbf314902543523cc9ff75315
layer_1  v1 = sha256:2608df82c454993aaa690b26280c03990bb1d8762565d0b00b0b86bb2feb822a
         v2 = sha256:3eaf16b9f4f6b55771e303cf030a5b70997aedebf2241e7be7ab308664d8cf74
layer_2  v1 = v2 = sha256:6dd9bc2a582a4df9dc309b12671c31c34a57a522d434f13a4391eff0b81c0270
```

## v1 正常运行

```text
status=success, exit_code=0, snapshot_version=5
training_executed=[alpha_classify, beta_extract, delta_guard, gamma_rerank]
merge restore_max_abs_error=4.757603987680381e-07
allowed_restore_error=1e-06
budget weight_used=123, compute_used=304
```

4 个任务均生成 10 轮损失轨迹，合并任务顺序为：

```text
alpha_classify → beta_extract → delta_guard → gamma_rerank
```

## v2 增量与全量对比

同一工作区从 v1 切到 v2：

```text
reuse=[alpha_classify, delta_guard]
retrain=[beta_extract, gamma_rerank]
training_executed=[beta_extract, gamma_rerank]
business_codes=[E_BASE_VERSION_MISMATCH]
exit_code=0
```

空白工作区直接使用 v2：

```text
reuse=[]
retrain=[alpha_classify, beta_extract, delta_guard, gamma_rerank]
training_executed=[alpha_classify, beta_extract, delta_guard, gamma_rerank]
```

对比结果：

```text
alpha_classify  A/B equal=True  loss_trajectory equal=True  (复用 vs 全量重训)
beta_extract    A/B equal=True  loss_trajectory equal=True
delta_guard     A/B equal=True  loss_trajectory equal=True  (复用 vs 全量重训)
gamma_rerank    A/B equal=True  loss_trajectory equal=True
```

复用产物的整包指纹仍是 v1，但目标层指纹与 v2 的 `layer_0`/`layer_2` 相同；重训产物的整包指纹为 v2。

## 并发一致性

配置分别为 `concurrency: 1` 与 `concurrency: 4` 的两个空白工作区：

```text
compatibility=True
compatibility_hash=True
reuse=True
retrain=True
training_executed=True
failures=True
business_codes=True
all adapter A matrices=True
all adapter B matrices=True
all loss trajectories=True
all content hashes=True
merge tasks=True
merge budget ledger=True
restore error=True
```

## 预算与不可逆

将 `merge_budget.weight_elements` 设为 `30`：

```text
merged_tasks=[alpha_classify]
pending_tasks=[beta_extract, delta_guard, gamma_rerank]
ledger decisions=[ALLOW, DENY]
ledger[-1].deny_reasons=[weight_elements]
business_codes=[E_MERGE_BUDGET_EXHAUSTED]
exit_code=1
```

未完成任务的 `layer_1`/`layer_2` 合并权重等于各自基座权重的量化值，没有写入半成品适配器增量。

将 `quantization.tolerance_steps` 设为 `0`：

```text
business_codes=[E_MERGE_NOT_REVERSIBLE]
merge.status=ERROR
merge.artifact=null
旧 layer_0.weight.bin 前后字节完全相同
```

## 信号与恢复

优雅信号（合并阶段临时目录写入后发送 `SIGTERM`）：

```text
exit_code=0
status=graceful_stop
shutdown.completed_stage=merge
CURRENT.stages=[load, merge, training]
report.snapshot.version=4
```

训练检查点强制终止：

```text
kill 后 CURRENT.version=2
training.index.complete=false
training.results=[alpha_classify, beta_extract]
重启 recovery.started_from_version=2
重启结束 snapshot_version=5
四个适配器 content_hash 与干净运行全部一致
```

合并阶段临时目录写入后强制终止：

```text
kill 后 CURRENT.stages 不包含 merge
重启 recovery.started_from_version=3
重启最终 CURRENT.stages 包含 merge 和 export
```
