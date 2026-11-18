# 实际验证记录

验证时间：2026-11-18。运行入口使用工作区 `.venv`，依赖版本为 Python 3.11.0rc1、numpy 2.4.6、PyYAML 6.0.3；依赖声明兼容 Python 3.10+，由本地稳定版本解析。

## 执行环境说明

本次命令实际运行在 Windows 主机中的 WSL2 Ubuntu 环境：

```text
Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
```

当前 WSL 未启用 Windows 可执行文件互操作，调用 `powershell.exe` 返回 `Exec format error`，因此没有在原生 PowerShell Python 下重复执行。实现仅使用跨平台标准库、numpy 与 PyYAML；已提供 `scripts/setup.ps1` 和 `scripts/validate.ps1`，可在 Windows 本机直接复跑同一验收。

## 自动化测试

执行：

```bash
.venv/bin/python -m unittest discover -v tests
```

结果：

```text
Ran 14 tests
OK
```

覆盖项包括：

- 未知列、类型不符、目标列缺失、分箱数为零、阈值步长非正、并发数非正同时汇总为 `E_CONFIG_INVALID`，不中断进程。
- 两次全量运行逐字段一致。
- 概率与指标定点范围、校准误差边界归属、阈值边界 `p >= threshold`。
- 保序映射单调、训练折中位数/众数填充。
- 关闭方案后校准产物中不存在该方案的空占位。
- 篡改验证折标签不改变模型原始概率和校准拟合结果。
- 逆序样本对包含样本 ID 与右侧样本所在箱号。
- 四步编辑后每一步均与相同最终输入的空白工作区全量重建逐字段一致。
- 1 线程与 4 线程核心报告和四阶段缓存键完全一致。
- 修改已关闭方案的分箱数不会使任何阶段失效，四阶段全部复用。
- 快照清单、阶段产物校验、同分区原子临时目录、外部配置哈希拒绝、版本不一致拒绝、预先请求停止时提交中断快照。

## 初始校准质量

初始配置包含 4 个模型候选和 4 个启用方案，共 16 组模型-方案结果。96 条样本按固定种子生成，训练/验证划分为 0.72；下表为验证折的定点 ECE/MCE（6 位小数）：

| model | scheme | ECE | MCE |
| --- | --- | ---: | ---: |
| logreg-compact | raw | 0.196829 | 0.940162 |
| logreg-compact | platt | 0.191164 | 0.577667 |
| logreg-compact | isotonic | 0.195073 | 0.833333 |
| logreg-compact | histogram | 0.179946 | 0.416667 |
| logreg-regularized | raw | 0.144326 | 0.892874 |
| logreg-regularized | platt | 0.173017 | 0.469521 |
| logreg-regularized | isotonic | 0.123457 | 0.350000 |
| logreg-regularized | histogram | 0.169473 | 0.375000 |
| tree-full | raw | 0.162671 | 0.800000 |
| tree-full | platt | 0.228986 | 0.927758 |
| tree-full | isotonic | 0.192775 | 0.928571 |
| tree-full | histogram | 0.220179 | 0.928571 |
| tree-shallow | raw | 0.211539 | 0.600000 |
| tree-shallow | platt | 0.249240 | 0.630122 |
| tree-shallow | isotonic | 0.226712 | 0.575000 |
| tree-shallow | histogram | 0.229184 | 0.607143 |

阈值网格按 0.05 生成 21 个阈值，共 336 行模型-方案-阈值记录；所有准确率、召回率、F1、混淆计数和边界样本均写入 JSON 报告。当前验证数据没有出现校准映射逆序对，逆序对编号/箱号路径由专门构造的逆序测试验证。

## 增量复用与全量一致性

初始核心输出 SHA-256：

```text
5906a9bad0c3171529036afa090300bc4bf86ec0c597b88bcf12a2d3d19e69ea
```

在初始状态依次执行“方案切换 + 校准分箱/评测分箱修改”“Platt/保序正则修改”“阈值步长修改”“单样本替换”四步。每一步都把当前配置与替换文件交给全新状态目录执行一次全量重建，再比较完整核心字段和四阶段缓存键：

| 步骤 | 复用阶段 | 重算阶段 | 增量核心 SHA-256 | 全量逐字段一致 | 阶段键一致 |
| ---: | --- | --- | --- | :---: | :---: |
| 1 | data, model | calibration, metrics | `b781f8cc9a2dd12efafd468ec110d64ba9d2238936a3319038cdeb7390ac6c64` | 是 | 是 |
| 2 | data, model | calibration, metrics | `1b7be4250fd2a0480ad65717cbf03acc7579bbf43889c43580929769e83fabbb` | 是 | 是 |
| 3 | data, model, calibration | metrics | `b98b86b2a4ec94996993ac1d12d41c90f72120b679ef1b05ec9efd18a7c445ca` | 是 | 是 |
| 4 | 无 | data, model, calibration, metrics | `9c99914f2c4fe0dc84b6f197d2e3148311d87376123bdc7d79f4d03b38c225dd` | 是 | 是 |

其中第 3 步 `invalidated_inputs` 精确只有 `metrics.threshold_step`；复用 calibration 阶段没有调用校准拟合，也没有读取已失效的模型/训练输入。完整机器可读结果见 `artifacts/validation/acceptance_summary.json`。

## 并发等价

同一配置分别设置 `concurrency.max_workers: 1` 与 `4`，各自在空白状态目录全量执行：

```text
1 worker core: 5906a9bad0c3171529036afa090300bc4bf86ec0c597b88bcf12a2d3d19e69ea
4 worker core: 5906a9bad0c3171529036afa090300bc4bf86ec0c597b88bcf12a2d3d19e69ea
core_equal: true
stage_keys_equal: true
```

模型、校准、指标、阈值网格、逆序记录和失败清单均按稳定键排序，不受线程完成顺序影响。并发测试还逐字段比较了 model/calibration 阶段产物，二者完全相同；线程数只作为调度参数，不写入内容寻址产物。

## 快照与信号退出

进程级实测：启动 `probcal run` 后 0.33 秒发送 `SIGTERM`。

```text
EXIT=0
status=interrupted
completed=['data', 'model']
pending=['calibration', 'metrics']
```

进程完成当前阶段提交后生成 `snapshots/v000000001/report.json` 与 `manifest.json`，然后以整数 0 退出。阶段临时目录位于 `state/cache`，快照临时目录位于 `state/snapshots`，报告中的 `snapshot_layout.same_device` 为 `true`。

## CLI 端到端

在模板的临时副本上实际执行四个 CLI 命令，退出码依次为 `0, 0, 0, 0`：

```text
run      status=ok  version=1  recomputed=[data, model, calibration, metrics]
edit     status=ok  version=2  reused=[data, model]  recomputed=[calibration, metrics]
resume   status=ok  version=3  reused=[data, model, calibration, metrics]
verify   status=ok  manifest_version=3  same_device=true
```

## 复现命令

```bash
.venv/bin/python scripts/validate.py
```

脚本同时执行单测与验收比较，并重写：

- `artifacts/validation/acceptance_summary.json`
- `artifacts/validation/acceptance_stdout.json`
