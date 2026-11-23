# 实际验证记录

验证时间：2026-11-24（Asia/Shanghai）

环境：

```text
Python 3.11.0rc1
NumPy 1.26.4
workspace /home/zsl/feedback/uidfc5909d3-default-210
```

## 命令

```bash
python scripts/generate_samples.py --output-dir sample_data --count 8 --seed 20261124
python scripts/selfcheck.py --workspace . > artifacts/latest/selfcheck.json
python -m rope_diag --workspace . --output artifacts/latest/report.json > artifacts/latest/run_first.txt
python -m rope_diag --workspace . --output artifacts/latest/report_incremental.json > artifacts/latest/run_incremental.txt
python -m rope_diag --workspace . --output artifacts/latest/report_full.json --force-full > artifacts/latest/run_full.txt
python -m unittest discover -s tests -v > artifacts/latest/test-results.txt 2>&1
python -m pip install . --no-deps --no-build-isolation --target artifacts/latest/install-test
```

## 自检

`artifacts/latest/selfcheck.json` 记录：

- NumPy 导入成功。
- 8 条样本全部合法。
- 4 条剪枝策略合法。
- 1 条默认异常注入项合法。
- `E_CONFIG_INVALID` 与 `E_SAMPLE_INVALID` 的自测分支均命中。
- `ok=true`。

## 默认诊断

首次运行 `artifacts/latest/report.json`：

| 方案 | 状态 | 完成样本 | 训练窗内打分 | 外推打分 | 外推打分阶段均值偏差 |
|---|---:|---:|---:|---:|---:|
| `rope` | `ok` | 8/8 | 46 | 50 | 4.040450 |
| `linear_interpolation` | `partial_fault` | 7/8 | 41 | 43 | 3.274033 |
| `ntk_aware` | `ok` | 8/8 | 46 | 50 | 3.391939 |

默认故障命中：

```text
INJ-SAMPLE-SCORING    sample_id=s0006            stage=scoring
```

失败清单按样本 ID 排序；聚合故障的空样本 ID 位于最前。

## 增量与全量

热缓存增量：

```text
approved=1 reused=23 pending=0
reused_stages=10 invalidated_or_faulted_stages=2
```

仅被注入强制执行的样本任务进入调度；其余位置编码、打分、误差追踪和敏感性结果直接复用。

语义摘要：

```text
first         61f0aaf5a04e2936aeb44182eea7229eeaf6c5c43252e85f9bcefd277c04ba10
incremental   61f0aaf5a04e2936aeb44182eea7229eeaf6c5c43252e85f9bcefd277c04ba10
force_full    61f0aaf5a04e2936aeb44182eea7229eeaf6c5c43252e85f9bcefd277c04ba10
```

三者一致。

安装验证：旧版 pip 环境下成功构建 `rope_extrapolation_diagnostics-1.0.0-py3-none-any.whl`，安装后的 `rope-diag --help` 入口可用；验证记录见 `artifacts/latest/pip-install.txt` 与 `artifacts/latest/installed-cli-help.txt`。

## 测试

`artifacts/latest/test-results.txt`：

```text
Ran 15 tests in 2.458s

OK
```

覆盖项：

- 配置、样本、剪枝、注入清单校验与错误码。
- 三方案真实角度差异、基频/缩放敏感性、旋转范数保持。
- 稳定 softmax 的平移不变性。
- 训练窗分层统计与无外推错误。
- 三阶段误差完整记录和首个超差环节。
- 基频/缩放邻域排序、缩放溢出、维度级截断。
- 策略选型的维度集合、分块边界和指纹差异。
- 热缓存复用、缩放变化的选择性失效、单样本替换、全量一致性。
- 点数预算待续、版本续跑、串行与并发一致性。
- 样本级和聚合级异常注入隔离。

## Windows 说明

当前执行环境为 WSL/Linux，不能在本机启动 Windows 主机进程。实现只使用 `pathlib.Path`、`os.replace` 和显式 UTF-8，并提供 `scripts/windows_selfcheck.ps1`。Windows 验收时应在该脚本所在的虚拟环境中实际执行一次，以确认 NumPy 导入、项目导入和样本校验。
