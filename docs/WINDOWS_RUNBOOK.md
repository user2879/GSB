# Windows 运行手册

## 1. 环境准备

需要 Python 3.11 或更高版本。打开命令提示符或 PowerShell，在项目根目录执行：

```powershell
python --version
python -m pip install numpy pyyaml
python -m pip show numpy pyyaml
```

如果 `python` 命令不可用，可使用 Python Launcher：

```powershell
py -3.11 --version
py -3.11 -m pip install numpy pyyaml
```

也可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_and_verify.ps1
```

## 2. 生成样例

```powershell
python -m vibpipe.cli generate `
  --out examples/local_sample `
  --injections examples/injections
```

生成内容：

- `examples/local_sample/pipeline_config.json`：管线、窗口、阈值、定点档位、并发、预算和快照配置。
- `examples/local_sample/samples.jsonl`：种子 `250` 的 64 条确定性样本，共 8 个批次。
- `examples/injections/`：环、位宽越界、非法档位、阈值越界、非有限读数、重复序号、批次倒退和运行时故障注入。

## 3. 校验与运行

```powershell
python -m vibpipe.cli validate `
  --config examples/local_sample/pipeline_config.json `
  --samples examples/local_sample/samples.jsonl `
  --report artifacts/validate.json

python -m vibpipe.cli run `
  --config examples/local_sample/pipeline_config.json `
  --samples examples/local_sample/samples.jsonl `
  --out out/demo
```

默认报告：

- `out/demo/reports/pipeline_report.json`
- `out/demo/stage_products.json`
- `out/demo/snapshots/CURRENT.json`
- `out/demo/cache/stages/<cache_key>.json`

## 4. 测试与验收

```powershell
python -m unittest discover -s tests -v
python -m vibpipe.verify
```

`python -m vibpipe.verify` 会在 `artifacts/acceptance/` 下重建隔离工作区，并生成：

- `artifacts/verification_report.json`
- `docs/VERIFICATION_RESULTS.md`

验收覆盖：

- 串行与线程池并发结果逐字段一致。
- 热启动快照恢复和六阶段缓存全部复用。
- 批次追加、单样本修正、截断参数、阶段引用、档位切换、窗口变更后的增量/冷重建一致。
- 内容指纹驱动的失效与复用清单。
- `E_POINT_BUDGET_PENDING` 待办持久化及预算恢复后的等价结果。
- 八类配置/样本注入和两类运行时批次故障。
- 高一精度档位误差归因。
- 精确阈值、半步边界和三类翻转归因规则。

## 5. 预算实验

示例每批 8 个样本，五个阶段每样本共 15 个特征点，因此每批完整预算为 120。执行：

```powershell
python -m vibpipe.cli run `
  --config examples/local_sample/pipeline_config.json `
  --samples examples/local_sample/samples.jsonl `
  --out out/budget39 `
  --budget 39
```

报告状态为 `budget_pending`，失败清单包含 `E_POINT_BUDGET_PENDING`，待办写入：

`out/budget39/snapshots/versions/v00000001/pending.json`。

使用配置内充足预算重跑到另一个目录或同一输出目录，可得到完整统计。

## 6. 故障注入

配置/样本负例：

```powershell
python -m vibpipe.cli validate `
  --config examples/injections/INJ-CONFIG-CYCLE-001.json `
  --samples examples/local_sample/samples.jsonl `
  --injection-id INJ-CONFIG-CYCLE-001
```

该命令按设计返回非零退出码，报告中包含 `E_CONFIG_INVALID` 和注入编号。

运行时计算/写盘故障：

```powershell
python -m vibpipe.cli run `
  --config examples/local_sample/pipeline_config.json `
  --samples examples/local_sample/samples.jsonl `
  --inject-runtime examples/injections/INJ-RUNTIME-FAULTS-009.json `
  --out out/faults
```

报告包含 `E_BATCH_COMPUTE`、`E_SNAPSHOT_WRITE`、批次号、阶段名和注入编号；第 6、7 批之外的批次继续处理。

## 7. 关键口径

- 所有阶段值使用 Python 任意精度整数，避免中间浮点误差。
- `decimal_width=12` 对应规范十进制缩放 `10^12`。
- 允许档位为 `Q3`、`Q6`、`Q9`、`Q12`、`Q15`、`Q18`；示例使用 `Q6`，归因运行使用高一档 `Q9`。
- 漂移采用精确整数交叉相乘：`delta_q * 10000 >= threshold_bp * denominator_q`。
- 阈值完全相等时判定漂移；完全相等或距离不超过当前档位半个量化步长时标记 `boundary=true`。
- 缓存键由上游内容指纹、阶段参数指纹、小数位宽和档位共同派生。
- 快照先写入同版本目录下的临时目录，再生成不可变版本目录，最后原子替换 `CURRENT.json` 指针。
