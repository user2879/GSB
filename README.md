# 分层扰动预算诊断工作台

本地、确定性的图像修复与风格迁移诊断工作台。样本是固定种子生成的 PNG + 整数矩形缺损掩码；系统构建粗到细金字塔，执行定点扩散修复、逐级 LUT 风格映射、瓦片级缓存失效、单批预算/待办续跑、原子快照恢复和串并行异常注入对比。

## 快速开始

要求 Windows/Linux/WSL 上的 Python 3.11+：

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 run_demo.py
py -3.11 -m unittest discover -s tests -v
py -3.11 -m workbench.cli check-env
py -3.11 -m workbench.cli gui
```

Windows 也可以运行 `run_demo.bat`，或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_windows.ps1
```

`python -m tkinter` 可检查官方 Python 的 Tcl/Tk；`python scripts\check_tkinter.py` 会做导入和短时窗口冒烟检查。

## 一键演示做了什么

`python run_demo.py` 会：

1. 确定性重建 `samples/`、`configs/` 和 `examples/`；
2. 清空并重建 `state/` 快照目录；
3. 用预算 6 演示 48 个瓦片中只提交前 6 片，其余 42 片进入 `E_TILE_BUDGET_PENDING`；
4. 从快照清单恢复新会话，验证前 6 片复用、下一批 6 片重算；
5. 应用完整 8 条命令流并逐次原子提交快照；
6. 对比增量结果与空白全量重建的每层像素、最终像素和全部缓存键；
7. 验证 `E_SNAPSHOT_STALE` 版本/哈希探测；
8. 运行 3 个异常注入的串并行一致性诊断；
9. 写出 `artifacts/reports/verification-report.json` 和 `outputs/*.png`。

当前实际结果：20 个 unittest 全部通过，增量/全量一致，3 个注入全部命中且串并行一致。详见 `docs/TECHNICAL_NOTES.md` 与 `artifacts/reports/test-run.txt`。

## 常用命令

```bash
# 生成样本和 JSON 样例
python -m workbench.cli generate-samples

# 校验 PNG 头、配置、掩码和命令流
python -m workbench.cli validate --config configs/workbench.json --commands configs/edit_stream.json

# 从空白按预算重放命令；--drain 自动续跑待办
python -m workbench.cli run-stream \
  --config configs/workbench.json \
  --commands configs/edit_stream.json \
  --clean --drain --budget 12 --workers 4

# 检查当前快照版本和配置哈希
python -m workbench.cli snapshot-check --config configs/workbench.json

# 配置版本/哈希变化后，整体重建恢复
python -m workbench.cli run-stream --config configs/workbench.json \
  --commands configs/edit_stream.json --recover-stale --drain
```

## 目录

- `workbench/configuration.py`：配置、枚举、金字塔因子和瓦片几何校验。
- `workbench/image_io.py`：PNG 签名/IHDR 分辨率校验与读写。
- `workbench/masks.py`：整数矩形掩码校验、栅格化和掩码金字塔。
- `workbench/commands.py`：命令流整批校验和状态更新。
- `workbench/pyramid.py`：块均值下采样、定点双线性上采样、层级指纹和依赖几何。
- `workbench/kernels.py`：由外向内扩散修复、引导混合、定点 LUT 风格映射。
- `workbench/engine.py`：瓦片缓存键、依赖、稳定调度和线程池执行。
- `workbench/session.py`：复用/失效/预算/待办状态机与清单。
- `workbench/snapshot.py`：临时目录、整体改名和原子提交指针。
- `workbench/diagnostics.py`：异常注入与串并行对比。
- `workbench/demo.py`：一键验收和 JSON 证据报告。
- `tests/test_workbench.py`：19 个回归测试。
- `configs/workbench.json`：主配置。
- `configs/edit_stream.json`：掩码增删改、迭代轮数、引导、风格权重和量化命令。
- `configs/injection.json`：独立异常注入样例。
- `examples/invalid_config.json`、`examples/invalid_mask.json`：负路径样例。

## 业务错误码

- `E_CONFIG_INVALID`：配置、PNG 头、命令或枚举不合法。
- `E_MASK_OUT_OF_RANGE`：掩码矩形非整数、宽高非正或越界。
- `E_TILE_GEOMETRY`：瓦片边不能整除对应层尺寸。
- `E_TILE_BUDGET_PENDING`：本批预算已耗尽，仍有稳定排序后的待办瓦片。
- `E_SNAPSHOT_STALE`：全局版本号或配置哈希与已提交快照不匹配。
- `E_NONFINITE_PIXELS` / `E_QUANT_OUT_OF_RANGE`：异常注入诊断中的真实输入故障码。

校验、命令流和注入配置都会在写入快照前完成解析；非法整批返回 JSON 错误，不产生新的部分提交。
