# 实际验证记录

验证环境（当前沙箱，与目标 Windows 同版本 Python 主线）：

- Python 3.11.0rc1（目标要求 3.11+）
- NumPy 1.26.4
- SQLite 3.37.2（标准库 `sqlite3`）
- Tkinter：沙箱缺少 `libtk8.6.so`，因此 GUI 在本环境仅通过 `py_compile` 与导入测试校验；Windows 安装器勾选 Tcl/Tk 后可直接运行。

## 1. 自动化测试

命令：`python -m unittest discover -s tests -v`

结果：`Ran 39 tests ... OK (skipped=1)`，唯一 skip 为无 Tk 本地库环境下的 GUI 实例化用例。

覆盖映射：

- `test_loader.py`（11）：节点/Pod 唯一性、标签合法性、正整数与 bool 拒绝、污点 effect 集合、引用完整性、权重万分比和、缩放因子范围、E_SPEC 首字段定位。
- `test_filter.py`（6）：四闸门顺序、首失败记录、反亲和 topology 冲突、资源闸门、打乱节点顺序的候选集与拒绝诊断一致性、不可调度 Pod。
- `test_scoring.py`（8）：定点 quantize 误差界严格包住浮点参考误差、零余数零误差、整数分值范围、截断方向、重复运行一致、插件顺序无关、误差影响标注稳定、精确平手的向量序/名字序决胜。
- `test_staleness.py`（4）：参数版本单调递增、旧报告整体陈旧、被替换条目稳定、无语义变化不增版本、权重变更后整视图替换。
- `test_ledger.py`（5）：格式版本与摘要往返、E_SNAPSHOT 不一致项、E_REPLAY 首不一致字段定位、UNKNOWN 阻断校验与重跑修复、在途 Pod 无退出结论。
- `test_determinism.py`（4）：打乱节点顺序逐字节一致、队列顺序变化的结构稳定性、零权重插件、摘要稳定性。
- `test_gui_import.py`（1）：GUI 模块可导入并暴露入口（无显示环境跳过实例化）。

## 2. 三场景演示

命令：`python scripts/demo.py`，完整输出留存于 `artifacts/demo_output.txt`；连续两次运行 `diff` 无差异（DETERMINISTIC: identical runs）。

场景 1（常规调度）关键结论：

- `web-1 → node-a`、`web-2 → node-d`、`api-1 → node-b`、`gpu-job-1 → node-b`、`batch-1 → node-a`；`web-3` 不可调度（四节点全部带拒绝原因）。
- `web-1`、`gpu-job-1`、`batch-1` 标注 `error_bound_tie / error_influenced=True`，例如 `web-1` 总分差 `102000` 微点，落在合成误差界 `208001` 微点内，仍由稳定决胜规则绑定 `node-a`。

场景 2（缩放因子 10000 → 100）：

- 参数版本 `1 → 2`，旧报告标记 `stale`，`replaced_entries` 稳定列出全部 6 个 Pod；新视图整体替换，快照 `snap-v1/snap-v2` 均入帐。

场景 3（强制终止恢复）：

- 在 seq=3 处终止，已提交 `web-1/web-2/api-1`，`gpu-job-1` 及其后 Pod 标记 `UNKNOWN`；重启恢复 `verified=False` 且明确列出未知 Pod。
- 重跑未知区间后再次重放，`verified=True`，账本最终绑定与场景 2 报告完全一致（含不可调度 `web-3 → None`）。

## 3. 离线与无容器声明

- 全部验证在无网络访问条件下完成，未使用 Docker 或其他容器；样本、SQLite 账本与演示产物均位于工作区内。
- NumPy 仅出现在 `k8ssim/analytics.py` 的只读汇总；任意绑定、误差界与决胜路径均无浮点运算。
