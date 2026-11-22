# Kubernetes 调度与亲和性仿真诊断工作台

本地、离线、确定性的 Kubernetes 调度仿真：JSON 声明节点与 Pod 队列，按 Filter/Score 两阶段调度，用整数定点完成四个插件打分、万分比加权、误差界合成与稳定决胜，并通过 SQLite 账本支持参数版本、快照恢复与强制终止重放。

## 快速开始（Windows）

```bat
python -m pip install numpy
python scripts\generate_samples.py
run_demo.bat
run_tests.bat
run_workbench.bat
```

## 关键入口

- 样本：`samples/nodes.json`、`samples/pods.json`
- 演示：`scripts/demo.py`（常规调度 → 缩放因子变更重算 → 强制终止恢复，固定顺序）
- 界面：`python -m k8ssim.gui`
- 设计与误差模型：`docs/DESIGN.md`
- 实际验证记录：`docs/VERIFICATION.md`（39 个测试 + 确定性演示输出）

## 错误码

- `E_SPEC`：JSON 规格非法，定位首个出错字段，整体拒绝。
- `E_SNAPSHOT`：快照格式版本/内容摘要不一致，列出不一致项并整体替换账本。
- `E_REPLAY`：重放结论与账本不一致，定位首个不一致的 Pod 与字段。

所有调度判定均为 Python 整数运算；NumPy 只用于只读分析，不参与绑定决策。
