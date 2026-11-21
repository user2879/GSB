# 图遍历可达距离定点估算与误差传播诊断台

面向本地 Windows 环境的纯 Python 桌面应用：对本地 JSON 声明的有向带权图做有界遍历，
以整数定点估算最短可达距离，逐路径传播量化/累加误差，并对距离上限做不可区分判定与
超界归因；遍历账本持久化到 SQLite，支持会话快照与恢复后一致性重放校验。

## 环境准备

- Python 3.11+（本机验证：3.11.0rc1）
- `pip install -r requirements.txt`（NumPy >= 1.26；本机验证 1.26.4）
- Tkinter：Windows 官方安装包自带；无 Tkinter 的解释器上 GUI 优雅降级，CLI 不受影响
- 仅使用标准库 `sqlite3`、`tkinter`、`concurrent.futures`，无外部网络/账号/数据集依赖

## 快速开始

```bash
python -m graphdiag.cli gen-samples           # 生成确定性样本到 data/
python -m graphdiag.cli run data/tiny_graph.json --budget 1000 --workers 2
python scripts/demo.py                        # 端到端可重复演示
python -m unittest discover -s tests          # 27 个测试
python -m graphdiag                           # 启动 Tkinter 界面（需 Tkinter）
```

CLI 还支持 `snapshot`、`restore`、`list-snapshots` 与 `--json-out`、
`--error-mode exact|conservative`。

## JSON 声明格式

```json
{
  "config": {"scale_exponent": 0},
  "nodes": [{"node_id": 0, "label": 100}],
  "edges": [{"source": 0, "target": 1, "weight_micro": 1500000}],
  "queries": [{"query_id": "q1", "source": 0, "targets": [1], "cap_micro": 4500000}]
}
```

边权与上限均为整数微单位（`1_000_000` = 1.0 逻辑单位）。非法声明以 `E_GRAPH`
整体拒绝并指出首个冲突节点/边/查询。

## 关键结论标签

- `within`：距离加误差上界仍不超过上限，确定在限内
- `confirmed_over`：距离本身已超上限
- `indeterminate_error_cross`：距离未超、但距离加误差上界跨越上限，判定受误差影响
- `unreachable`：限内无路径（含预算耗尽未到达）

错误码：`E_GRAPH`（声明非法）、`E_SCALE`（缩放致定点域截断）、
`E_REPLAY`（恢复重放不一致）、`E_STATE`（状态/格式不兼容）。

## 文档

- 设计与口径：`docs/DESIGN.md`
- 实际验证记录：`docs/VERIFICATION.txt`
