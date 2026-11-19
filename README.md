# Block Storage Snapshot & Rollback（块存储快照与回滚教学组件）

纯 Python 3.11 标准库实现的本地块设备快照/回滚领域组件：CoW 快照、内容寻址块仓、
三条自适应回滚路径（位图差分 / 全量比对 / 增量链回放，逐位一致）、原子切换与中断恢复、
清单格式版本整体替换、版本化审计与结构化业务码日志，附 CLI 与 Tkinter 界面。

## 快速开始

```bash
python --version                       # 需要 3.11+
python scripts/demo.py                 # 四场景可重复演示
python -m unittest discover -s tests   # 20 个测试
python scripts/blocksnap_cli.py --help # 命令行
python scripts/launch_ui.py            # Tkinter 界面（需桌面环境）
```

- 设计、数据模型、恢复机制与验证记录见 `docs-TECHNICAL.md`。
- 三组确定性样本卷定义在 `block_snap/samples.py`。
- 业务码：`E_INPUT / E_PATH / E_INTEGRITY / E_CHAIN / E_SHAPE / E_STATE`。
- 无第三方依赖（见 `requirements.txt`）；无 tkinter 时 UI 优雅降级，其余功能不受影响。
