# cfgdist — 跨组件配置分发平台

本地子进程 + 回环 TCP JSON 行协议的配置编排、定点误差传播与结果验证平台。
详细设计见 `docs/ARCHITECTURE.md`。

## 环境准备

- Windows 10+（亦可在 Linux/macOS 运行），Python 3.11+
- 安装依赖：`pip install -r requirements.txt`（NumPy；SQLite、Tkinter 为标准库）
- Tkinter 仅在使用 `view --gui` 时需要，核心管线不依赖 GUI

## 快速开始

```bash
python -m cfgdist validate                         # 清单/注册表强校验
python -m cfgdist run --session-id demo \
    --concurrency 2 --max-attempts 3 --timeout-ms 10000
python -m cfgdist replay --session-id demo         # 权威账本重放
python -m cfgdist view --snapshot data/snapshots/demo-v3.json
python scripts/demo.py                             # 一键演示（含异常注入业务码）
```

异常注入（只改输入文件/参数，不改源码与账本）：

```bash
python -m cfgdist inject corrupt                  # E_MANIFEST
python -m cfgdist inject cycle                    # E_CYCLE
python -m cfgdist inject bad_grid --bad-grid-q 99 # E_GRID
python -m cfgdist run --versions 1 --inject-node node-c=handshake_fail  # E_HANDSHAKE
python -m cfgdist run --versions 1 --inject-node node-b=node_fail       # E_NODE
```

## 验证结果

- `python -m unittest discover -s tests`：21/21 通过（定点、清单校验、增量失效、
  网格、规范化、真实子进程端到端、视图一致性、重放）。
- `python scripts/demo.py`：9/9 检查通过，覆盖收敛、误差界、快照重放与
  E_MANIFEST / E_CYCLE / E_GRID / E_NODE / E_HANDSHAKE 业务码。
- 另经定向脚本验证 E_PORT、E_CONVERGE、E_REF、E_VIEW、E_REPLAY 与超时重试幂等语义。

## 关键保证

- 未变化配置项的编译产物跨版本逐字节一致（内容寻址，产物不含版本号）。
- 定点乘加全部整数运算、向零截断；误差界用精确分数逐级加权合成并可重放。
- 调度并发绝不超过全局上限；重试保持单次下发语义（同 request_id 幂等回执）。
- 快照按内容摘要整体替换恢复，重放不可复现以 E_REPLAY 报明键与字段。
