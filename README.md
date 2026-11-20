# 隧道重组窗口裁决组件

面向本地 Windows（亦可跨平台运行）的可集成领域组件：基于**相对时钟偏差的中位数
估计**驱动重组窗口的开启/关闭判定，把偏差版本、窗口开闭、翻转、槽位/偏移丢弃与
组结局固化为可复核的 sqlite 账本，并支持强制终止后的快照恢复与一致性重放。

仅依赖 Python 3.11+ 标准库（`socket/sqlite3/tkinter/json/hashlib/threading/
time/statistics`），不使用 Docker，不访问外网，全部验证在 `127.0.0.1` 回环完成。

## 快速开始（Windows 命令行）

```bat
python scripts\generate_samples.py
python scripts\demo.py run
python scripts\demo.py replay
python scripts\demo.py faults
python scripts\demo.py udp
python scripts\demo.py ui
python -m unittest discover -s tests -v
```

也可直接运行 `scripts\01_*.bat` 到 `07_*.bat`，输出保存在 `logs\`。

运行前确认 `samples\channels.json` 中的高位端口（39101-39103）未被占用；
`udp` 子命令内置端口占用探测，占用时报 `E_PROTOCOL`。

## 子命令

- `run` 确定性裁决，写账本与快照，打印各通道估计版本/窗口/翻转/结局。
- `replay` 恢复最新快照并重跑同一脚本，逐项比对（差异报 `E_REPLAY`）。
- `faults` 注入声明损坏、样本不足、快照摘要损坏三类异常。
- `udp` 真实回环 UDP 收发，验证片头编解码与到达时刻记录。
- `ui` Tkinter 诊断界面；无 Tkinter 时自动降级为稳定排序的文本视图。
- `inspect` 列出账本中的快照及内容摘要。

## 样本设计（确定性）

- `tun-alpha`：偏差两段跳变 + 缺失片 + 重复片 + 乱序，产生 `OFFSET_DROPPED`、
  `INCOMPLETE` 与多估计版本。
- `tun-beta`：`slot_count=1` 且同窗口并发，产生稳定的 `SLOT_DROPPED`。
- `tun-gamma`：受控两段偏差（skew 4010→4030），固定锚定窗口下产生可复核的
  窗口翻转（`WINDOW_CHANGE`）。

## 业务码

| 码 | 含义 |
| --- | --- |
| `E_CHANNEL` | 通道声明/脚本强校验失败，整体拒绝并给出字段路径 |
| `E_ESTIMATE` | 偏差样本不足，估计未定，拒绝开窗 |
| `E_DIGEST` | 重组摘要与发送端登记值不一致 |
| `E_SNAPSHOT` | 快照版本或内容摘要不符，整体替换恢复 |
| `E_REPLAY` | 恢复重放与账本逐项比对不一致，定位到通道/组/字段 |
| `E_PROTOCOL` | UDP 片头/长度/CRC 自洽校验失败 |

## 关键设计

详见 `docs/DESIGN.md`。核心要点：偏差采用整数下中位数（无浮点、逐位可复现）；
窗口在首个有效估计版本锚定后跨版本固定，偏差平移只改变组对固定边界的归属，
从而产生真实翻转；全部裁决为纯函数，与片到达顺序无关，随机洗牌输入得到同一
窗口序列、翻转集合、丢弃集合与摘要。

## 验证结果

`python -m unittest discover -s tests -v` 共 30 个用例全部通过，覆盖业务要求 1-8
及真实回环 UDP；演示输出与结构化日志见 `logs/`。
