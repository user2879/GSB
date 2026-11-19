# 实际验证记录

验证日期：2026-11-19

运行环境：WSL2 Linux，Python 3.11.0rc1，NumPy 2.2.6。当前 WSL 发行版未启用 Windows interop，无法直接执行 `cmd.exe`/PowerShell；项目代码使用跨平台路径、原子替换和文件 API，Windows 启动/安装脚本位于 `scripts/`。

## 回归测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：18 项通过，0 失败，34.11 秒。覆盖：

- 相同配置两次生成 AVI、逐帧真值和清单逐字节一致；
- 不同种子产生不同视频、像素内容和真值哈希；
- 逐帧真值数量、镜头边界、GOP 边界、色块、水印字段；
- 像素内容哈希从 AVI 重新读取后可复算；
- 采样、掩膜、分析窗口的精确闭包；
- 单 GOP 变化仅重算两个相邻 `xgop` 比对项；
- 增量与全量逐节点一致，差异项含节点、比对项和边上下文；
- 快照哈希/版本冲突、源变化后禁止旧缓存复用；
- 正常关闭清单、强制终止、孤儿回收、恢复后逐节点一致；
- 六个业务错误码链路。

## 完整 60 秒演示

命令：

```bash
.venv/bin/python -m forensic_workbench demo --workspace workspace-final
```

结果：退出码 0，总耗时 127.2896 秒。

素材：

- 帧数：1800
- GOP 数：60
- 分辨率/帧率：640x360 / 30fps
- AVI 字节数：1,244,203,432
- 生成耗时：6.1380 秒
- 视频 SHA-256：`ae558d0ab266aea87bf24f48ffeea770a7da1b7b2f14a4ee06af4392093d6b0d`
- 像素内容 SHA-256：`a5b3844235e4162652ec16359f81f78aca5a4bed0ac0ee9ec7b7c8be8dc26957`
- 真值清单 SHA-256：`61c9accace67ae1c98f31615a8c6c03ec04987e59930df52e74c658f52edc7f1`
- 素材清单 SHA-256：`e16d63b8ac7bedb43342a3bac83596c8f5d918a56da7bacadf0398071b442213`
- 确定性探针：AVI、真值、清单三项字节一致

增量场景：

| 场景 | 重算节点 | 复用节点 | 撤销/窗外节点 | 全量对照 |
|---|---:|---:|---:|---|
| 基线 | 1921 | 0 | 0 | 通过 |
| 掩膜变更 | 34 | 1887 | 0 | 通过 |
| 采样变更 | 34 | 1887 | 0 | 通过 |
| 分析窗口变更 | 4 | 56 | 1861 | 通过 |
| 恢复后复用 | 0 | 60 | 0 | 参数未变，按提交状态复用 |

掩膜与采样变更的 34 个重算节点由 30 个帧节点、1 个 GOP、2 个相邻跨 GOP 项和 1 个结论组成，符合单 GOP 失效闭包。

## 重启与强制终止

- 重启重新加载版本 4 快照，逐节点哈希与提交状态一致。
- 强制终止子进程退出码 17，`close.json` 不存在，运行状态为 `unclean_termination`。
- 恢复隔离并记录 1 个未引用快照、1 个待提交摘要，事件业务码均为 `E_SNAPSHOT_VERSION`。
- 恢复后的当前快照版本、快照哈希和全部节点哈希与崩溃前提交状态一致。
- 恢复后以相同参数运行：重算 0 节点，结论哈希仍然一致。

完整机器可读证据：本地运行生成的 `workspace-final/reports/demo_report.json`、`workspace-final/state/recovery/recovery-000001.json` 和各 `workspace-final/runs/*/close.json`。
