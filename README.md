# 跨组件配置分发与一致性审计平台

面向本地 Windows 环境的配置版本化管理、定点量化误差传播、并行分发、回执
收敛与 SQLite 快照恢复平台。节点为独立子进程，经回环 TCP 的 JSON 行协议
通信；无外部网络、无容器、无付费依赖。

## 环境准备

1. 安装 Python 3.10+（Windows 官方安装包勾选 *Add python.exe to PATH*
   与 *tcl/tk and IDLE*，后者提供 Tkinter）。
2. 安装 NumPy：

   ```bat
   python -m pip install -r requirements.txt
   ```

   SQLite 使用标准库 `sqlite3`，无需额外安装。
3. 回环端口 19001–19003（样本默认）需可自由绑定，无需管理员权限。

## 一键命令行演示（PowerShell / CMD）

```bat
rem 1. 清空上次的 SQLite 会话库与快照，版本号从初始状态起算
python -m ccdplatform.cli clean

rem 2. 编译并分发全部三版清单（真实拉起 3 个节点子进程）
python -m ccdplatform.cli run

rem 3. 打开最新快照，整体替换恢复
python -m ccdplatform.cli restore

rem 4. 恢复后重放校验
python -m ccdplatform.cli replay
```

可选参数：`--exponent 3`（缩放因子 10^3）、`--concurrency 2`
（全局并发上限）、`--manifest/--nodes/--db/--log` 自定义路径。

### 异常注入演示（仅改输入文件或快照参数）

```bat
rem E_RANGE：先复制清单再注入越界值
copy samples\manifest.json runtime\m_range.json
python -m ccdplatform.cli --manifest runtime\m_range.json inject --kind range --target-key alpha

rem E_CYCLE：构造派生环
copy samples\manifest.json runtime\m_cycle.json
python -m ccdplatform.cli --manifest runtime\m_cycle.json inject --kind cycle

rem E_GRID：删除快照精度格度后打开
python -m ccdplatform.cli inject --kind grid
```

非法清单整体拒绝（带字段路径与版本号）：

```bat
python -m ccdplatform.cli --manifest samples\manifest_invalid.json run
```

## Tkinter 图形界面

```bat
python -m ccdplatform.gui
```

界面提供 *清空运行时 / 运行全版本 / 恢复快照 / 重放校验* 按钮，输出区展示
结构化结果与业务错误码。若提示缺少 Tkinter，请在 Windows Python 安装器中
勾选 *tcl/tk and IDLE* 后修复安装；无图形环境时使用上面的 CLI。

## 运行测试

```bat
python -m unittest discover -s tests -v
```

34 个用例覆盖：清单强校验、定点量化与误差界、DAG 环检测与失效传播、
增量编译逐字节不变、真实子进程端到端收敛、有界并发与超时重试、
快照恢复/重放/篡改定位，以及 E_PORT/E_NODE/E_HANDSHAKE/E_RANGE/
E_CYCLE/E_GRID/E_REF/E_MANIFEST/E_CONVERGE/E_REPLAY 全部业务码。

## 产物与样本

* `samples/nodes.json`：三个回环节点（19001–19003）。
* `samples/manifest.json`：三版演进清单（基础项 + 线性派生项，
  数值在 scale=1000 的 63 位定点范围内可精确表示）。
* `runtime/session.sqlite`：会话快照；`runtime/events.jsonl`：结构化日志，
  首次回执与每次重试回执分别成条，含单调时钟毫秒。

设计与口径细节见 [DESIGN.md](DESIGN.md)。
