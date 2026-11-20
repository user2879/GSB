# 分布式数据并行训练梯度压缩通信仿真台（gradcomp）

面向本地 Windows 环境的**分布式数据并行（DDP）训练梯度压缩通信仿真台**。
系统以本地回环 TCP 端口模拟 4 个独立工作进程，在环形拓扑上执行
reduce-scatter 与 all-gather，发送前对梯度做整数定点缩放量化（截断），
并对量化误差界、误差传播归因、全局梯度一致性与快照恢复重放给出可核验结论。

## 1. 环境准备

1. 安装 Python 3.11+，安装时勾选 **tcl/tk and IDLE**（提供 Tkinter），
   并将 `python`、`pip` 加入 PATH。
2. 安装依赖：
   ```bat
   pip install -r requirements.txt
   ```
3. 确认 `127.0.0.1` 高位端口段空闲；端口被占用时改 `base_port` 重跑
   （占用会以 `E_PORT` 报错并给出端口号）。
4. 准备可写工作目录保存配置、快照与日志（默认 `workdir/`）。

> 不使用 Docker 或任何容器；仅使用本地确定性样本，不访问外部目标。

## 2. 快速开始

```bat
REM 生成样本配置
python -m gradcomp gen-config --output samples/my.json --transport tcp --base-port 39000
REM 数据摘要（确定性自检）
python -m gradcomp summary samples/my.json
REM 训练（每步原子写快照）
python -m gradcomp train samples/my.json --workdir workdir/run1
REM 查看快照与未完成文件
python -m gradcomp snapshots workdir/run1
REM 从最新快照整体替换恢复并做重放校验
python -m gradcomp replay samples/my.json --workdir workdir/run1 --extra-steps 2
REM Tkinter 界面
python -m gradcomp gui --config samples/my.json --workdir workdir/run1
```

一键演示（未压缩步 → 压缩步 → 压缩快照恢复续训，输出稳定顺序总结报告）：

```bat
run_demo.bat tcp
```

无头/无 Tkinter 环境可用 `transport=loopback`（进程内线程环，数值逻辑与
TCP 完全一致），或直接运行测试 `run_tests.bat`。

## 3. 模块结构

| 模块 | 职责 |
| --- | --- |
| `config.py` | 命令行配置、严格校验（`E_CONFIG` 整体拒绝）、配置摘要 |
| `data.py` | PCG64 确定性初始参数与各 rank 梯度分片 |
| `quantize.py` | 定点量化、三种截断方向、单值/向量/合成误差界 |
| `protocol.py` | 显式二进制帧协议（魔数、版本、阶段、块序号、CRC32） |
| `transport.py` | 回环 TCP 端点（`E_PORT`/`E_PROTO`）与进程内回环传输 |
| `ring.py` | 环形 reduce-scatter / all-gather / barrier，乱序归位 |
| `worker.py` | 单工作进程训练步（量化→环通信→参数更新） |
| `reference.py` | 单进程真值重算、一致性、误差界、两类误差归因 |
| `snapshot.py` | 原子快照、版本识别、整体替换恢复、未完成步标记 |
| `replay.py` | 恢复后续训与不中断基线逐步比对、差异定位 |
| `engine.py` | 会话编排：spawn 4 进程 / 线程、`E_DIVERGE` 校验 |
| `gui.py` | Tkinter 界面 + 无头同源状态导出 |
| `cli.py` / `__main__.py` | 命令行入口与退出码 |

## 4. 量化与误差界

量化映射 `q = trunc(x·s)`，解压 `x̂ = q/s`，`s = 10^scale_exponent`。
截断方向：

- `floor`（向负无穷）：误差 `e = x̂ − x ∈ (−1/s, 0]`，单值界 `1/s`
- `nearest`（四舍五入到偶数）：`|e| ≤ 0.5/s`
- `toward_zero`（向零）：`|e| < 1/s`

由逐元素界推导整向量范数界（`c` 为 1 或 0.5）：

- 向量 2-范数绝对界：`‖e‖₂ ≤ (c/s)·√d`
- 相对误差界：`(c/s)·√d / ‖x‖₂`
- k 个元素合成：2-范数 `(c/s)·√k`，1-范数 `(c/s)·k`
- n 个工作进程同元素误差合成界按 `k=n` 给出

误差界依赖项（缩放因子、截断方向、梯度范围观测 min/max/max_abs、维度、
进程数）随每步结果记录，并同时给出实测误差与 `bound_respected` 核验位。
浮点舍入通过相对余量 `FLOAT_SLACK=1e-9` 处理，保证“实测 ≤ 理论界”可判定。

## 5. 两类误差与传播归因

- **压缩截断误差**：压缩路径结果与“同环形累加顺序的浮点结果”之差。
- **累加顺序误差**：环形浮点累加顺序与规范固定顺序
  （rank 0→1→…→n−1 单进程重算）之差。

归因按元素计算两类误差绝对值，输出主导来源、占比与按贡献降序的
top 元素；参数偏差由 `Δθ = −η·(截断误差+顺序误差)` 追踪。归因是纯
单进程计算，**与工作进程数量、块大小、传输方式无关**，同输入重复运行
输出完全相同。

## 6. 通信协议与一致性

帧头为 30 字节定长字段（大端）：魔数 `GCMP`、协议版本、消息类型、
阶段标记、训练步、源/目的 rank、分片序号、轮次、**块序号**、负载 dtype、
元素数、CRC32，后接负载原始字节。消息按 `(chunk, round, block)` 显式
编号，乱序到达先入 `_pending` 缓存再归位，重复块、阶段错配、魔数/CRC
错误、身份不匹配均报 `E_PROTO`。

所有 rank 最终全局梯度必须**逐字节一致**，否则报 `E_DIVERGE` 并给出首个
差异元素序号；真值来源始终是单进程内按固定累加顺序重算的参考梯度。

## 7. 快照与恢复

- 写入采用 `临时文件 + fsync + os.replace` 原子替换，强制终止后只有完整
  快照可见；未完成训练步留下 `.pending` 标记，恢复时列为 unknown 并要求重跑。
- 快照含参数向量、参数版本号、种子、维度、缩放因子、截断、块大小、
  已完成步数与各分片梯度摘要。
- 恢复时比较快照版本与会话版本：不一致走 **full_replace 整体替换**，
  绝不增量合并。
- 重放校验逐位比较参数、参数版本号与全局梯度摘要，差异定位到首个
  不一致的训练步、工作进程与元素序号；缩放因子或分片划分不一致以
  `E_REPLAY` 报明具体依据字段。

## 8. 业务错误码

| 码 | 含义 | 退出码 |
| --- | --- | --- |
| E_CONFIG | 配置非法（含出错字段） | 2 |
| E_PORT | 回环端口占用/不可连接 | 3 |
| E_PROTO | 协议违规（魔数/CRC/阶段/重复块/身份） | 4 |
| E_DIVERGE | 跨进程全局梯度不一致 | 5 |
| E_REPLAY | 恢复重放不可复现 | 6 |
| E_SNAPSHOT | 快照缺失/损坏/版本不可识别 | 7 |
| E_INTERNAL | 工作进程内部失败 | 70 |

## 9. 实际验证结果

在 Python 3.11 + NumPy 2.4 上执行（`python -m unittest discover -s tests`）：

- **41 个测试全部通过**，覆盖确定性、量化界、乱序不敏感、块大小/进程数
  不变性、TCP 四进程环、`E_CONFIG/E_PORT/E_PROTO/E_DIVERGE/E_REPLAY`、
  原子快照与重放、CLI 退出码、无头界面状态稳定性。
- 真实回环 TCP 下 4 个独立进程的全局梯度逐字节一致，且与 loopback
  传输摘要完全相同。
- `run_demo.py` 连续两次运行产出**逐字节相同**的总结报告，验收三项
  （压缩全局逐字节一致、误差界可核验、快照恢复续训逐位一致）均为 true。

## 10. 备注与限制

- 当前环境为无头 Linux，未安装 Tkinter；`gui.py` 在无 Tkinter 时给出
  明确提示，其数据面通过 `build_headless_state` 完整测试。Windows 官方
  Python 安装包勾选 tcl/tk 后可直接 `run_gui.bat`。
- 量化整数按观测范围自动选择 int8/16/32/64，超 int64 报 `E_CONFIG`。
