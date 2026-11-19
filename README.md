# Loopback DNS Workbench

面向本地 Windows/Linux 桌面环境的确定性 DNS 解析、缓存、上游校验与快照工作台。服务只监听
`127.0.0.1`，所有授权区、上游应答和线格式样本都来自本工作区，不访问真实域名或线上解析器。

交付包含：

- 手写 DNS 线格式编解码器（`dnslib` 仅用于测试交叉核对）。
- UDP/TCP 回环解析服务，权威区、TTL 缓存、负应答、CNAME/通配符解析。
- 本地上游 TCP 解析器模拟器、事务/问题回显/应答链校验和冲突仲裁。
- 命名、只读、带校验和的会话快照；恢复时整表替换区、缓存、策略与查询历史。
- 交互式 Tkinter 工作台、`cmd.Cmd` 命令行、确定性三场景演示。
- 确定性 JSON 区文件、上游版本文件、DNS 线格式样本和 pytest 测试。

## 快速启动（Windows）

1. 安装官方 Python 3.11+，安装时保留默认的 `tcl/tk and IDLE` 组件。
2. 在项目根目录执行 `scripts\setup_windows.bat`。
3. 启动完整桌面工作台：

```bat
scripts\start_gui.bat
```

该脚本会先启动 `127.0.0.1:15354` 的上游模拟器，再打开 Tk 工作台。也可分别启动：

```bat
scripts\start_upstream.bat
scripts\start_console.bat
```

运行确定性演示：

```bat
scripts\run_demo.bat
```

Linux/macOS 或已有虚拟环境时：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python run_demo.py
.venv/bin/python -m dns_local.gui --zone data/zones/lab.v1.json
```

Tkinter 是 Python 的可选桌面组件；Linux 可安装 `python3-tk`，Windows 官方安装包默认包含。

## 默认端点

| 组件 | 地址 | 说明 |
|---|---|---|
| DNS UDP | `127.0.0.1:15353` | 最大 512 字节，超限置 TC 并清空记录 |
| DNS TCP | `127.0.0.1:15353` | 两字节网络序长度前缀 |
| 上游模拟器 | `127.0.0.1:15354` | 仅 TCP，使用 `data/upstream/v1.json` |
| 权威区 | `data/zones/lab.v1.json` | 版本 1 样本；`lab.v2.json` 用于版本演进 |

启动前会预检 UDP/TCP 端口。端口占用返回 `E_PORT`，不会退化为随机端口。除 IPv4 回环地址外，
服务构造和上游连接都会以 `E_ARGUMENT` 拒绝配置，避免误连外部网络。

## 交互命令行

```text
python -m dns_local.cli --zone data/zones/lab.v1.json --auto-start
```

常用命令：

```text
status
query www.lab.test. A udp
query alias.lab.test. A tcp
query api.upstream.test. A udp
zone load data/zones/lab.v2.json
cache list
history 50
upstream refresh
negative 20
snapshot save demo-session
snapshot list
snapshot diff 0001-demo-session
snapshot open 0001-demo-session
start
stop
quit
```

`query` 使用 UDP/TCP 时走真实回环监听器；服务未启动或指定 `internal` 时直接调用同一核心。所有
结果都显示业务码、缓存命中状态、来源、来源序列号、剩余 TTL、归因和应答记录。

## Tkinter 工作台

`python -m dns_local.gui` 提供：

- UDP/TCP 监听启停、上游版本探测、负 TTL 上限调整。
- 查询控制台与逐条应答/归因展示。
- 缓存、历史、区记录三个实时表格。
- 快照保存、列差异、恢复；恢复按钮明确提示“整体替换”并保留旧快照只读。
- 单调时钟来源、区版本和缓存容量状态栏。

## 确定性演示

`python run_demo.py` 只向 stdout 输出稳定结果行，耗时写到 stderr。场景顺序：

1. `cold-start-local`：冷启动、大小写命中、CNAME、通配符、NXDOMAIN/NODATA 负缓存。
2. `upstream-version-change`：TCP 上游 v1 命中、切换 v2、整体失效旧上游条目、冲突仲裁。
3. `snapshot-save-restore`：保存 v1 会话、加载区 v2、保存第二版快照、关闭并重建服务、恢复
   v1 快照并验证区与缓存均为整体替换。

实测两次运行 stdout 的 `diff` 为空。参考结果见 `docs/demo-output.txt`，完整验证记录见
`docs/VERIFICATION.md`。

## 数据与语义摘要

- 名称统一为 ASCII 小写绝对名；查询匹配大小写不敏感。
- 通配符只接受最左标签 `*`；精确 owner 优先；合成记录 owner 始终是查询名。
- CNAME 最大跳数默认 8，超限或成环返回 `E_CNAME`。
- 区版本只在文件字节摘要变化时递增，时间戳变化不会产生新版本。
- 缓存键为 `归一化名称|类型|类别`；TTL 由注入的单调时钟递减，不使用墙钟。
- 负 TTL 为 `min(SOA TTL, SOA MINIMUM)`，再受可配置上限约束。
- 容量满时先回收过期项，再按稳定 LRU 与规范键名次序淘汰。
- 同一键并发回填使用单飞机制；所有等待者收敛到同一来源序列号。
- 上游 SOA serial 变化会原子失效全部上游来源条目；显式 `upstream refresh` 可立即探测。
- 本地权威区与上游返回的同区记录冲突时，规则固定为 `local-authority-wins` 并写入归因。
- 快照一致性比较区版本、区摘要、上游版本、负 TTL 上限以及每个缓存键的来源/来源序列号。
- 快照恢复不做增量合并：在缓存锁内一次性替换完整表，旧快照文件保持只读。

## 错误码

| 代码 | 含义 |
|---|---|
| `OK` | 正常完成 |
| `E_WIRE` | 非法 DNS 线格式；记录出错偏移 |
| `E_ZONE` | JSON 区整体拒绝；记录下标与字段名 |
| `E_CNAME` | CNAME 超跳数或成环 |
| `E_UPSTREAM` | 上游不可达/超时 |
| `E_CHAIN` | 上游事务、问题回显或应答链校验失败 |
| `E_PORT` | 回环端口被占用或绑定失败 |
| `E_SNAPSHOT` | 快照损坏、校验和失败或格式不支持 |
| `E_STATE` | 并发状态变化导致本次结果无法安全提交 |
| `E_ARGUMENT` | 非回环目标或不支持的参数 |

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

测试覆盖线格式与 `dnslib` 对照、压缩指针攻击样本、区拒载、通配符/CNAME、单调 TTL、稳定淘汰、
单飞并发、上游校验、UDP 截断/TCP 帧、端口冲突、快照整表恢复和演示逐字节复现。当前环境实测
`50 passed`。

详细设计见 `docs/DESIGN.md`。
