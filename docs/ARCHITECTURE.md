# cfgdist — 跨组件配置分发平台技术说明

面向本地 Windows（亦可在 macOS/Linux 运行）的配置编排与依赖数据处理管线。每个组件节点是独立
子进程，通过回环 TCP 上的 JSON 行协议通信；平台负责清单强校验、定点量化与误差传播、增量失效、
受预算约束的并行分发、回执收敛、视图与权威重算一致性，以及 SQLite 快照持久化与重放校验。

## 1. 目录结构

```
cfgdist/
  errors.py                 E_* 业务码与统一异常
  injection.py              异常注入（只改输入文件/参数，不碰源码与账本）
  cli.py / __main__.py      validate/run/replay/view/inject 命令
  gui.py                    Tkinter 桌面视图（惰性导入，无显示环境用文本渲染）
  config/
    grid.py                 精度格度 Grid（缩放因子 10^q，E_GRID 校验）
    fixedpoint.py           整数定点乘加 + 精确分数误差界逐级合成
    manifest.py             清单/注册表加载与强校验、拓扑排序、环检测
    compiler.py             内容寻址、逐字节稳定的编译产物
    incremental.py          增量重编译与派生/节点引用失效推导（E_REF）
    planner.py              全量/分片分发计划与节点引用依赖 DAG
  runtime/
    node_worker.py          节点子进程（每连接一线程、幂等回执、故障注入）
    node_process.py         子进程生命周期、握手、E_NODE/E_HANDSHAKE/E_PORT、强制回收
    scheduler.py            信号量并发上限、排队计时、超时重试、依赖门控
    summary.py              节点与平台共享的规范化摘要（NumPy 聚合）
    converge.py             期望摘要重算与逐字节收敛判定（E_CONVERGE）
    view.py                 视图模型、陈旧标记、E_VIEW 一致性
    replay.py               账本权威重放与 E_REPLAY 可复现性校验
    state_machine.py        显式状态机与迁移历史
    session.py              端到端版本演进编排
  persistence/
    database.py             SQLite 表结构、内容摘要、整体替换语义
    snapshot.py             快照文件 + 数据库协调、版本/摘要辨识
  util/                     规范化编码、单调时钟、结构化日志
samples/                    确定性 registry.json 与 manifest.json（3 版本）
scripts/demo.py             一键演示（含全部业务码），run-demo.bat 为 Windows 入口
tests/                      21 个 unittest 用例
```

## 2. 定点量化与误差传播口径

* 格度：`q` 为非负整数（0..12），缩放因子 `S=10^q`；非法指数以 `E_GRID` 拒绝。
* 量化：数值与派生系数统一量化为定点整数 `mantissa = trunc(x*S)`，截断方向显式为
  **向零截断（C trunc）**，与符号无关。
* 运算：乘法 `(a*b)/S`、加法均为 Python 整数运算；每次乘积分辨率重缩放的余数被显式记录为
  本地截断误差（精确分数 `remainder/S^2`），偏移量化误差同样记录。
* 误差界：对线性派生 `y = offset + Σ c_i*x_i`，立即误差界为
  `B_y = Σ |c_i| * B_xi + 本地截断项`，系数按绝对值加权。
* 多级传播：沿派生 DAG 自底向上逐级合成，输出每条叶子→目标的传播路径、深度级别、贡献占比，
  以及每个直接上游/本地截断项的占比。误差界以 `Fraction` 精确保存，报告渲染才转十进制，
  因此权威重放可逐字节复现。

## 3. 增量编译与失效

* 产物为内容寻址：产物载荷**不含版本号**，仅由格度、定点值、目标集合、派生指纹与误差界决定，
  故未变化项跨版本的 `artifact_bytes` 与 `artifact_id` 逐字节不变（测试逐项断言）。
* 变更分类：值/系数变更、目标集合变更、新增键。值/系数变更只沿派生图向下游传播；目标集合变更
  与新增键额外沿“同节点引用”边传播。失效集合按稳定 BFS 顺序输出，每步记录触发源。
* 删除仍被派生引用的键整体拒绝，返回 `E_REF` 并列出引用者。

## 4. 分发调度与收敛

* 全局并发上限由单个 `BoundedSemaphore` 强制，任何请求都不能绕过；记录每次尝试的排队起止、
  发送起止单调时钟毫秒，并观测实际最大在途数。
* 分片策略下，仅当本节点派生项的上游由其他节点持有时建立分发依赖，依赖图跟随派生 DAG，保证
  无环；下游等待上游“完成事件”，仅在全部上游成功门置位后才执行，失败则记为 blocked。
* 超时按策略重试，单次下发语义不变：节点对同一 `request_id` 缓存首份回执，重投返回
  `duplicate=true` 且序列号、摘要字节不变。首次与重试在账本中分别记录。
* 收敛：平台按版本与派生关系重算每个节点的期望摘要，与回执做规范化逐字节比对；不一致或缺回执
  以 `E_CONVERGE` 稳定列出偏差节点。

## 5. 视图一致性与陈旧隔离

视图行携带 `source_version`。引用旧参数版本的行先标记 `stale=true` 并退出当前报告；仅新鲜行
参与与权威重算的逐字节比对，不一致以 `E_VIEW` 报告具体键。

## 6. 快照、整体替换与重放

* SQLite 持久化清单版本、格度、计划、回执、误差界、状态机迁移，每张快照与每条记录携带
  SHA-256 内容摘要；同时写一份 JSON 快照文件便于检视。
* 重新打开先比对清单版本与内容摘要；任何差异都执行**整体替换**（先删该 session/version 的全部
  行再写入），绝不增量合并，版本差异通过 `SnapshotIdentity` 可辨识。
* 恢复后按快照版本重新推导失效集、重算误差界与收敛，逐字段比对；不可复现以 `E_REPLAY` 报明
  具体配置项键与字段（如 `replay.error.total_bound`、`replay.receipt`）。

## 7. 业务码与退出码

| 业务码 | 含义 | CLI 退出码 |
| --- | --- | --- |
| E_MANIFEST | 清单/注册表强校验失败、文件损坏 | 2 |
| E_CYCLE | 派生图成环 | 3 |
| E_GRID | 精度格度参数非法 | 4 |
| E_REF | 删除仍被引用的配置项 | 5 |
| E_NODE | 节点子进程启动失败/死亡 | 6 |
| E_HANDSHAKE | 握手超时或畸形 | 7 |
| E_PORT | 回环端口占用/绑定失败 | 8 |
| E_CONVERGE | 回执摘要不一致或缺回执 | 9 |
| E_VIEW | 视图与权威重算不一致 | 10 |
| E_REPLAY | 快照重放不可复现 | 11 |
| E_STATE | 非法状态机迁移 | 12 |

## 8. 复现实验

```bash
pip install -r requirements.txt
python -m unittest discover -s tests     # 21 个单元/集成用例
python scripts/demo.py                   # 一键演示，覆盖全部业务码
```

Windows 下使用 `scripts\run-demo.bat`。GUI（需桌面与 Tkinter）：
`python -m cfgdist view --snapshot data/snapshots/demo-v3.json --gui`。
