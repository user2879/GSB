# 扇出下发调度账本组件 — 技术说明

面向本地 Windows（同样在 Linux/macOS 运行）容量评估与压测场景的确定性扇出
下发调度账本。全部目标节点是本仓库附带的、监听 `127.0.0.1` 高位端口的本地
确定性模拟服务；不依赖任何真实账号、线上站点、付费资源或 GPU，也不使用容器。

## 1. 环境与目录

要求 Python 3.11+。Windows 使用 python.org 官方安装包（自带 Tk/Tkinter）。

```
fanout_ledger/          # 核心包
  config.py             # 场景加载与强校验（E_SCENE，字段路径）
  mock_service.py       # 确定性回环模拟服务 + 端口占用检测（E_PORT）
  http_client.py        # 裸 socket 回环客户端（真实超时/真实连接错误）
  scheduler.py          # 双重并发预算、排队计时、退避重试、E_RETRY
  ledger.py             # SQLite 账本（单事务原子写、unknown 标记）
  metrics.py            # 整数定点聚合 + 可复现 SLO 判定
  snapshot.py           # 内容摘要、原子整体替换、E_SNAPSHOT
  recovery.py           # 强制终止恢复 + 一致性重放（E_REPLAY）
  engine.py             # 会话编排与稳定顺序总结报告
  cli.py / __main__.py  # validate/init-db/run/replay/ui
  logging_util.py       # 结构化 JSONL 日志
  ui/dashboard.py       # Tkinter 仪表盘（Tk 缺失时显式报错）
samples/                # 三份确定性场景 JSON（由脚本生成）
scripts/                # 样本生成、模拟服务、演示与 Windows 辅助脚本
tests/                  # 27 个 unittest 用例
run/                    # 运行产物（账本、快照、JSONL 日志、报告）
```

环境初始化：

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# 或 Linux:
python3.11 -m venv --without-pip .venv  # 再引导 pip（见 README）
```

## 1.1 端口选择（跨平台避免临时端口冲突）

样本服务固定使用 488xx/489xx/490xx，测试使用 48720–48736，全部落在
**48716–49151** 安全间隙：高于本机 Linux 临时端口范围（44620–48715），
低于 Windows 默认动态端口范围下界（49152）。这样内核不会把这些高位端口
随机分配给出站连接，避免偶发 `E_PORT`。`mock_service.stop()` 还会等待端口
真正释放后才返回，消除连续场景间的启动竞态。

## 2. 确定性模拟规则

每个节点的第 `n` 次请求结局是 `(seed, n, failure_rate, slow_rate)` 的纯函数：

```
bucket = (seed * 7919 + n * 104729) % 10000
bucket < failure_rate                       -> HTTP 500 failure
failure_rate <= bucket < failure_rate+slow_rate -> 慢响应（额外 sleep）
否则                                        -> HTTP 200 success
```

慢响应在服务端真实 sleep（`process_ms + slow_extra_ms`）。当该总时长超过客户
端单次尝试超时时，客户端通过**真实套接字超时**得到 `timeout`，没有任何人为
延迟来伪造超时。样本里的 seed/速率组合经过离线核验，结局序列可证明，例如
seed=2、`slow_rate=9000`、超时 120ms 而慢响应 300ms 时，三次尝试全部是真实
socket timeout，投递终局为 E_RETRY。

## 3. 八项业务要求的落点

1. **强校验（E_SCENE）**：`config.py` 收集全部非法字段路径后整体拒绝。校验
   节点 id 唯一、host 必须是字面量 `127.0.0.1`、端口 1..65535 且全局唯一、
   载荷/最大尝试/超时为正整数、退避序列为单调不减非负整数、成功率门槛
   0..100、P95 门槛为正整数。损坏 JSON 同样以 E_SCENE 拒绝。
2. **双重预算调度**：全局 `BoundedSemaphore(global_concurrency)` 与每节点
   `BoundedSemaphore(node_concurrency)` 同时生效；任务按“投递序号外层、场景
   节点顺序内层”的稳定 round-robin 提交，不依赖字典/哈希遍历。每次尝试记录
   单调时钟的排队起止、运行起止、结局与预算消耗。并发上界由测试持续采样断言。
3. **真实超时与退避**：连接阶段使用 OS connect 超时（拒绝/不可达归类
   `unreachable`），请求阶段使用配置的 per-attempt socket 超时（归类
   `timeout`）。失败按退避序列等待后用**相同**超时语义重试；超过
   `max_attempts` 写入 `retry_exhaustions` 表（E_RETRY），记录已用尝试数与
   累计退避毫秒。该标记不是网络尝试，不进入指标聚合分母。
4. **定点指标与 SLO**：成功率定点标度 10000（1bp，0.01%），向零截断，误差上界
   `< 1/10000`；吞吐标度 100（0.01 KB/s），向零截断，误差 `< 0.01 KB/s`；
   P95 用整数毫秒 nearest-rank，标度 1、误差 0ms。成功率按**投递终局**计算
   （重试不膨胀分母），尝试级结局分布另列。判定输出触发值/门槛值/口径，序列
   化用排序键+固定分隔符，同一账本重复判定逐字节一致。
5. **账本与快照恢复**：每次尝试在单条 SQLite 事务内写齐记录、结局与预算；
   WAL + `synchronous=FULL`。进行中的尝试没有任何提交行，进程被杀后恢复时在
   `recovery_events` 标记 `unknown` 并要求重跑，绝不基于未知状态生成退出报告。
   快照经临时文件 + `os.replace` 原子整体替换，携带 SHA-256 内容摘要；被杀时
   残留的 `.snap-*.tmp` 由 `cleanup_staging_files` 在恢复前清理。
6. **重放校验（E_REPLAY）与不可复现（E_SNAPSHOT）**：先核对场景版本、载荷
   摘要、模拟失败率配置，不一致逐项列入 `problems` 并判不可复现；再按快照轨迹
   顺序重驱相同尝试序列，逐尝试比较结局、再比较节点结局分布与聚合成功率（精确
   相等）。P95 是真实回环上的整数毫秒墙钟，仅它带显式声明的
   `max(2ms, 5%)` 测量容忍；首个差异定位到节点、投递/尝试序号与字段。
7. **异常链路与结构化日志**：`scripts/demo.py --inject {corrupt-scene,
   occupied-port,bad-snapshot}` 仅改输入文件或启动参数，不改源码，分别产出
   E_SCENE/E_PORT/E_SNAPSHOT 并写 JSONL。E_SCENE/E_PORT 在任何账本写入前失败，
   E_SNAPSHOT 只拒绝被篡改快照、已提交账本保持一致，无半写冲突。
8. **Tkinter UI 与可重复演示**：`ui/dashboard.py` 以场景顺序/快照文件名稳定
   渲染节点进度、运行/排队数、预算、结局分布、P95、SLO 原因、场景版本与快照
   列表；无 Tk 环境延迟导入并返回 `E_UI_UNAVAILABLE`。演示顺序执行冷启动全量
   扇出、收紧 SLO+提高并发后重跑、故障注入后强制终止并恢复重放。

## 4. 三个演示场景的确定性结论

| 场景 | 投递成功 | 尝试结局 | SLO | 关键现象 |
|---|---|---|---|---|
| 冷启动 `cold-start-v1` | 5/5（10000bp） | 成功5/失败1/超时1 | 通过 | 失败与超时各重试一次后成功 |
| 调参 `retune-v2` | 3/5（6000bp） | 成功3/失败3/超时3 | 不达标 | 成功率与 P95 两类原因同时触发；n2/n3 E_RETRY |
| 崩溃恢复 `crash-recovery-v1` | 8/10（8000bp） | 成功8/失败2/超时6 | 通过 | SIGKILL 后 unknown 标记、缺失投递重跑、重放逐尝试一致 |

## 5. 常用命令

```
python -m fanout_ledger validate samples/scene_cold_start.json
python -m fanout_ledger init-db run/cli_check
python -m fanout_ledger run samples/scene_cold_start.json --run-dir run/s1
python -m fanout_ledger replay samples/scene_cold_start.json --run-dir run/s1
python -m fanout_ledger ui --run-dir run/demo/01_cold_start
python scripts/demo.py
python scripts/demo.py --inject occupied-port
python -m unittest discover -s tests -v
```

## 6. 设计取舍与误差声明

* 成功率以投递终局为 SLO 分母，尝试级分布单独保留，避免“重试越多成功率越
  低”的口径误导；两者都进账本与快照。
* 唯一非精确比较项是 P95 的物理测量抖动，容忍公式 `max(2ms, 5%)` 与依据写入
  快照、重放结果和 E_REPLAY 详情；结局分类、结局分布、成功率均为精确比较。
* 模拟服务固定处理时长提供可辨识基线（样本约 20–35ms），使 OS 调度抖动相对
  占比可忽略，重放 P95 实测抖动稳定在 0–1ms。
* 所有时间戳使用 `time.monotonic_ns()` 派生的单调毫秒，不受墙钟回拨影响。
