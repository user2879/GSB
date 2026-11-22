# 验证记录

环境：Python 3.11（标准库 sqlite3/socket/threading/json 可用）、NumPy 1.26.4、
仅使用 127.0.0.1 高位端口 19001-19003、19101-19102（启动前预检，全部空闲）。
当前 Linux 容器缺少 Tcl/Tk 原生库，Tk 仪表盘走懒加载；Windows 安装时勾选 Tcl/Tk 即可用，
无头视图与全部领域组件不依赖 Tk。

## 自动化测试

`python -m pytest tests/ -q` 结果：**35 passed**。

- `tests/test_core.py`（19）：强校验 E_RULE、稳定哈希顺序无关/客户端亲和/零权重不可选/
  权重分布、定点 SLI 与预算、量化误差上界、样本不足、熔断三态与半开重试。
- `tests/test_integration.py`（14）：真实回环分发、跨运行实例序列一致、真实连接错误
  E_UPSTREAM、真实 socket 超时分类、权威重算逐字节一致、缓存篡改 E_PREVAIL、
  重放篡改 E_REPLAY、规则演进整体置陈旧、强制终止未知请求与恢复、
  快照摘要守卫 E_SNAPSHOT、上游不可达与规则损坏注入。
- `tests/test_numpy_crosscheck.py`（2）：定点整数 SLI 对 NumPy 浮点的误差 ≤ 1/SCALE，
  以及视图 p95 与 `numpy.percentile` 一致。

## 场景演示（确定性输出）

`python -m meshbench.cli demo` 顺序执行三场景：

1. 正常分发：54 个请求、3 个窗口，缓存视图以单调时钟计时渲染。
2. 熔断：独立账本下 `payments-a` 触发 OPEN（真实单调时钟冷却 400ms），
   冷却后探测 HALF_OPEN 路径被执行。
3. 规则演进：v1→v2，稳定列出被替换窗口
   `orders-api:v1:w16`、`orders-api:v1:w17` 与 4 个受影响实例；
   权威重算与缓存逐字节匹配（约 1ms）。

## 强制终止恢复

`python scripts/crash_recovery_demo.py`：

- Phase A 提交两批（cursor=29），随后制造 4 个仅存在于 pending 表、无提交批次的在途请求，
  模拟 kill -9，不生成退出报告。
- Phase B 全新恢复：哈希链与 `format_version` 校验通过，逐窗口重放
  `byte_for_byte_replay_ok=true`，4 个请求列入 `unknown_requests` 要求重跑。

## 异常注入（结构化日志 logs/meshbench.log）

`python -m meshbench.cli inject {fault}`：

| 注入 | 错误码 |
| --- | --- |
| `rule-corrupt`（临时副本，不污染样本） | `E_RULE` |
| `upstream-down`（无监听预检） | `E_UPSTREAM` |
| `snapshot-corrupt`（账本头摘要被篡改） | `E_SNAPSHOT` |

## 计时与真值来源

`view_timings` 表记录每次 `view_query` 与 `authoritative_recompute` 的单调时钟毫秒差；
权威真值来源为账本逐次请求记录，重算只读账本，不依赖在线状态。

## 复现步骤（Windows）

```bat
scripts\setup_windows.bat
scripts\run_demo.bat
scripts\crash_recovery_demo.bat
python -m meshbench.cli inject rule-corrupt
python -m meshbench.cli recover --rules samples\rules_v2.json --rules-version 2
```
