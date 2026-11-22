# 服务降级传播与故障域收敛诊断组件 — 技术说明

## 1. 组件定位

与“按请求逐个限流”或“按连续失败计数开合熔断器”不同，本组件解决的是：

> 降级决策如何沿服务依赖拓扑向上游传播，并在故障域边界收敛为唯一、可复核的降级方案。

组件以真实 127.0.0.1 回环连接驱动确定性模拟服务采集结局，再做窗口可用性判定、
拓扑传播、故障域收敛、定点误差裁决与 SQLite 账本化复核。全程不访问外网、
不依赖真实账号或付费资源、不使用容器。

## 2. 目录结构

```
degprop/
  errors.py          业务码 E_TOPO/E_CYCLE/E_CAPACITY/E_SNAPSHOT/E_REPLAY
  fixedpoint.py      整数定点基础设施与三类误差界（NumPy 批量量化/窗口统计）
  models.py          服务/边/事件/故障域/会话参数领域模型
  loader.py          JSON+NDJSON 强校验、DAG 拓扑排序、内容摘要
  outcome.py         固定整数种子 LCG 的四类确定性结局规则
  mock_service.py    真实回环模拟服务（live 监听 + bind-not-listen 拒绝端口）
  collector.py       全局/每域双信号量并发调度与真实 socket 采集
  availability.py    观测窗口成功率定点值与三态判定、降级触发
  propagation.py     反向边传播与故障域两套候选方案收敛/决胜
  ledger.py          SQLite 单事务账本、版本/摘要整体替换、重放校验
  engine.py          编排引擎
  cli.py             命令行与三类异常注入
  viewer.py          可选 Tkinter 窗口（惰性导入）
samples/             确定性拓扑与 NDJSON 事件样本
scripts/             demo.sh / demo.bat / 独立模拟服务入口
tests/               28 个 unittest 用例
```

## 3. 整数定点与误差界推导

缩放因子 `F = 10**k`，`k ∈ {0,1,2,3}`。比例量统一量化为分母 F 的定点整数，
容量计数以 `count*F` 编码，使容量分摊与阈值比较全部为整数运算。

* 截断方向：量化与除法一律 floor（向负无穷）；本组件输入非负，即向下截断，
  量化值不大于真实值，残差写入中间量 `remainder_fixed`。
* 饱和规则：声明域为有符号 64 位 `[0, 2^63-1]`；加/乘越界饱和到 FP_MAX，
  中间量置 `saturated=true`；容量上限×域配额×F 的原始乘积在饱和前由领域层
  守卫，越界直接报 `E_CAPACITY`。

三类误差界（`degprop/fixedpoint.py`）：

1. 单值量化误差：`q = 1/F`（定点单位下为 `1*F`），见 `quantized_interval`。
2. 路径聚合误差：把每个量化量表示为区间 `(value, err)`，沿传播路径连乘，
   一阶区间公式
   `err = (v1*e2 + v2*e1 + e1*e2) / F`（误差上界向上取整），
   见 `path_aggregate` / `interval_mul`，全部中间量随结果输出。
3. 方案判定误差界：两候选方案聚合误差界之和 `J = B_a + B_b`，
   定点差 `|v_a - v_b| <= J` 即判为不可区分，见 `judgment_bound`。

容量跨域分摊为定点整数运算：`share = floor(capacity * quota * F / total_quota)`，
余数按 `(余数降序, 域标识升序)` 稳定再分配，杜绝哈希/枚举顺序依赖。

## 4. 真实回环语义边界

* `connection_refused`：模拟服务额外持有一个“bind 但不 listen”的套接字，
  对该端口的 `connect` 在 Linux 与 Windows 上都得到真实连接拒绝
  （Windows 对应 WSAECONNREFUSED）。组件不模拟、不吞掉该错误。
* `timeout`：服务端对慢/超时分支真实 `time.sleep`，客户端 `settimeout`
  先到期抛出真实 `socket.timeout`；组件从不用本地 sleep 冒充超时。
* 重试只重发同一 `seq` 的相同请求，不改变单次调用语义；连接拒绝不重试。
* 动态拒绝端口号不进入账本，持久化的是稳定的 `port_kind`（live/refused），
  因此跨进程恢复重放可比。

## 5. 双约束并发调度

全局 `BoundedSemaphore(global_concurrency)` 与每故障域
`BoundedSemaphore(domain_quota)` 串行获取，任一耗尽即真实排队，
单调时钟记录 `wait_start_ms/wait_end_ms/wait_ms`，并统计峰值在途数与排队次数。
测试断言全局与每域峰值均不超限。

## 6. 传播与故障域收敛

反向边 BFS（下游→上游），候选集合按标识稳定排序：

* forced 边：无条件把上游纳入降级集合；
* threshold 边：仅当下游定点可用比例 `<` 边阈值时传播，健康上游不并入。

每个出现降级成员的故障域构造两套候选：
`local_priority`（域内容量直接加总）与 `cross_domain_apportion`（配额占比定点分摊）。
比较后按 `(容量降序, 域标识序, 降级集合字典序)` 稳定决胜；落入判定误差界时
`indistinguishable=true` 并保留完整裁决中间量。

## 7. 账本、整体替换与重放

每批次在单个 SQLite 事务内写 `call_outcomes / service_states /
degrade_triggers / propagation_hops / domain_plans / cursor`，并写批次 SHA-256 摘要。
重开快照时若 `format_version` 或 `topology_digest` 不符：先报 `E_SNAPSHOT`，
再整体替换（删库重建，绝不增量合并）。恢复后按账本拓扑版本重放已持久化事件，
逐字段比对，首个不一致定位到 `seq + service_id + field` 并报 `E_REPLAY`。

## 8. 异常注入（仅改输入/参数，不改源码）

| 注入 | 手段 | 业务码 |
| --- | --- | --- |
| `--inject cycle` | 复制拓扑追加成环边 | E_CYCLE |
| `--inject capacity` | 复制拓扑放大容量上限使乘积越界 | E_CAPACITY |
| `--inject snapshot` | 写账本后篡改 meta 摘要 | E_SNAPSHOT（随后整体替换） |
| `--inject replay` | 篡改一条已持久化结局 | E_REPLAY（首个不一致定位） |

## 9. Windows 运行

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m unittest discover -s tests
.\scripts\demo.bat
```

可选 GUI：`py -3 -m degprop --gui`（标准 Windows Python 自带 tkinter；
无显示环境会自动降级为终端输出）。
