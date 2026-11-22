# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）

## 环境

- 执行环境：本地工作区，无 Docker/容器。
- Python：3.11.0rc1。
- NumPy：1.26.4。
- SQLite：3.37.2。
- 网络边界：仅 `127.0.0.1` 回环，未访问真实站点或账号。
- 样本端口：`47321`、`47322`、`47323`、`47324`。
- 当前 Linux 验收容器缺少系统 `libtk8.6.so`，因此无法启动 Tk 窗口；界面模块完成编译，只读视图模型、刷新摘要稳定性和 UI 命令的无 Tk 友好错误路径已验证。Windows 官方 Python 自带 Tk 时直接使用 README 的 `ui` 命令。

## 自动化测试

命令：

```bash
python3 -m unittest discover -v tests
```

结果：`Ran 36 tests ... OK`，覆盖：

- 服务/实例/端口/权重/修订号/退避的 `E_SPEC` 字段与位置；
- 累计权重、零权重排除、稳定重复选择；
- 熔断触发、OPEN 快速失败、真实冷却、半开成功/失败与试探数；
- 全局/服务双重预算峰值约束和等待时钟；
- 重试退避期间释放预算、最终 `E_EXHAUST`；
- 真实套接字超时 `E_TIMEOUT`、缺应答字段 `E_REPLY`、不可达 `E_HOST`、端口占用 `E_PORT`；
- 整数定点比例/P95/吞吐、误差界、稳定决胜与输入顺序无关；
- SQLite 同事务提交、内容摘要校验、账本篡改拒绝；
- 未提交意图 `unknown` 标记、`exit_report_available=false`；
- 快照摘要/版本不符的 `E_SNAPSHOT`、整体替换与稳定差异；
- 重放一致性和首个差异 `E_REPLAY`；
- 三类 CLI 注入业务码；
- 真实子进程强制终止后恢复与重跑。

## 三场景演示

命令：

```bash
python3 demo/run_demos.py
```

最终输出：`PASS`，验证文件：`artifacts/demo/verification.json`。

### 场景 1：冷启动分发与预算

- 请求：18，成功：18。
- 全局并发限制：4；实测峰值：4。
- 服务配额峰值：`checkout=2/2`、`search=2/2`、`fragile=1/1`。
- 预算授予：18；结束时在飞：0；等待：0。
- 样本选择由 SHA-256 位置落入累计区间，实例和选择序号全部写入账本。

### 场景 2：熔断触发与恢复

- 尝试 1、2：上游声明错误，消耗预算。
- 尝试 3：`circuit_open` / `E_CIRCUIT_OPEN`，`budget_used=false`。
- 断路器达到阈值 2 后为 `OPEN`，剩余冷却约 154 ms。
- 冷却结束后进入 `HALF_OPEN`，声明 1 个探测；`cb-ok-001` 成功，状态恢复为 `CLOSED`。
- 状态迁移：`CLOSED → OPEN → HALF_OPEN → CLOSED`。

### 场景 3：快照、强制终止与重放

快照：

- 有效快照重新打开返回 `valid`。
- 写入合法摘要但错误 `route_revision=999` / 实例版本的快照，正常验证返回 `E_SNAPSHOT` 并整体替换。
- 替换差异：`scalar_changed=["instance_version","route_revision"]`。
- 保留旧摘要修改 payload 的副本读取返回 `E_SNAPSHOT`。

强制终止：

- 在观察到 1 个真实套接字意图后向工作台发送 `SIGKILL`。
- 重启恢复从账本已提交尝试重建状态，标记 1 个未知尝试，重跑 24 个未完成请求。
- 报告 `exit_report_available=false`、`pre_crash_exit_report=null`、`state_source=committed_ledger_attempts_only`。

重放：

- 源请求 12，重放请求 12，实例/尝试/结局逐项一致。
- 比较结果：`status=consistent`、`code=OK`、`skipped_incomplete=0`。
- 专业化篡改首个尝试结局为 `error` 后返回 `E_REPLAY`，定位：
  - `request_id=replay-001`
  - `service=search`
  - `field=attempts[0].outcome`
  - `expected=success`, `actual=error`

## 异常注入结果

实际命令均返回退出码 2，并写入结构化 JSONL：

- `--inject corrupt-rules`：`E_SPEC`，`location=$.route_revisions[1].revision`。
- `--inject port-occupied`：`E_PORT`，`location=127.0.0.1:47321`。
- `--inject snapshot-digest`：`E_SNAPSHOT`，`field=snapshot.digest`，包含 stored/actual 摘要。

## 已确认结论

- 真实回环子进程、真实 TCP 握手/请求/应答和真实套接字超时均工作。
- 全局与每服务预算峰值未越界，退避不占预算。
- 熔断冷却与半开探测由真实时钟驱动，冷却内快速失败不计成功且不占预算。
- 账本尝试记录、结局、预算和选择序号同事务持久化；崩溃后未知尝试被标记且不伪造退出报告。
- 快照/账本版本或摘要不一致时不增量合并，而是整体替换。
- 确定性重放可逐尝试比较，首个差异可定位到请求、服务与字段。
- 刷新只读视图前后账本内容摘要一致。
