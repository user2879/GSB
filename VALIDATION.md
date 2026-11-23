# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）

## 环境

* Python：3.11.0rc1
* NumPy：1.26.4（已安装；核心实现不依赖浮点随机语义）
* 平台：Linux 容器侧执行；代码路径使用 Windows 兼容子进程、回环 TCP、SQLite 与 Tkinter
* Tkinter：当前容器缺少 `libtk8.6.so`，`python -m eventbus_diag ui` 按设计返回
  可读错误和退出码 2；Windows 官方 Python 安装包自带 Tk，界面代码已通过语法/导入容错验证

## 自动化测试

命令：

```powershell
python -m unittest discover -v
```

结果：`Ran 25 tests in 10.332s`，`OK`。

关键测试：

* 重复端口准确报告 `E_SPEC field=subscribers[1].port line=17`。
* 同一事件集合正序与反序输入产生完全相同的批标识、顺序和批内容。
* 全局上限 2、订阅者上限 1 时，等待线程从未观察到超限。
* 通道速率上限 2 时，第三次派发等待约 1000 ms，等待原因记录为
  `channel_rate_limit`。
* 一个通道的速率窗口耗尽时，另一通道仍可先获得租约并使用空闲线程池槽位，
  跨通道队头阻塞测试通过。
* 错误续约版本返回 `E_LEASE` 且到期时刻不变；正确续约版本递增。
* 20 ms TTL 租约真实过期，回收单调时刻落在测试前后时钟区间内并返回批标识。
* 真实子进程样本完成 6/6 批，账本 6 条租约、6 条配额快照、12 条状态迁移。
* 缺确认字段、占用端口、1 ms 握手超时分别产生 `E_ACK`、`E_PORT`、`E_TIMEOUT`。
* 延迟 250 ms 的服务使 60 ms 租约真实过期，批失败且 SQLite 租约状态为 `EXPIRED`。
* `--crash-after 1` 强制终止后，只保留已提交批；恢复 CLI 将未提交批标记
  `UNKNOWN` 并重跑至 `COMPLETED`，最终无活跃租约、在途批数为 0。
* 篡改已确认批 `ack_digest` 后，恢复 CLI 返回退出码 3，`E_REPLAY` 定位到首个
  受影响批及 `mismatch_field`。
* 使用 `PYTHONHASHSEED=1` 与 `PYTHONHASHSEED=987` 分别生成计划，两个 CLI 输出
  文件逐字节一致（SHA-256 均为
  `633ad18464042d7c75e2c8f669afe2b53c328f22f13d800147cac55dd171af5f`）。

## 样本验证

样本：`samples/spec.json`、`samples/events.ndjson`。

* 声明版本：3
* 订阅者：3
* 通道：3
* 事件：10
* 确定性批：11
* 订阅选择：`ch-alerts -> s-beta,s-gamma`、`ch-orders -> s-alpha,s-beta`、
  `ch-payments -> s-alpha`

完整运行、同版本无变更恢复、强制终止恢复均以 11/11 批确认结束；进度游标归一到
11，运行状态 `COMPLETED`，无活跃租约。样本通道速率上限会在真实滑动窗口上产生
约 1000 ms 的可见等待，证明速率限制不是静态检查。

最近一次完整运行：

* `run`：退出码 0，11/11 确认，耗时 1354.087 ms，最大单批等待 1007.703 ms。
* 同版本 `recover`：退出码 0，11/11 已确认且校验通过，耗时 18.116 ms；账本内容
  摘要与首次完成后一致
  （`60ae4ff6a09ec952b7506211e91b6fafb66c4cfe98e1cbe4c610dffffeafd19e`）。
* `--crash-after 4`：退出码 75，崩溃后仅保留已提交记录；日志仅有预期的
  `E_TEST_FORCED_TERMINATION`，无 `E_PORT`、`E_TIMEOUT`、`E_ACK` 或租约错误。
* 强制终止后在同一 shell 中零等待立即 `recover`：退出码 0，7 个未提交批先标记
  `UNKNOWN` 再重跑，耗时 467.221 ms，最终 11/11 确认并通过重放校验；游标 11、
  在途批数 0、活跃租约 0。恢复启动观察到一次短暂 `RECOVERY_PORT_WAIT` 及随后
  的 `RECOVERY_PORT_READY`，未产生 `E_PORT` 失败，验证了终止后即时端口接管。
* 最终崩溃恢复账本摘要
  ：`183a831cd89bb5b6464f08a4d079f05c3fc051e4fd279544548a3bc8ad35d746`；账本
  格式版本 1，7 条 `UNKNOWN` 迁移和 11 条配额快照与最终确认态一致。

## 已知环境限制

* 当前 Linux 容器缺少 Tk 动态库，因此不能在此环境截图或人工点击 Tkinter 界面；
  `tkinter` 导入失败路径已实测返回预期提示。
* 窗口界面需在 Windows 官方 Python（含 Tk）或安装了 Tk 运行库的桌面环境运行。
