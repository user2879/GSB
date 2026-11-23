# Serverless 并发租约与配额归还诊断工作台

这是一个完全本地运行的 Windows 诊断工作台。它把 Serverless 并发视为有硬归还截止的资源租约，以确定性事件推进、SQLite 批次账本、逐事件归属裁决和 Tk 诊断界面回答：

- 每次调用何时取得并发、租约何时到期、并发最终归还给谁；
- 高优先级函数如何在显式抢占关系内借用/抢占低优先级并发；
- 同一函数在到期、借出归还和排队之间如何只裁决一次；
- 固定点水位及量化、借用归还延迟误差如何影响排序；
- 声明版本演进后哪些旧租约/队列/抢占结论失效，哪些仍可沿用；
- 进程被强制终止后，哪些租约只存在于未提交事务中，为什么必须重算。

## 快速开始（Windows）

要求 Python 3.10+、pip、NumPy、Tkinter；SQLite 使用标准库 `sqlite3`，无服务端、账号或网络依赖。

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\generate_demo.py
run_demo.cmd
```

`run_demo.cmd` 会执行两版声明、11 个事件、账本重放、崩溃恢复、三类异常注入和稳定渲染，并在 `artifacts\` 写入结果。

打开 Tk 工作台：

```bat
run_ui.cmd
```

如果官方 Python 安装时未包含 Tk，请重新运行安装器并勾选 **tcl/tk and IDLE**。无显示环境可使用：

```bat
python -m leasebench ui --db artifacts\demo.sqlite --headless
```

## 样本格式

声明 JSON 示例见 `samples/declarations_v1.json`。函数字段为：

```json
{
  "function_id": "api",
  "memory_mib": 1024,
  "instance_concurrency": 2,
  "lease_ms": 700,
  "min_concurrency": 2,
  "max_concurrency": 4,
  "priority": 30
}
```

`priority` 必须全部提供或全部省略；全部省略时由无环抢占图推导。顶层还必须提供 `declaration_version`、`global_concurrency`、`functions`、`preemptions`。`preemption_checkpoint_ms` 默认为 100，用于定义安全检查点边界。

抢占关系：

```json
{"higher": "api", "lower": "report", "return_deadline_ms": 400}
```

事件 NDJSON 每行一个对象：

```json
{"event_seq":1,"arrival_ms":0,"function_id":"api","expected_occupancy_ms":800,"concurrency_demand":4}
```

加载器以 `E_DECL` 整体拒绝并把 `field` 定位到首个违规项，包括重复函数标识、未知引用、抢占成环、布尔冒充整数、负数、最小配额大于最大配额、到达时刻回退、事件序号不连续等。

## 命令行

```bat
python -m leasebench validate --declaration samples\declarations_v1.json --events samples\events.ndjson
python -m leasebench run --declaration samples\declarations_v1.json --events samples\events.ndjson --db artifacts\demo.sqlite --limit-events 6
python -m leasebench run --declaration samples\declarations_v2.json --events samples\events.ndjson --db artifacts\demo.sqlite
python -m leasebench replay --db artifacts\demo.sqlite
python -m leasebench view --db artifacts\demo.sqlite --text-out artifacts\view.txt
python -m leasebench ui --db artifacts\demo.sqlite
```

异常注入：

```bat
python -m leasebench run --declaration samples\declarations_v1.json --events samples\events.ndjson --db artifacts\cycle.sqlite --inject-cycle
python -m leasebench run --declaration samples\declarations_v1.json --events samples\events.ndjson --db artifacts\quota.sqlite --inject-quota
python -m leasebench replay --db artifacts\demo.sqlite --inject-snapshot
```

退出码：`0` 成功，`E_DECL=2`，`E_QUOTA=3`，`E_SNAPSHOT=4`，`E_REPLAY=5`。加 `--log` 会追加 JSON Lines 结构化日志。

## 确定性裁决口径

1. 时间同刻先处理租约回收与借出归还，再处理新到达；同一时刻的释放类别优先级为 `BORROW_RETURN_DUE > PREEMPTED > CALL_COMPLETED > LEASE_EXPIRED`，每份租约只产生一条释放记录。
2. 每个事件只申请一份租约，标识为 `L{event_seq:08d}`，再次授予为 `L{event_seq:08d}Rnn`。标识只依赖事件序号和尝试次数，不依赖函数遍历顺序。
3. 授予同时检查全局预算和函数有效容量。函数有效容量为 `max + borrowed_in - lent_out`。低优先级函数可借出的上限为 `max - max(used, min) - lent_out`，即保留最小配额。
4. 借入只沿显式 `higher -> lower` 关系，且要求优先级严格下降；每笔借入有独立的 `return_deadline_ms`。租约提前释放时借出即刻归还，借出到期早于占用完成时强制释放借用租约并重排原调用。
5. 高优先级短缺且无可借空闲时，选择低优先级活跃租约，在下一个检查点边界安全挂起；选择规则为被抢占函数优先级升序、边界升序、租约标识升序。被抢占调用以 `event_seq` 为身份写入新队列项，`not_before = 边界 + 关系归还截止`，不丢事件。
6. 队列按 `priority` 降序、入队时刻、排队序号、事件序号稳定排序；所有集合在输出前显式排序。

更完整说明见 `docs/DESIGN.md`。

## 固定点水位与误差

会话参数 `--watermark-scale` 必须是 10 的整数幂，默认 1000。比例采用 `FLOOR_TOWARD_ZERO`：

```text
ratio_fp = floor(numerator * scale / denominator)
```

每个值量化误差上界为 1 个定点单位；两个值比较的量化误差段为 2 个定点单位。`borrow_return_error_ms` 窗口内即将到期的借入并发另计：

```text
borrow_delay_error_fp = ceil(near_deadline_borrowed * scale / capacity)
```

总比较误差为量化段加借用归还延迟段，并标记主导段。两项差值落在误差界内时仍按函数标识稳定决胜，同时以 `comparison_affected_by_error=true` 标注判定受误差影响。

## 版本演进与陈旧结论

声明版本只能递增；同版本内容不可改变。配额、租约时长、优先级、抢占关系或全局预算变化时，工作台沿函数、活跃借入和抢占引用闭包推导失效对象。受影响租约变为 `STALE`，受影响队列项作废并重新入队；无关活跃租约可携带到新版本。旧批次水位仍作为审计记录保留，但当前视图整体由新版本状态重建，并稳定列出 `lease/queue/preemption` 对象、原因和版本迁移。

## 账本、崩溃恢复与重放

每个到达时刻的事件组是一个完整批次。批次内申请、授予、回收、借入、归还、抢占、水位、游标、声明和状态快照在同一 SQLite 事务提交。账本包含整数 `format_version=1`、逐批 SHA-256 内容摘要和链式摘要。

处理批次前先单独记录 `pending_batches` 意向。若进程被强制终止，事务回滚，重启只能加载最后完整提交的批次；意向中的候选租约列为 `UNKNOWN_RECOMPUTE_REQUIRED`，从已提交游标后重算。强制终止不保证生成退出报告，本工作台也不作此声明。

`replay` 按账本记录的声明版本、缩放因子和事件重新推进；逐项比较申请、授予、回收、借入、归还、抢占和水位。首个差异以 `E_REPLAY` 返回事件序号与租约标识。

## 验证

本机实际验证结果：

```text
pytest -q
28 passed

python scripts/run_demo.py
status: PASS
replay batches: 9, events: 11
E_DECL=2, E_QUOTA=3, E_SNAPSHOT=4, E_REPLAY=5
forced termination exit: 17; recovery: CLEAN
headless render stable: true
```

详细结果见 `artifacts/demo_report.json`、`artifacts/demo_summary.txt`、`artifacts/view_v2_final.txt` 和 `docs/VALIDATION.md`。当前验证容器缺少 Tk 共享库，因此 Tk 窗口未在本容器启动；界面代码惰性导入，固定文本渲染已在无头模式验证稳定，Windows 官方 Python 自带 Tk 时可直接启动。
