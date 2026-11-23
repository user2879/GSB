# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）。

## 环境

```text
Python       3.11.0rc1
NumPy        1.26.4
SQLite       3.37.2
当前执行机  WSL/Linux x86_64
```

实现包含 Windows 专用分支：`CREATE_NO_WINDOW`、`SO_EXCLUSIVEADDRUSE`、Windows 父进程句柄看门狗；本次实际执行环境不是原生 Windows，因此这些分支未在原生 Windows 上实跑。核心协议、子进程、SQLite、端口冲突和强杀恢复均已在当前环境实跑。

## 自动化测试

命令：

```text
python -m unittest discover -v
```

结果：

```text
Ran 32 tests in 12.792s
OK
```

覆盖点：

- 两个集群端口重复定位 `clusters[1].port`；
- 非回环地址、负分片数、布尔冒充整数、父修订不存在、版本倒退、派生环、未知目标/键；
- 清单编辑顺序变化后计划逐字节一致；
- 计划差异在首个分片字段报告 `E_DIVERGE`；
- 三重限制同时生效且工作线程真实并发，排队毫秒有序；
- 占用端口返回 `E_PORT`，缺失回执字段返回 `E_REPLY`；
- 同一请求两次分发聚合摘要完全相同；
- 无关键演进不重算，派生依赖变化重算全部受影响分片并保留依据；
- 目标/分片参数变化产生新增、删除和重算集合；
- 演进后旧快照 `is_stale=1`、`is_current=0`，当前仅一个有效快照；
- 损坏快照文件触发整体替换，随后完整重放；
- 模拟服务聚合差异以 `E_REPLAY` 定位首个分片嵌套字段；
- 真实子进程 `exit(137)` 后仅一个完整账本行，running 分片在恢复时成为 unknown，重放 1 片后补齐剩余 2 片。

## 样本完整 CLI 运行

生成样本：

```text
python scripts/generate_samples.py --output-dir samples
```

首次运行：

```text
python -m config_pipeline run samples/manifest.json \
  --db artifacts/demo.db --snapshot artifacts/demo.snapshot.json
```

实际摘要：

```text
分片数                 10
全部状态               success
账本游标               10
账本格式版本           1
账本内容摘要           0652a6e382b3e78fd46618e8fd2ae275b29fa5dd8880d4722f5e0894c504b74e
快照内容摘要           3b9f51d53514d6c87e7141393cf506868a18a33d18f0eac711dc007115c3c624
全局预算               4
峰值资源占用           3
峰值集群运行数         alpha=2, beta=1
峰值任务运行数         rollout=2, audit=1
线程池实测并发线程     3
```

演进到样本修订 3：

```text
python -m config_pipeline evolve samples/manifest.json \
  samples/manifest.evolved.json --db artifacts/demo.db \
  --snapshot artifacts/demo.snapshot.json
```

实际结果：

```text
重算                   8 个 rollout 分片
保留                   2 个 audit 分片，修订 1、编译摘要与聚合摘要不变
新增/删除              0 / 0
摘要变化               8 个 rollout 分片
失效依据               config/v1->v2/capacity:10->14; dependency:scaled -> capacity; task:rollout
当前账本游标           18
演进账本内容摘要       7dd6f8274d406c663020621c258114147aada8adb68a565222f91d882a19a02a
演进快照内容摘要       8a5a6d9cc194083e400785725a42cef5d94729d67d7a18a6438ea48dcf57e3f5
```

重放：

```text
python -m config_pipeline replay samples/manifest.evolved.json \
  --db artifacts/demo.db --snapshot artifacts/demo.snapshot.json
```

实际重放 10 个当前分片全部一致，返回相同账本游标 18。

## 强制终止实跑

使用 `--crash-after-completed 1` 或测试内子进程：

```text
进程退出码             137
stdout                 空（无退出报告）
完整账本行             1
提交分片               c0::task::s000000, success
磁盘 inflight 状态     running（重启时改为 unknown）
恢复后 unknown         1 个
恢复后重放             1 个成功账本分片
恢复后补齐             2 个分片
最终账本游标           3
```

另以 `SIGKILL` 直接杀死持有模拟服务的父进程，模拟服务看门狗在约 150ms 内退出，声明端口恢复可绑定。

## 说明

上述账本/快照摘要包含世代、提交时间和单调时钟轨迹，因此每次重新执行可能不同；这是内容完整性摘要，不是跨运行复现摘要。分片聚合摘要、编译摘要和计划序列化本身对相同清单是确定的。
