# 实际验证记录

验证日期：2026-11-19 至 2026-11-20（Asia/Shanghai）

## 环境

```text
Python 3.11.0rc1
NumPy 1.26.4
PyYAML 6.0.3
Linux/WSL Ubuntu 22.04，本地 127.0.0.1 TCP 回环
无 Docker、无外部模型权重、无外部数据下载
```

Python 虚拟环境使用本地 NumPy wheel 和 PyYAML wheel 安装，项目以 editable 模式安装。

## 自动化测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：

```text
Ran 26 tests in 43.152s

OK
```

覆盖内容：

- 2/3/4/6/8/12/16 位对称与非对称量化往返；
- 压缩包长度与数值无关、重复解压逐元素一致、头/负载 CRC 和期望元数据校验；
- 参数与伪梯度重复运行逐元素一致；
- 工作号/张量维度的 `E_PARAM_NONFINITE` 记录；
- 候选评分、预算过滤、字节降序与同字节张量名决断；
- 真实回环套接字全连接收发与环形 allgather；
- 预算耗尽记录窗口、链路和已发送字节；
- 单链路注入失败隔离并继续其他发送；
- `communicated` 阶段崩溃后恢复且不重复收发；
- 配置/拓扑哈希不匹配拒绝续跑；
- 多步压缩窗口（`window_steps=2`）的窗口边界恢复；
- 子进程收到真实 `SIGINT` 后完成当前窗口、提交 `tracked` 快照并以退出码 0 退出，随后可无损续跑；
- 并发度 1 与 4 最终参数哈希、发送记录顺序和误差归因顺序一致；
- 三阶段误差、首次超限阶段、低位宽边界判定；
- 保真梯度轮转和快照验证。

## 样本端到端运行

样本参数：3 个工作号、2 层、隐藏维度 8、4 个训练步、环形拓扑、每链路预算 4096 字节、窗口预算 65536 字节、容差 0.001、保真代数 2、并发 4。

第一段命令：

```bash
.venv/bin/dp-grad-sim run \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots --max-windows 2
```

结果：

```text
completed_steps=2
status=stopped
stopped_reason=max_windows
snapshot_version=8
snapshot_phase=tracked
error_count=0
parameters_equal_across_workers=true
```

恢复命令：

```bash
.venv/bin/dp-grad-sim run \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots
```

结果：

```text
completed_steps=4
status=completed
snapshot_version=16
snapshot_phase=tracked
error_count=0
parameters_equal_across_workers=true
```

最终样本指标：

```text
每窗口选择：asymmetric-8
每窗口聚合顺序：bytes_desc
每窗口实际线路字节：2928
每窗口最大单链路字节：976（预算 4096）
参数哈希（工作号 0/1/2）：
  beb51c1206d4fa20b6d80530404d15eade0799e8bcd9427935c020e7d4ef458c
保真 ID：
  fidelity_w000002_s000003
  fidelity_w000003_s000004
轮转事件数：2
边界张量数：39（跨 4 个窗口）
```

边界样例：

```text
layer_0.bias    current=8 within tolerance, lower=6 exceeds, source=quantization
layer_0.weight  current=8 within tolerance, lower=6 exceeds, source=quantization
layer_1.bias    current=8 within tolerance, lower=6 exceeds, source=fixed_point_aggregation
layer_1.weight  current=8 within tolerance, lower=6 exceeds, source=quantization
```

## 验证工具结果

命令：

```bash
.venv/bin/dp-grad-sim verify \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots
```

结果：

```text
ok=true
checks=19
failed_checks=[]
completed_steps=4
snapshot_version=16
parameters_equal_across_workers=true
```

通过的检查包括：快照存在、配置/拓扑/数据哈希匹配、阶段合法、游标范围、参数有限、参数张量清单、窗口记录唯一且有序、完成窗口数量、窗口预算与账本一致、决策与发送记录数量对应、确定性报告排序、保真代数上限、保真 ID 唯一、保真张量形状、压缩包可解压且长度符合决策、保真引用可重建、当前保真条目被报告引用。

## 结论

本地样本已实际验证：

- 定点压缩包经 `127.0.0.1` 套接字完成工作号间收发；
- 每窗口位宽与聚合顺序基于预算和误差容差作出，并逐条记录；
- 量化打包、解压和跨工作号定点聚合三个阶段均产生可核对误差；
- 低位宽边界张量与误差来源可区分；
- 阶段快照中断后恢复完成，未重复已完成窗口；
- 最终三个工作号参数哈希一致，验证工具全部检查通过。
