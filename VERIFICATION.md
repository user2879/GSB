# 实际验证记录

验证环境：Python 3.11.0rc1，NumPy 1.26.4，Linux（WSL Ubuntu 22.04 容器外环境）。
代码面向 Windows 本地环境编写（`sys.executable` 子进程、跨平台 `os.kill`、
Tkinter 懒加载），全部验证使用 `127.0.0.1` 回环高位端口，无容器、无外部访问。

## 1. 单元与端到端测试

```
cd tests && python3 -m unittest discover -v
Ran 35 tests — OK
```

- `test_spec.py`（13 项）：E_SPEC 首个违规字段与行号、E_QUOTA 份额和超限。
- `test_arbiter.py`（8 项）：逐字节确定性、配额耗尽拒绝、排队→授予与等待时刻、
  突发额度标记、ejected 拒绝、unhealthy 排队、在途上限不变式。
- `test_propagation.py`（4 项）：依赖图引用完整、受影响集合稳定有序、
  未受影响请求逐字节不变。
- `test_e2e.py`（9 项，真实套接字+子进程）：完整运行+重放一致、两次运行内容
  摘要逐字节相同、篡改账本触发 E_REPLAY 并定位请求/租户/字段、强制终止恢复、
  三类异常注入、快照版本识别与整体替换、NumPy 结局分布聚合。

## 2. 验收流程（确定性样本 seed=7，40 请求，4 实例）

```
python3 scripts/gen_sample.py
python3 -m quota_workbench run  ...            -> {"status":"ok","requests":40,
  "content_digest":"dc789867f017f954d8a923a59fd206dd2e2aa5f0d666b2fe815728cb389151b3"}
python3 -m quota_workbench replay ...          -> {"status":"replay_ok","requests":40,"transitions":7}
python3 -m quota_workbench snapshot-verify ... -> {"status":"snapshot_ok","diffs":[]}
```

健康迁移（真实套接字判定）：svc-a `unknown→healthy`（ok）；svc-b（延迟 5000ms，
超时 150ms）`unknown→unhealthy→ejected`（recv_timeout）；svc-c（错误返回）同路径
（bad_response）；svc-d（连接拒绝模式）同路径（connect_timeout，本环境对闭合端口
的 connect 由真实套接字超时判定；Windows 上通常为 connect_refused，二者均为真实
套接字判定）。仲裁结局分布：granted 14 / rejected 26，含排队→授予（等待起止
毫秒已记录）、burst_used=1 授予 7 笔、alpha 配额耗尽拒绝 2 笔。

## 3. 强制终止与恢复

```
run --batch-size 5 --crash-after-batch 2   -> 进程退出码 3（os._exit 模拟强杀）
recover                                     -> {"finalized_committed":10,
  "unknown_requests":["req-010"], "note":"...no exit report is produced for the killed run"}
```

崩溃点已提交批次中处于 in_flight 的 req-010 被标记 unknown 并要求重跑。

## 4. 异常注入（均写入结构化日志 work_accept/run.log）

- `--inject port-occupied` → E_PORT：`instance 'svc-a' failed to bind
  127.0.0.1:18301: BINDFAIL [Errno 98] Address already in use`
- `--inject quota-overflow` → E_QUOTA：`share_sum 101 > total_share_cap 100`
- `--inject snapshot-corrupt` → E_SNAPSHOT：`diffs=["snapshot_digest"]`

## 5. 快照版本识别与整体替换

将 `tenant_set_version` 改为 2 后 `snapshot-verify --replace`：
差异列表 `["tenant_set_version","spec_digest"]`（稳定排序），账本文件被整体删除，
退出码 2 报 E_SNAPSHOT，不做增量合并。

## 6. Tkinter 界面

本验证环境无显示服务器且未安装 tkinter，已验证 `quota_workbench.gui` 模块可导入
（tkinter 在使用时懒加载）。Windows 桌面执行 `python -m quota_workbench gui
--db work\ledger.db` 或 `scripts\demo_gui.py` 即可打开只读诊断界面。
