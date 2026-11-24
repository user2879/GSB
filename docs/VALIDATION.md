# 验证记录（实际执行结果）

## 环境

- OS：WSL2 Linux（交付目标为 Windows 本地回环；代码仅用跨平台标准库 + NumPy）
- Python：3.11.0rc1（满足 ≥ 3.11）
- 依赖：numpy 1.26.4；pytest 8.2.0（开发依赖）
- Tkinter：当前 WSL 缺少 `libtk86.so` 无法起窗口；GUI 数据构建函数不依赖 Tk，
  `launch_gui` 惰性导入 Tkinter 并在不可用时给出明确降级说明，Windows 官方
  安装包内置 Tkinter 可直接打开界面。无显示环境使用 `dashboard --text`。

## 复现命令

```bash
python samples/generate_samples.py     # 生成确定性样本（已随库提供）
python scripts/check_ports.py --requests samples/requests.json \
  --workers samples/workers.json --budget samples/budget.json
python -m pytest                        # 单元/集成测试
python scripts/demo.py                  # 端到端可重复演示，完整输出见 docs/demo_output.txt
```

## 业务码验证结论

| 注入/场景 | 期望码 | 实测 |
| --- | --- | --- |
| 请求流损坏（arrival_seq=-1） | E_SPEC | E_SPEC，定位 `requests[0].arrival_seq` |
| 重复投递引用悬空 | E_CHAIN | E_CHAIN，定位批/请求/`duplicate_ref` |
| 账本内容摘要被篡改 | E_SNAPSHOT | E_SNAPSHOT，字段 `content_digest` |
| 恢复重放篡改批摘要 | E_REPLAY | E_REPLAY，定位首个分歧批与 `output_digest`（见 `test_replay_matches_and_detects_first_divergence`） |
| 独立验证正常账本 | OK / E_VERIFY 守卫 | verify status=match；违规路径由 `VerifyError` 覆盖 |
| 回执未知版本/缺字段/负耗时/名次越界 | E_FIELD | 由 `tests/test_receipts.py` 覆盖 |

## 关键验收点实测

- 三格式归一：r1 补齐 6 个字段、r2 补齐 3 个字段、r3 无补齐；每条补齐带显式
  默认值与依据，可逐条回溯。
- 确定性：重复构建批计划 `byte-identical rebuild: True`；乱序工作器声明与
  字典遍历顺序下批划分/批标识一致（`test_plan_independent_of_dict_and_dispatch_order`）。
- 余数批：11 个请求下生成 `batch-worker-b-0001-...`（`remainder=True`），
  携带显式依据，不丢请求不改顺序。
- 真实回环：`WorkerService` 监听 127.0.0.1，`WorkerClient` 长度前缀 JSON 帧
  真实往返；`test_timeout_is_real_and_retried` 验证 50ms 超时触发 3 次幂等重试
  并抛 E_TIMEOUT；并发测试验证全局上限 2、w-a 上限 1、w-b 上限 2 不被突破。
- 失效传播与仅失效重算：worker-a 种子/参数版本变化仅失效其 2 个批与 6 个请求；
  worker-b 的 2 个批 `preserved byte-identical: True`（逐字节比对）。
- 账本：单事务逐批提交；崩溃模拟只提交首批后重开，其余批进入
  `unknown_batches` 要求重跑；快照不一致生成 `0o444` 只读备份并整体替换。
- 重放/独立验证：`replay status: match`、`verify status: match`。

## 测试结果

```
36 passed in ~2.4s
pyflakes: clean
compileall: ok
```

测试分布：强校验 7、回执 8、计划/推理 5、失效 4、网络/调度 4、账本/重放/验证 6、
GUI/确定性 2。
