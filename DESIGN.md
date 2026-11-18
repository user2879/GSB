# 跨组件配置分发与一致性审计平台 — 技术说明

## 1. 总体架构

```
samples/                确定性样本（节点注册表 + 三版清单 + 非法清单）
ccdplatform/
  errors.py             统一业务错误码（E_*）
  model.py              不可变数据模型（含编译产物逐字节 canonical 表示）
  manifest.py           清单/注册表加载与强校验（E_MANIFEST，带字段路径+版本号）
  fixedpoint.py         定点量化、整数乘加、误差界（NumPy 批量化）
  graph.py              派生 DAG：稳定拓扑序、环检测、下游失效闭包、可达矩阵
  compiler.py           增量编译器 + 依赖感知分发计划切片
  node_simulator.py     独立子进程节点（回环 TCP JSON 行协议）
  orchestrator.py       节点编排：端口预检/启动/握手（E_PORT/E_NODE/E_HANDSHAKE）
  dispatcher.py         有界并发、排队、超时重试、迟到回执排空、单调时钟记录
  convergence.py        期望摘要重算与 E_CONVERGE 收敛判定
  snapshot.py           SQLite 快照、内容摘要、整体替换恢复
  replay.py             恢复后全量重编译重放、逐项/逐字节比对（E_REPLAY）
  injection.py          三类异常注入（E_RANGE/E_CYCLE/E_GRID）
  engine.py             状态机 + 端到端编排
  structured_log.py     JSONL 结构化日志（首次/重试回执分别成条）
  cli.py / gui.py       命令行入口 / Tkinter 界面
tests/                  34 个 unittest（含真实子进程集成测试）
runtime/                运行产物（SQLite、JSONL 日志），演示前清空
```

## 2. 定点量化口径（要求3）

* 缩放因子 `scale = 10^exponent`，启动时强校验为 10 的整数幂。
* 量化 `q(x) = trunc_toward_zero(x * scale)`，截断方向显式由
  `fixedpoint.trunc_zero` 实现（负数向零而非 Python `//` 的向下取整）。
* 派生 `y = Σ c_i x_i` 的整数实现：
  `q(y) = trunc_zero( Σ q(c_i)·q(x_i) / scale )`，全程整数，除以 scale
  回到同一格度。
* 误差界以**整数格**表示。基础项单次量化界为 1 格；线性派生项按系数
  绝对值加权合成，并叠加系数量化与最终 rescale 截断：

  ```
  B(y) = 1 + Σ_i ( ceil(|q(c_i)|·B(x_i)/scale)
                   + ceil(|q(x_i)|/scale) )
  ```

  每个上游的分项贡献（`contributions`）与完整叶→根传播路径
  （`propagation_path`）随编译产物输出。scale 改变时所有格点整数等比
  缩放，物理值口径保持不变（测试 `test_scale_changes_interpretation`）。

## 3. 增量更新与失效传播（要求4）

* 每个配置项维护指纹 `item_fingerprint(value, targets, derivation)`。
* 仅指纹变化的键进入重编译集合；在 DAG 上求直接+间接下游闭包
  （`graph.downstream_closure`，稳定拓扑序）标记失效并级联重编译。
* 未变化项原样复用上一版 `CompiledItem`，其 `canonical_bytes()`
  （确定性 JSON）逐字节不变，测试 `test_unchanged_item_byte_identical`
  与重放逐字节校验双重保证。
* 删除仍被引用的键在编译前以 E_REF 拒绝并列出引用者。

## 4. 分发计划与有界并发（要求5）

* 计划按节点组织。由于节点按批独立重算派生摘要，切片采用**依赖感知**
  策略：基础项按 `batch_size` 切；每个派生项携带其在该节点目标集合内的
  完整传递上游闭包独立成批（正确性优先，必要时允许超批）。
* `Dispatcher` 用 `BoundedSemaphore(max_concurrency)` 约束真实在飞任务，
  配额耗尽时任务在队列等待，`peak_inflight` 被断言永不超预算。
* 每批按 `timeout_ms` 设 socket 超时，失败按 `max_retries` 重试；
  排队、出队、每次尝试的单调时钟毫秒全部落 `DispatchRecord` 与日志。
* 重试的迟到回执由 `_drain_stale` + seq/attempt 匹配丢弃，避免陈旧报文
  污染当前结果；收敛只采信最终成功回执。

## 5. 回执协议与收敛（要求2、6）

* 节点 accept 后先发 `hello` 握手；每批回 `receipt`
  （node/version/digest/seq/attempt）。
* 平台用与节点完全一致的整数口径重算每节点期望摘要（SHA256 over
  规范化 `{node, version, q}`），未回执或摘要不符统一 E_CONVERGE，
  稳定列出偏差节点、期望/实际摘要、判定所用 seq/attempt。
* 端口占用 E_PORT、子进程启动失败 E_NODE、握手超时 E_HANDSHAKE
  分别在编排器三个阶段产生并写结构化日志。

## 6. 快照、恢复与重放（要求7、8）

* SQLite 表：`meta / versions / compiled_items / receipts /
  dispatch_records / transitions / error_events`。
* 每版保存清单 JSON、格度、计划、图、指纹、编译产物 canonical 字节与
  内容摘要（SHA256 over 规范化状态）。
* 恢复时先重算摘要并与快照比较，随后**整体替换**内存中的配置项、派生图
  与编译器记忆（绝不增量合并），旧快照库保持只读可继续演进新版本。
* 重放以快照内清单 JSON + 格度驱动**全新编译器**全量重编译，比对内容
  摘要、逐字节产物、失效集合、回执收敛；首个差异定位到
  `(配置项键, 字段)` 并报 E_REPLAY（测试含篡改量化值的定位用例）。

## 7. 异常注入（要求8）

`injection.py` 只改输入文件或快照参数，不改业务代码：

| 注入 | 手段 | 业务码 |
| --- | --- | --- |
| range | 把目标键改写为超出定点物理范围的值 | E_RANGE |
| cycle | 在最新版构造 α↔β 二元派生环 | E_CYCLE |
| grid | 删除快照 `meta.grid` 行 | E_GRID |

## 8. 状态机

`INIT → LOADED → COMPILED → DISPATCHING → CONVERGED → PERSISTED`，
失败分支 `DISPATCHING → FAILED → PERSISTED`；恢复/重放分支
`LOADED → RESTORED → REPLAYED → COMPILED`。所有迁移经白名单校验并写
`transitions` 表与 JSONL 日志。
