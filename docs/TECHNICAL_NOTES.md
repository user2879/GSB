# 下发许可令牌系统 — 技术说明

## 1. 环境与落地

- 运行环境：WSL Ubuntu 22.04，Python 3.11（解释器 `3.11.0rc1`），无 Docker。
- 依赖隔离：本机 `python3.11 -m venv` 因缺 `python3.11-venv`/`ensurepip` 且无
  sudo 无法直接使用，改用同解释器的 `virtualenv` 创建等价虚拟环境 `.venv`
  （内含 pip 26.3），激活方式不变：`source .venv/bin/activate`。
- 依赖锁定：`requirements.lock` / `requirements.txt` 固定 `numpy==1.26.4`、
  `pytest==8.2.0`；SQLite 使用标准库 `sqlite3`（实测 3.37.2）。
- Tkinter：系统未安装 `python3-tk`，且当前账号无 sudo 密码，无法安装；
  本系统所有输出均为 JSON/JSONL 与 SQLite，不依赖任何 GUI，故 Tkinter 缺失
  不影响交付能力。如后续需要，执行 `sudo apt install python3-tk` 即可补齐。
- 网络：仅绑定 `127.0.0.1` 高位端口 49101–49103（样本），启动前做占用预检，
  不访问任何外部服务或数据集。

## 2. 包结构

```
src/dispatch_tokens/
  errors.py        # E_SPEC/E_PORT/E_TIMEOUT/E_FIELD/E_LEASE/E_DIFF/E_REPLAY
  fixedpoint.py    # 整数定点口径：截断方向、饱和规则、缩放因子
  clock.py         # DeterministicClock（事件驱动）与 MonotonicClock（真实单调）
  spec.py          # JSON 强校验，首个违规 location+field，整体失败不部分加载
  bucket.py        # 整数令牌桶、余数累积、修订边界结转、补充审计
  lease.py         # 许可令牌、乐观续约、到期判定、唯一归属裁决
  watermark.py     # 定点水位、量化误差界、可区分性决胜
  invalidation.py  # 修订失效传播、稳定拓扑序、逐步依据
  sample.py        # 固定种子样本与 NumPy 整数分组聚合
  protocol.py      # JSON 行帧与握手/下发/租约/关停常量
  worker.py        # 节点子进程服务端（真实 TCP 监听）
  child.py         # python -m dispatch_tokens.child 入口
  client.py        # 主进程真实套接字客户端（套接字级超时）
  supervisor.py    # Popen 编排、端口预检、就绪探测、SIGKILL
  engine.py        # 确定性调度状态机（不做网络 IO）
  ledger.py        # SQLite 原子提交、快照整体替换、恢复重放
  runner.py        # 引擎 + 真实子进程 + 账本的端到端编排
  cli.py           # validate/run/verify/replay
tests/             # 57 个 pytest 用例
samples/demo_spec.json
scripts/demo.sh, scripts/forced_kill_demo.py
```

## 3. 关键语义决策

### 3.1 整数定点与截断（需求 3、7）

- 缩放因子 `S ∈ {1,10,100,1000}`（10 的 0–3 次幂），余额以
  `q = 逻辑令牌 * S` 的整数单位保存。
- 补充公式：`added = (elapsed_ms * rate * S + remainder) // 1000`，
  新余数 `= (...) % 1000`。截断方向显式固定为**向零截断
  `TRUNC_TOWARD_ZERO`**；所有参与量非负，等价于向下截断。
- 余数跨补充轮次累积，因此把事件流切成任意批次、或在事件之间插入任意
  真实等待，只要经过的单调毫秒相同，补充量与余额逐位一致。
- 夹取采用**饱和 `SATURATE`**：余额夹在 `[0, burst*S]`；夹取发生时
  保留分数余数（被夹掉的是整令牌），保证可重放。
- 每次补充输出审计四元组：`elapsed_ms / added / before_clamp /
  after_clamp`，另附 `cap/rate/scale/param_version/clamp_applied`。
- 水位比例 `ratio = balance * S // cap`（整数乘除，无浮点）；单节点量化
  误差上界 1 个比例单位，逐节点比较误差界取 2，全局聚合误差上界 = 节点数。
- 误差界内不可区分时，决胜顺序固定为：可负担优先 → 稳定拓扑序 →
  分片标识字典序 → 优先级 → 需求差，并在结果中标注 `error_influenced`。

### 3.2 修订边界结转（需求 3、6）

- 修订提交时，每个桶先把余额补充到提交时刻，再做一次参数切换；
  因子变化时通过“先反量化再量化”再量化结转，**绝不重算历史补充**。
- 失效传播：节点额度变化 → 该节点持有的分片直接失效，再沿分片依赖图
  向下游 BFS 传播；分片依赖编辑 → 被编辑分片直接失效并向下游传播；
  全局速率/突发/缩放因子变化 → 全部分片直接失效。输出稳定拓扑序与
  每步 `downstream_of:<shard>` 依据。
- 未失效分片的聚合摘要、令牌归属与余量记录不重算，字节保持不变；
  `verify` 命令把增量结果与从空白全量重建逐项比对，差异抛 `E_DIFF`
  并定位到 `shards[<id>].<field>`。

### 3.3 许可令牌与唯一归属（需求 4、5）

- 令牌字段：`token_id/shard_id/node_id/granted_ms/expires_ms/
  renew_count/token_version`，另记 `demand/active`。
- 节点维度与分片维度双重互斥：同一节点同一时刻只能有一个活跃令牌，
  同一份额度不能被两个分片或两个节点同时占用，冲突抛 `E_LEASE`。
- 续约为乐观版本控制：请求必须携带期望版本，节点子进程通过真实 TCP
  `DT_LEASE_CHECK` 校验；版本不符抛 `E_LEASE`，**不递增版本、不延长
  到期时刻**，裁决表记录 `RENEW_REJECTED`。
- 取消只对活跃令牌做一次额度回收并留 `CANCEL` 裁决；同刻修订只做
  桶结转与失效推导，不重复回收同一额度。同到达序号事件按
  `(seq, shard_id, 事件类型序, node_id)` 稳定重排，结论与事件物理
  到达顺序无关。

### 3.4 真实回环协议与错误码（需求 2）

- 子进程 `Popen` 启动后在声明端口 `listen`，主进程 `connect` 后先发
  `DT_HANDSHAKE`，再发 `DT_DISPATCH`；回执含 `node_id/shard_id/
  param_version/summary/elapsed_ms`，`elapsed_ms` 来自节点侧
  `time.monotonic_ns()`。
- 所有超时挂在真实套接字 `socket.settimeout` 上（连接、握手、读取、
  租约校验），不使用线程超时。
- 端口占用/拒绝 → `E_PORT`；套接字握手或读取超时 → `E_TIMEOUT`；
  回执缺字段/帧缺字段/连接在完整帧前关闭 → `E_FIELD`；
  令牌版本冲突 → `E_LEASE`。

### 3.5 SQLite 账本与强制终止恢复（需求 8）

- 每个分片结局在**单个事务**内写 `shard_outcomes` + `refill_audit`
  + `decisions`，`meta` 表携带整数 `format_version=1` 与声明 SHA-256
  `spec_digest`。
- 重新打开时若格式版本或摘要与当前声明不一致，**整体替换**全部表，
  不做增量合并。
- 节点被 `SIGKILL`（退出码 -9）后不生成退出报告；恢复时把在飞分片
  标记为 `UNKNOWN`，登记 `UNKNOWN_RERUN_RECLAIM_FIRST`（先回收额度
  再重跑），再从空白全量重放，与已完整提交记录逐字段比对，首个不一致
  抛 `E_REPLAY` 并定位 `shards[<id>].<field>`。
- 网络聚合 `summary` 由真实子进程产生，纯引擎重建不产生该字段，因此
  账本重放比对覆盖 `node_id/status/param_version/token_id/token_version/
  ownership/bucket`；聚合摘要的确定性由固定种子与 NumPy 整数聚合的
  校验和单独保证（重复生成逐字节相等）。

## 4. 复现实验步骤

```bash
source .venv/bin/activate
pip install -r requirements.lock          # 已预装
python -m pytest                           # 全量测试
bash scripts/demo.sh                       # 六段端到端演示
python scripts/forced_kill_demo.py         # 强制终止恢复单独演示
```

## 5. 实际验证结果（本机执行）

- `pytest`：57 passed（含真实 TCP 子进程、E_PORT/E_TIMEOUT/E_FIELD/
  E_LEASE/E_DIFF/E_REPLAY 路径）。
- `bash scripts/demo.sh`：六段全部 OK；离线与真实回环两次运行的
  余量/归属/裁决逐字节一致，增量 vs 全量重建 E_DIFF-free，账本重放
  E_REPLAY-free。
- 强制终止：`node-a` 被 SIGKILL，退出码 -9，在飞 `sh-3` 标记
  UNKNOWN 并登记先回收额度，重放与已提交记录一致，产物见
  `artifacts/demo.kill.recovery.json`。
- 续约：真实节点 RPC 同时演示成功（期望版本匹配）与 E_LEASE
  （期望版本 99 不符），失败路径到期时刻保持不变。
