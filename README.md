# LeaseCache

面向构建流水线制品缓存的跨节点租约仲裁中间件。它不逐条登记“缓存命中”，而是把每个制品的缓存复用权建模为同一时刻只能由一个租约持有的容量资源，并在回环 TCP、真实子进程、线程池和 SQLite 账本上完成授予、排队、续约、到期回收、中断恢复与独立重放。

原始任务说明保存在 `docs/TASK.md`；设计与不变量见 `docs/DESIGN.md`；本轮实际验证记录见 `reports/acceptance_summary.json` 和 `VALIDATION.md`。

## 环境

- Windows 10/11 本地运行，不依赖 Docker 或容器。
- Python 3.11+，`python` 与 `pip` 可用。
- NumPy 1.26+。
- Tkinter 随 Python 安装（安装 Python 时勾选 “tcl/tk and IDLE”）。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import numpy, tkinter; print(numpy.__version__, tkinter.TkVersion)"
```

样例占用 `127.0.0.1` 的 `38741-38744`；运行前确保这些端口空闲。

## 快速开始

```powershell
python -m leasecache validate --spec samples\demo_spec.json

python -m leasecache run --spec samples\demo_spec.json --db work\demo.sqlite --workspace work\cache
python -m leasecache run --spec samples\demo_spec.json --db work\demo.sqlite --workspace work\cache
# 第二次应输出 reused_keys=全部、recomputed_keys=[]

python -m leasecache verify --spec samples\demo_spec.json --db work\demo.sqlite
python -m leasecache show --db work\demo.sqlite --events 30

python -m leasecache gui --spec samples\demo_spec.json --db reports\gui.sqlite --workspace reports\gui-workspace
```

Windows 双击 `run_demo.bat` 会执行完整验收并打开 Tkinter 控制台；`run_gui.bat` 只打开 GUI。

## 完整验收

```powershell
python -m leasecache demo --root reports\acceptance
# 或
python scripts\run_acceptance.py reports\acceptance
```

验收自动覆盖：

- 首次构建全部重算，第二次同规格运行零网络请求复用。
- 修改 `obj/lib` 的引用后仅重算 `obj/lib` 与依赖闭包 `pkg/bundle`。
- 删除已声明节点/制品时事务化清理当前投影，保留可独立重放的历史授予事件。
- 编辑后的增量结果与全新数据库全量重建逐业务字段一致。
- 在 2 次真实授予后用 `os._exit(86)` 强制终止，不生成退出报告；重启只使用已提交事务恢复，未知租约重新申请。
- 恢复结果与全量重建逐业务字段一致。
- 三个最终账本均通过只读原始声明与账本的独立重放校验。
- 修改节点种子后重新生成摘要，稳定列出全部受影响制品；真实重启后二次摘要检查为空。

测试：

```powershell
python -m unittest discover -v
```

## 规格格式

```json
{
  "format_version": 1,
  "nodes": [
    {
      "id": "node-a",
      "host": "127.0.0.1",
      "port": 38741,
      "capacity_slots": 4,
      "lease_limit": 2,
      "random_seed": 101
    }
  ],
  "pipeline": {
    "id": "demo-pipeline",
    "global_concurrency": 3,
    "lease_ms": 350,
    "stages": [
      {
        "id": "source",
        "concurrency": 2,
        "artifacts": [
          {"key": "src/app", "references": []}
        ]
      }
    ]
  }
}
```

强校验按输入顺序返回首个 `E_SPEC`：`field` 定位首个违规字段，`expected` 给出期望约束。校验项包括唯一节点/制品键、`127.0.0.1` 字面量、全局唯一高位端口、正整数容量/租约上限/配额、种子范围、引用存在性与无环性。

## 关键语义

- 副本摘要规则固定为 `SHA256(canonical-json({format,seed,key,vector}))`，向量由 NumPy `PCG64(SeedSequence([seed, sha256(key)[:8]]))` 生成。
- 每个节点的制品文件真实写入工作区：`<workspace>/node_cache/<node>/<sha256(key)[:24]>.artifact`，同时原子写 `manifest.json`。
- 规范副本摘要取节点标识序最小节点的当前摘要。复用来源集合是所有摘要与引用均等于规范值的节点；若引用闭包也必须可复用，则 `reusable=true`。
- 唯一服务节点按“摘要匹配 → 剩余容量降序 → 节点标识升序”选择；不满足条件的节点保留否决理由。所有候选都不可用时进入容量队列，不绕过容量或租约上限。
- 每次授予都写入全局唯一制品 claim、节点占用、租约集合、来源与进度游标；同一制品同一时刻不能有两个容量所有者。
- 到期由本地 `time.monotonic_ns()` 真实计时触发；SQLite 事务只允许 `ACTIVE` 租约迁移一次，因此到期回收与配额回收竞争时只有一个赢家。
- 中断恢复把物理所有权已不可确认的 `ACTIVE/PENDING` 租约标为 `UNKNOWN`，释放其逻辑容量并使相关复用结论失效，随后必须重新申请。
- 账本含整数 `format_version=1`、`schema_version=1` 与全账本 `content_digest`。`--replace` 先在临时数据库完整重建，再原子替换目标文件。

## 代码结构

- `leasecache/spec.py`：JSON 加载与强校验。
- `leasecache/digest.py`：NumPy 确定性副本与工作区文件。
- `leasecache/node_server.py` / `node_manager.py`：回环 TCP 子进程节点。
- `leasecache/ledger.py`：SQLite 事件账本与容量/租约/来源投影。
- `leasecache/arbiter.py` / `planner.py`：唯一节点仲裁与复用闭包。
- `leasecache/scheduler.py` / `expiry.py`：真实线程池调度、配额排队与单调时钟到期。
- `leasecache/verifier.py`：仅读声明与账本的独立重放校验。
- `leasecache/engine.py`：运行编排、报告与业务快照比较。
- `leasecache/gui.py`：Tkinter 桌面控制台。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明/流水线校验失败，定位首个字段 |
| `E_PORT` | 回环端口占用或子进程绑定失败 |
| `E_HANDSHAKE` | 真实套接字握手超时/失败 |
| `E_ACK` | 回执缺字段、格式错误或身份不一致 |
| `E_LEASE_LIMIT` / `E_CAPACITY` | 节点配额或容量不足，必须排队 |
| `E_QUEUE` | 容量队列超过等待时限 |
| `E_VERIFY` | 独立重放发现首个制品/节点/字段差异 |
| `E_LEDGER` | 账本版本、事务或路径错误 |
