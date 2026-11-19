# 状态同步数据处理管线 — 技术说明

面向多人协作对局的确定性状态同步管线。所有产物落在本地工作区，不依赖任何
远程服务、账号或网络附件。固定种子（默认 `20261119`）驱动 NumPy 随机数生成
样本，两次相同输入的整棵产物目录逐字节一致。

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `syncpipe/errors.py` | 7 个业务错误码与结构化错误记录（含版本、tick、事件定位） |
| `syncpipe/state.py` | 纯函数原语：规范序列化、SHA-256 哈希、操作应用、字段级 LWW 裁决、快照构造 |
| `syncpipe/parser.py` | JSONL 解析、非法 tick/序号断裂校验、tick 分组与确定性提交序 |
| `syncpipe/storage.py` | 原子写（tmp+fsync+rename）、版本化快照、CRC 滚动日志、manifest 链清单 |
| `syncpipe/engine.py` | 在线合成引擎：跨 tick 工作列表、挂起/重试、版本推进、快照链、压缩、锚点重建 |
| `syncpipe/recovery.py` | 崩溃恢复：只依据持久化文件，识别强制终止、回放尾部、截断隔离 |
| `syncpipe/validator.py` | 独立验证器：不导入 engine，自 genesis 全量重放并逐版本比对哈希 |
| `syncpipe/generator.py` | 固定种子生成 4 类样本（正常/冲突/追赶/终止） |
| `syncpipe/worker.py` | 可被强制终止的子进程执行器，逐版本原子刷新进度文件 |
| `syncpipe/cli.py` | 命令行入口与端到端 `demo` 编排 |

## 2. 确定性基石

- 所有哈希经 `canonical_bytes`：`json.dumps(sort_keys=True, separators=(",",":"))`
  后 UTF-8 编码再做 SHA-256，键顺序、空格、Unicode 均固定。
- NumPy 标量经 `to_native` 转成原生 JSON 类型，禁止 `NaN/Infinity`。
- 临时文件使用确定名称（`*.tmp`），单写入者无并发命名竞争。
- 持久化内容不含时间戳、PID、绝对路径等环境量；恢复/验证报告中的路径均为
  相对存储根路径或纯文件名。

## 3. 事件排序与因果挂起

- 解析器按 `(tick, peer_id, local_seq)` 升序形成确定性提交序。
- 每个 peer 的 `local_seq` 必须从 0 连续；断裂或非法 tick 判 `E_PEER_GAP` 并隔离。
- 引擎维护跨 tick 的单一工作列表，固定点循环：每轮选取提交序最靠前且
  `base_snapshot_version == 当前权威版本` 的事件应用；无就绪事件时，提交序最靠前
  的滞后事件作为“版本推进器”应用并记录 `E_BASE_VERSION` 漂移，保证权威持续前进、
  其余滞后事件随后满足 base。事件既不静默丢弃也不按到达顺序回退。

## 4. 冲突裁决

- 同 tick 同实体的字段级并发写采用最后写者胜（LWW），定序依据
  `(peer_id, local_seq)`，裁决依据字符串固定为
  `field_lww:commit_order(peer_id,local_seq)`。
- 每条裁决记录被覆盖操作、胜出者与依据，写入日志条目并汇总到 summary。
- 结构冲突（重复 spawn、写/销毁不存在实体、复活墓碑实体）判
  `E_CONFLICT_UNRESOLVED`，事件被拒绝但仍推进一个版本，状态保持不变。

## 5. 快照链、增量追赶与压缩

- 每个事件产生一个版本；偶数版本落快照文件，所有版本都在 manifest 登记
  `state_hash/snapshot_hash/parent_version`，形成连续父子链。
- 任一版本可由祖先快照加操作日志重建（`rebuild_from_anchor`，支持显式锚点）。
- 快照数超阈值时 `compact()` 仅保留 genesis、每 5 个版本锚点与 head，删除中间
  快照文件但保留全部日志与 manifest 版本记录，版本号连续、可重建性不变。

## 6. 持久化、原子性与崩溃恢复

- 快照、manifest、关闭标记均为“写临时文件 → fsync → 原子 rename”，写入完成前
  中断只会留下不被引用的 `*.tmp`，重启时清理，绝不出现半写快照。
- 每条日志带 CRC32；首个 CRC/JSON 失败的行触发 `E_LOG_TRUNCATED`，剩余字节隔离到
  `quarantine/` 并停止该段回放。
- 正常退出写 `shutdown.marker`；强制终止后无该标记，恢复报告显式
  `forced_termination=true` 且不生成任何退出报告。
- 恢复从最新锚点快照回放至最近一致版本，恢复快照写入独立 `recovered/` 目录，不
  污染权威快照链。

## 7. 错误码

| 码 | 触发链路 | 定位信息 |
| --- | --- | --- |
| `E_BASE_VERSION` | base 与权威不匹配、挂起追赶、漂移推进 | 版本 + tick + peer/seq |
| `E_PEER_GAP` | 非法 tick、同端序号断裂、畸形 JSON、缺字段 | 行号或 peer/seq |
| `E_SNAPSHOT_HASH` | 快照内容哈希不匹配、回放哈希不一致 | 版本 + tick |
| `E_LOG_TRUNCATED` | 日志行 CRC/JSON 失败（截断） | 版本 + 段内行号 |
| `E_VERSION_DIVERGE` | 快照目录整体替换后同版本哈希变化 | 版本 + 新旧哈希 |
| `E_CONFLICT_UNRESOLVED` | 不可裁决的结构冲突 | 版本 + tick + peer/seq |
| `E_CHAIN_BROKEN` | 父版本指针断裂、缺锚点 | 版本 + tick |

## 8. 独立验证器

`syncpipe/validator.py` 不导入 `engine`，仅复用 `state` 纯函数与 `storage` 布局
帮助。它校验父子链、每个快照文件哈希、每条日志 CRC，自 genesis 全量重放并与
manifest 逐版本比对 `state_hash`，定位首个不一致版本并输出该版本上下文与参与
事件；另做一次随机锚点重建，比对字段级与哈希级一致性。

## 9. 命令入口（Windows / Linux 通用）

```bash
# 环境（Windows 用 py -3.11 -m venv .venv 后 .venv\\Scripts\\python -m pip install numpy）
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\\Scripts\\python

# 一条命令完成：生成样本 → 合成 → 压缩 → 强制终止 → 截断恢复 → 验证器比对
.venv/bin/python -m syncpipe.cli demo --workspace workspace --seed 20261119

# 独立验证入口
.venv/bin/python -m syncpipe.cli validate --root workspace/runs/normal_game --seed 20261119

# 单元测试
.venv/bin/python -m unittest discover -s tests
```

## 10. 验证结果（实际运行记录）

- 单元测试：23 个全部通过（含 7 个错误码的异常链路覆盖）。
- 端到端 demo：`all_validations_ok=true`，4 类场景全部合成并通过独立验证。
- 强制终止：子进程 `returncode=-15`、无关闭标记、`E_LOG_TRUNCATED` 被识别并隔离，
  恢复版本状态哈希与未中断参考运行逐字节相同。
- 两次完整运行对整棵产物目录 `diff -r` 无差异（逐字节一致）。
- 逐版本哈希见 `workspace/reports/version_hashes.txt`，恢复报告见
  `workspace/reports/recovery_report.txt`。

## 11. 环境偏差说明

任务预期 Python 3.11；当前容器的系统 Python 3.11 缺少 `venv`/`ensurepip` 组件且
无 sudo 权限，因此虚拟环境基于可用的 Python 3.10.12 构建（通过 `get-pip.py` 引导
pip 后安装 NumPy 2.2.6）。代码仅使用 3.10/3.11 共有的语法与标准库 API，在
Windows 的 Python 3.11 上可直接按第 9 节命令运行，无需修改。
