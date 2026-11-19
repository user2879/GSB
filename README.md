# GSB：多人协作对局状态同步管线与验证工具

面向“不同版本、不同传输格式的协作端与权威端达成一致状态”的确定性数据处理管线。
纯本地运行，不依赖远程服务、真实账号或外部数据附件；样本由固定种子程序化生成。

## 环境

- Python 3.11+（Windows 命令行直接 `python` 即可，无容器依赖）
- 虚拟环境安装依赖：`pip install -r requirements.txt`（numpy，用于固定种子随机流）

## 一条命令复现验证

```bash
python -m gsb.cli run --out artifacts/runs/r1
```

固定序列：生成五局样本（正常局 normal、混格式局 mixed、版本升级局 upgrade、
并发冲突局 conflict、强制终止局 kill）→ 三格式解析与等价性核对 → 迁移幂等检查 →
因果排序合成 → 压缩合并 → 跨版本追赶 → 强制终止恢复 → 独立验证器逐版本比对。
同一命令两次运行的 `report.json` 逐字节相同。

单元测试：`python -m unittest discover -s tests -v`（15 项）。

## 三种承载与统一事件模型

- 自有 JSON 行（`gsb/formats/jsonl.py`）：必需字段按 `schema_version`（v1/v2/v3）分档校验。
- 紧凑二进制帧（`gsb/formats/binary.py`）：`GB` magic + 版本标记 + 字节序标记 +
  索引宽度（2/4）+ 字段位图；字节序/宽度/位图与声明不一致时在解析阶段返回
  `E_FORMAT_FIELD` 并指明帧内偏移，不进入合成阶段。
- 旧版 JSON（`gsb/formats/legacy.py`）：数字型 `peer` 规范化为 `peer-<n>` 字符串，
  毫秒 `ts_ms` 按 `TICK_MS=1000` 换算为 tick。

三者统一解析为 `gsb.model.Event`（tick、peer_id、local_seq、op_type、payload、
base_snapshot_version、schema_version、payload_encoding、time_base）。

## 关键语义

- 迁移（`gsb/migrate.py`）：固定迁移表 v1→v2→v3，逐项可审计（字段名、来源版本、
  默认值）；只补缺失字段，重复迁移内容哈希不变（幂等）；高于支持上限返回
  `E_SCHEMA_TOO_NEW`，不猜测字段语义。
- 等价性（`gsb/equivalence.py`）：逐对格式（jsonl~binary、jsonl~legacy、
  binary~legacy）逐字段比对，报告首个不相等字段。
- 因果排序（`gsb/engine.py`）：按 tick 分组，tick 内按 (peer_id 字典序, 本地序号)
  形成提交序；序号跳变 `E_PEER_GAP`，未知基版本 `E_BASE_VERSION`。
- 并发裁决：同 tick 同实体同字段写入按字段级 LWW（提交序定胜负），每次覆盖记入
  `adjudication.jsonl`（胜出/被覆盖操作、来源格式、裁决依据、最终值），可回溯。
- 版本链与追赶：快照按版本号父子成链；落后端只拉取缺失操作日志追赶，追赶哈希与
  目标快照逐字段一致；链长 > 8 时压缩为保留最新 4 个快照，压缩记录进
  `chain_ledger.jsonl`，压缩后全量重建不变。
- 持久化（`gsb/store.py`）：临时文件 + fsync + `os.replace` 原子提交，清单
  `manifest.json` 是提交点；正常退出写 `SHUTDOWN_OK`；强制终止后重启只依据已提交
  清单恢复并回收未提交尾部，不伪造退出报告；快照目录整体替换以版本戳与内容哈希
  判定新旧，拒绝盲目录入。
- 独立验证器（`gsb/verifier.py`）：只消费持久化目录，从链基重放并与在线演进
  逐版本比对状态哈希，定位首个不一致版本及其参与操作与来源格式。

## 实际验证结果（seed=250）

- 五局三格式等价性核对全部 `all_equal=true`；迁移幂等检查 20/20 通过。
- 负向注入：`E_SCHEMA_TOO_NEW`、`E_FORMAT_FIELD`（偏移 3/4/67）、`E_PEER_GAP`、
  `E_BASE_VERSION` 均按约定触发。
- 强制终止局：worker 提交至 v5 后被杀（退出码 137），恢复回收未提交尾部
  （`snapshots/v6.json`、`logs/v6.jsonl.tmp`），自 v5 续跑 31 个事件至 v12，
  验证器逐版本比对一致；normal/kill 局触发压缩（compacted_base=5），重建不变。
- 两次完整运行 `report.json` 逐字节相同（`cmp` 通过）。
- 逐版本哈希清单见 `artifacts/runs/r1/report.json` 各场景 `hashes` 字段。
