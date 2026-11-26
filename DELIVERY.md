# 交付说明：本地确定性推理服务（回执归一 / 失效重算 / 账本恢复重放）

## 目录结构

- `src/core.py` — 定点口径声明、确定性参数/请求生成、批划分、NumPy 整数定点前向、摘要计算
- `src/receipts.py` — 回执三格式 r1/r2/r3 编解码与强校验（E_FIELD / E_REF，定位源行号与字段名）
- `src/normalize.py` — 三格式归一视图（来源格式、补齐字段集合、补齐依据）与稳定采纳规则
- `src/dag.py` — 工作进程参数版本 → 批次 → 请求的失效传播（三步轨迹）
- `src/worker.py` — 工作进程子进程：127.0.0.1 回环端口行协议服务，stdin EOF 看门狗
- `src/scheduler.py` — 全局并发上限 + 每工作进程配额调度，真实套接字超时（E_TIMEOUT / E_PORT）
- `src/ledger.py` — SQLite 权威账本：单事务逐批提交、格式版本 + 内容摘要、只读备份、快照
- `src/pipeline.py` — 主编排：计划构建 → 失效传播 → 仅失效子集重算 → 流式逐批原子提交
- `src/replay.py` — 恢复后一致性重放校验（E_REPLAY，定位首个不一致的批/请求/字段）
- `src/verify.py` — 增量重算 vs 从零全量重建的逐项比对命令
- `src/samplegen.py` — 固定整数种子的确定性样本（三格式历史回执 + 异常样本）
- `src/demo.py` — 可重复演示脚本（`python3 -m src.demo`）
- `src/ui.py` — Tkinter 诊断界面（只读刷新，渲染顺序稳定）
- `config/pipeline.json` / `config/ports.json` — 业务配置与回环端口声明（1024..65535）
- `samples/` — 生成的确定性样本；`tests/` — 47 项 unittest 测试

## 关键设计

- 定点口径显式声明于 `config/pipeline.json`：`frac_bits=8, word_bits=32, rounding=floor
  （向负无穷取整，算术右移）, saturation=saturate（饱和到 int32）`；口径进入参数摘要，
  回执经参数版本序号 → 参数规格摘要可复核。
- 输出摘要绑定：排序后的参与请求标识集合 + 参数版本序号 + 参数摘要；请求输入仅由
  (样本种子, 请求标识) 派生，与派发顺序、字典遍历顺序、批执行顺序无关。
- 采纳规则：真实参数版本降序（补齐版本视为 -1，不得覆盖高版本真实值）→ 格式版本降序
  → 当期计算来源优先于历史文件 → 源文件/源行号升序；否决回执保留并记录理由。
- 账本每批单事务写入：批次归属、采纳/否决回执、失效传播轨迹、进度游标、内容摘要；
  重开时格式版本或内容摘要不一致 → 整体替换并保留只读备份（`.bak-<时间戳>`，chmod 444）。
- 参数版本序号是账本按变更历史维护的整数簿记；`verify` 按版本无关的规范结论
  （批集合、派工、成员、规格摘要、规范内容摘要）逐项比对增量与全量重建。
- 工作进程父子关系：调度器以 `stdin=PIPE` 托管子进程，kill -9 后管道 EOF 触发
  子进程看门狗退出并释放回环端口（WSL 下 getppid 重挂不可靠，故用管道 EOF）。

## 运行方式

```bash
python3 -m pip install --user numpy==1.26.4
python3 -m src.samplegen                       # 生成确定性样本
python3 -m src.pipeline run --config config/pipeline.json --ports config/ports.json \
    --ledger state/ledger.db --receipts samples/receipts_r1.txt samples/receipts_r2.txt \
    samples/receipts_r3.txt --snapshot
python3 -m src.verify --config config/pipeline.json --ports config/ports.json \
    --ledger state/ledger.db --receipts samples/receipts_r1.txt samples/receipts_r2.txt \
    samples/receipts_r3.txt
python3 -m src.replay --ledger state/ledger.db
python3 -m src.demo                            # 端到端可重复演示
./run_ui.sh                                    # Tkinter 诊断界面（自动补齐 Tk 库路径）
python3 -m unittest discover -s tests -t .     # 测试套件
```

## 实际验证结果（本机 WSL Ubuntu，Python 3.11.0rc1，NumPy 1.26.4）

- `python3 -m unittest discover -s tests -t .`：**47 项测试全部通过**（约 21s），覆盖
  三格式编解码与 E_FIELD/E_REF 定位、定点舍入/饱和、跨顺序确定性、归一补齐与采纳、
  失效传播、账本原子提交/损坏替换/只读备份、调度配额/E_TIMEOUT/E_PORT、
  增量重算 + 全量比对 + 重放、kill -9 恢复。
- `python3 -m src.demo` 实测输出要点：
  - 首次运行提交 6 批，采纳回执全部为 r3（24 条），历史 r1/r2/r3 共 21 条被否决并记录理由；
  - 修改 w1 种子后仅 `batch-0001/0003/0005` 及其 12 个请求失效重算，
    未失效批摘要逐字节不变（`untouched_byte_identical: True`）；
  - `verify` 增量 vs 全量重建：`IDENTICAL`（6 批）；
  - `replay`：6 批重放一致；
  - kill -9 前已提交 2 批，恢复后 4 批标记 `unknown` 并重跑，重放再次一致；
  - 异常演示：`E_FIELD (line=2 field=elapsed_ms)`、`E_REF (line=2 field=rank)`、
    `E_REF (line=1 field=dup_of)`、`E_TIMEOUT (port=23101)`、`E_PORT (port=23101)`。
- Tkinter 界面冒烟：5 个页签渲染（批次 6 行、回执 45 行、失效轨迹 36 行），
  连续刷新前后账本内容摘要不一致性校验为相等（刷新只读）。

## 环境备注

- 本机无 sudo 权限且系统 libtk 缺失，`run_ui.sh` 在未检测到可用 Tk 时自动注入
  本机已有的 vendored Tk/Tcl 库路径（`LD_LIBRARY_PATH`/`TCL_LIBRARY`/`TK_LIBRARY`）；
  有系统 Tk 的环境直接 `python3 -m src.ui` 即可。
- 全程仅使用 127.0.0.1 回环端口（`config/ports.json` 声明，23101/23102），
  不访问任何外部网络或账号；未使用 Docker 等容器。
