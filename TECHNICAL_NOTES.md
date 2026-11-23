# GSB 模型热更新灰度验证组件 — 技术说明

面向 Windows 本地环境的纯 Python 实现（Python 3.11+，仅依赖 numpy），
不使用 Docker/容器，不访问网络与真实账号；所有后端在 127.0.0.1 回环上
以子进程托管并动态分配端口。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| gsb/spec.py | 后端声明 / 路由规则 / NDJSON 请求流加载与强校验（E_SPEC） |
| gsb/receipts.py | h1/h2/h3（v1/v2/v3）回执线格式编解码、强校验与归一（E_FORMAT） |
| gsb/routing.py | 稳定哈希分桶（sha256 前 8 字节 mod 10000）与逐请求命中记录 |
| gsb/hotupdate.py | 参数版本注册表：热更新整体替换、版本递增、旧摘要冻结 |
| gsb/merge.py | 跨版本回执归并（参数版本降序、格式版本降序、归并序号升序）与灰度结论 |
| gsb/backend_server.py | 推理后端子进程：回环动态端口、exec/update/ping/shutdown |
| gsb/backend_host.py | 父进程托管：启动子进程、真实套接字超时、过期版本识别 |
| gsb/scheduler.py | 全局并发上限 + 每后端配额、单调时钟等待观测、超时不绕配额重试 |
| gsb/ledger.py | SQLite 账本：单事务提交、快照版本 + 内容摘要、整体替换恢复 |
| gsb/replay.py | 恢复后重放分流/隔离/归并并逐项比对（E_REPLAY） |
| gsb/pipeline.py | 主流水线：分流→调度→归一→归并→落账；崩溃恢复与重跑 |
| gsb/ui.py | Tkinter 诊断界面（惰性导入 tkinter） |

## 关键设计

### 回执线格式与归一
帧结构：头部行 `GSB-RECEIPT <vN>`，长度行 `LEN <字节数>`，随后 canonical JSON 负载。
解码校验长度自洽、必需字段齐全（v1: batch_id/route/output_digest；
v2 增 param_digest/quant_profile/elapsed_ms；v3 再增 sample_ids/param_version）
与摘要绑定：output_digest = sha256("gsb|batch|route|param_digest|sample_ids")。
旧格式缺失字段从派发上下文补齐并记录依据（dispatch:*、backend_decl:*、default:*），
归一记录携带来源格式版本与补齐字段集合。

### 分流与隔离
分桶只依赖 (rule_id, client_key) 与权重，结论不依赖字典/哈希遍历顺序。
派发时按 (批次, 后端) 冻结 (param_version, param_digest)；热更新后新请求
绑定新版本，在途请求以冻结摘要完成。同一批次同一后端出现两个参数版本
（或后端拒绝过期版本返回 STALE）即以 E_MIXED 拒绝并列出批次与冲突版本。

### 账本与快照
每批次完成时在同一 SQLite 事务写入：命中记录、归一化回执（归并输入）、
灰度结论、等待观测、活跃参数版本与进度游标。meta 表携带整数
snapshot_version / ledger_format_version 与输入规格内容摘要；重开时任一
不一致即清空全部数据表整体替换，并稳定排序列出差异项（不做增量合并）。

### 崩溃恢复
崩溃后重启只承认已完整提交（committed=1）的批次；缺失批次标记为未知且
不生成退出报告。resume_session 对已提交区间按相同种子确定性重放热更新，
恢复参数版本序号与摘要后仅重跑未知批次，最后整体重放校验。

## 验证结果（本机实际执行）

- python -m pytest tests/ -q：59 passed（约 17s）。
  覆盖：E_SPEC 全字段校验、三种格式往返与 E_FORMAT 定位、分桶确定性、
  热更新冻结、归并采纳/否决、配额与等待观测、重试不绕配额、快照整体替换、
  端到端热更新隔离、崩溃恢复重跑、篡改账本触发 E_REPLAY 精确定位。
- python run_demo.py：四阶段演示实际输出 —
  - 正常会话（batch-3 前热更新 b-gray-a）：batch-1/2 全部参数 v1，
    batch-3/4 灰度后端升至 v2，同批次同后端无混装；
  - batch-3 同时归并 v1+v2+v3 三种来源格式回执并采纳 v3；
  - 模拟崩溃后仅 2 批可恢复，batch-3/4 标记未知且 exit_report_available=False；
  - 重跑后 4 批重放校验全部通过。

## 使用

    pip install numpy
    python -m pytest tests/ -q        :: 运行测试
    python run_demo.py                :: 端到端演示（生成 gsb_ledger.db）
    python -m gsb.ui gsb_ledger.db    :: Tkinter 诊断界面（Windows 本地）
