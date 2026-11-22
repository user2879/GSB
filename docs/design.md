# 设计说明：本地推理服务身份与令牌盖章组件

## 1. 目标与边界

组件聚焦两条主线：

1. 同一推理后端的身份摘要如何随参数版本（维度、种子、量化口径、
   身份版本号）演进；
2. 由身份摘要派生的推理令牌如何被盖章、验章、持久化与重放。

所有数据在本地生成：参数矩阵与请求输入均由整数种子经纯 Python 的
splitmix64 派生，不下载模型，不访问外部服务，不使用容器。

## 2. 目录结构

```
inference_stamper/        核心库
  prng.py                 splitmix64 确定性随机源
  parameters.py           参数矩阵、逐层摘要、身份摘要
  spec.py                 JSON 声明加载与 E_SPEC 强校验
  stamping.py             HMAC 盖章、幂等、验章、E_STALE/E_IDENTITY
  ledger.py               SQLite 账本、格式版本、内容摘要、整体替换
  scheduler.py            虚拟时钟确定性调度 + 真实线程预算执行
  recovery.py             强制终止恢复（只认完整提交）
  replay.py               恢复后一致性重放（E_REPLAY）
  app.py                  演示编排
  logging_utils.py        JSONL 结构化日志
  errors.py               统一业务异常与错误码
scripts/
  generate_samples.py     生成确定性样本与演进/损坏变体
  demo.py                 端到端演示与异常注入入口
  audit.py                独立只读验章核对工具（E_AUDIT）
tests/test_all.py         全量测试
samples/                  本地声明 JSON
docs/                     技术说明
```

## 3. 确定性身份摘要

### 3.1 随机源

`BitStream` 基于 splitmix64，64 位状态步进，输出高 23 位映射为
`[-1, 1)` 的 float32。不使用 `numpy.random`，因此结果不随 NumPy
随机算法版本变化，同一声明多次运行逐位一致。

### 3.2 参数矩阵

固定层级与合成顺序：

| 层序号 | 名称 | 形状 |
| --- | --- | --- |
| 0 | W1 | (hidden_dim, model_dim) |
| 1 | b1 | (hidden_dim,) |
| 2 | W2 | (model_dim, hidden_dim) |
| 3 | b2 | (model_dim,) |

层种子 = `mix_seed(后端种子, 身份版本号, 层序号, model_dim, hidden_dim)`，
顺序敏感。

### 3.3 量化口径

| 口径 | 规则 |
| --- | --- |
| fp32 | 保持 float32，scale=1 |
| fp16 | 转 IEEE 754 float16，scale=1 |
| int8 | 逐层 absmax 对称量化到 int8，scale=absmax/127 |
| int4 | 逐层 absmax 对称量化到有符号 4bit，大端打包，scale=absmax/7 |

### 3.4 摘要

- 逐层摘要：`SHA256(帧(层名) + 帧(层序号) + 帧(口径) + 帧(形状)
  + 帧(量化字节) + 帧(scale))`
- 身份摘要：`SHA256(帧(scheme=1) + 帧(后端标识) + 帧(身份版本号)
  + 帧(维度) + 帧(种子) + 帧(口径) + 各层逐层摘要)`

维度、种子、口径、身份版本任一变化都会改变层种子或量化字节，
从而改变身份摘要。规则本身演进时提升 `IDENTITY_SCHEME_VERSION`。

## 4. 声明强校验（E_SPEC）

全部规则校验通过才返回对象，绝不部分载入；首个违规即拒绝并给出
`item_id` 与 `field`：

- 后端：标识唯一且非空、维度为正整数（拒绝 bool）、种子为整数、
  量化口径受支持、身份版本号非负、令牌口径版本为正整数；
- 请求：令牌标识唯一、后端引用存在、输入向量为非空数字数组且长度
  匹配对应 `model_dim`、请求序号全局互不重复；
- JSON 语法损坏同样以 E_SPEC 报告行列号。

## 5. 盖章与验章

### 5.1 令牌结构

令牌含 `token_id, backend_id, identity_digest, token_spec_version,
request_seq, stamp_seq, checksum, stamped_at_ms`。

- `stamp_seq` 按后端独立单调递增，账本 `MAX(stamp_seq)+1`；
- 同一 `token_id` 重复盖章直接返回账本原记录（`reused=True`），
  序号不重复递增；
- `checksum = HMAC-SHA256(本地固定密钥, 规范化帧序列)`，规范化字段
  顺序固定，密钥由固定域分隔常量与本地材料经 SHA256 派生。

### 5.2 验章判定顺序

1. 记录不存在或 HMAC 重算不符 => `E_IDENTITY`（字段 `checksum`）；
2. 令牌身份摘要 != 当前后端身份摘要 => `E_STALE`
   （字段 `identity_digest`）；
3. 令牌口径版本 != 当前后端口径版本 => `E_STALE`
   （字段 `token_spec_version`）；
4. 全部一致 => `OK`。

被拒令牌写入 `verifications` 表但 `accepted=0`，不计入有效集合，
保留原因供对照。

## 6. SQLite 账本与整体替换

表：`meta(key,value)`、`stamps`、`verifications`。

- 整数格式版本 `LEDGER_FORMAT_VERSION=1`；
- 内容摘要对全部盖章行与验章行按稳定帧格式计算（行按 token_id
  排序，单元格带类型标签）；
- 重新打开时版本或摘要不一致：
  - `on_mismatch="raise"`：抛 `E_SNAPSHOT`；
  - `on_mismatch="replace"`（默认）：旧库复制为
    `*.backup-NNN.db` 并置只读位，再整体删除重建空表，绝不增量合并；
  - `ReplacementReport` 稳定列出被替换的盖章/验章 token 与原因。
- 每条盖章在独立事务中提交（`with conn`），崩溃时不会留下半条记录。

## 7. 并行调度与预算控制

两套实现共享 `TraceEntry`（等待开始/结束、运行开始/结束、等待毫秒）：

- `plan_virtual_schedule`：虚拟单调时钟，任务时长由标识确定性派生，
  全局上限与每后端配额同时生效，准入按后端标识字典序平局判定，
  轨迹完全可复算，与线程完成顺序无关；
- `run_threaded`：单协调线程 + 有界线程池，按相同准入规则派发真实
  盖章/验章，`time.monotonic_ns()` 记录等待区间，账本连接加锁并
  允许跨线程，保证任一时刻并发数不超限。

## 8. 强制终止恢复与重放

- 恢复只依据 `stamps` 表：已提交 => committed；缺失 => unknown，
  要求重盖，不生成退出报告；
- 重放按账本身份版本与口径版本重新 `build_identity`，重算 HMAC，
  并重放验章结论；首个不一致令牌与字段抛 `E_REPLAY`，detail 中
  同时给出 stored/recomputed 依据。

## 9. 错误码

| 错误码 | 触发场景 | 定位字段 |
| --- | --- | --- |
| E_SPEC | 声明/请求集不合法、JSON 损坏 | 首个违规标识与字段 |
| E_STALE | 身份摘要或令牌口径版本演进 | identity_digest / token_spec_version |
| E_IDENTITY | 盖章校验值被篡改或记录缺失 | checksum / token_id |
| E_SNAPSHOT | 账本格式版本或内容摘要不符 | format-version / content-digest |
| E_REPLAY | 恢复重放发现首个不一致令牌 | 具体字段 |
| E_AUDIT | 独立核对工具发现账本与重算不符 | 首个不一致令牌与字段 |

## 10. Windows 运行

仅依赖 Python 3.11+、NumPy 与标准库 sqlite3，无 Tkinter、无容器：

```powershell
py -3 -m pip install -r requirements.txt
py -3 scripts\generate_samples.py
py -3 scripts\demo.py
py -3 scripts\audit.py --spec samples\spec.base.json --ledger run\main.db
py -3 -m unittest tests.test_all -v
```

路径全部使用 `pathlib`，SQLite 备份只读位通过 `os.chmod` 设置；
在 Windows 上只读位同样生效，若文件被资源管理器占用，先关闭
SQLite 连接再替换。
