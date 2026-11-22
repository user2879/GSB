# 实际验证结果（本机回环，确定性样本）

环境：Python 3.11、NumPy 1.26.4、标准库 sqlite3/socket/threading/logging；
tkinter 在当前 Linux 沙箱缺原生库，组件已改为惰性导入，Windows 标准安装可用。
所有端口绑定 127.0.0.1 高位端口，不访问外网，不使用容器。

## 1. 自动化测试

命令：`python3 -m unittest discover -s tests`

结果：`Ran 28 tests ... OK`，覆盖

* 定点量化、floor 截断、64 位饱和、单值/路径/判定三类误差界与 NumPy 窗口统计；
* 拓扑与事件强校验（重复标识/端口、非回环 host、故障域引用、容量非正、
  事件序号连续性、到达时刻单调性），首个违规字段与行号、E_TOPO 整体拒绝；
* 真实回环四类结局（成功/慢成功/连接拒绝/超时），拒绝来自真实
  ConnectionRefusedError、超时来自真实 socket.timeout；
* 全局与每故障域并发峰值不超限、配额耗尽真实排队；
* forced/threshold 传播链、稳定拓扑序、成功与降级互斥；
* 容量差落入判定误差界时 indistinguishable 标注与稳定决胜（与枚举顺序无关）；
* 账本单事务写入、摘要/版本不符整体替换、E_REPLAY 首个不一致定位。

## 2. 三场景演示

命令：`bash scripts/demo.sh`（Windows：`scripts\\demo.bat`），退出码 0。

### 场景 1 冷启动采集（两批次，每批 8 事件）

* 批次1结局：success=6, timeout=2；批次2结局：success=2, timeout=2,
  connection_refused=4，失效集中在第二观测窗口。
* 采集峰值：全局在途上限 3 未超限；dom-core 峰值 2、dom-data 峰值 1，
  均等于配额且未超限；存在真实排队（queued_calls>=1）。

### 场景 2 降级传播与故障域收敛（稳定顺序）

传播链（反向边）：

```
svc-db        --forced-->    svc-order      （无条件）
svc-inventory --threshold 70--> svc-bff     （可用率 50 < 70，传播）
svc-bff       --forced-->    svc-gateway    （无条件）
svc-order     --threshold 90--> svc-bff     （可用率 100 >= 90，不传播）
```

故障域收敛（每域唯一方案，服务集合按字典序）：

| 故障域 | 降级服务集合 | 总降级容量 | 选用方案 |
| --- | --- | --- | --- |
| dom-edge | svc-bff, svc-gateway | 160 | local_priority |
| dom-core | svc-inventory, svc-order | 70 | local_priority |
| dom-data | svc-db | 20 | local_priority |

跨域分摊经整数定点运算复核，例如 dom-core 容量 70、配额占比 2/5，
`floor(70*2*100/5)=2800`（即 28），与域内方案差值 4200 大于判定界 J=200，
故可区分，选容量更大的域内方案。合成测试中容量 3、配额 1/3 时差为 200，
落入 J=200，正确标注 indistinguishable 并按稳定决胜选 local_priority。

### 场景 3 快照保存与恢复重放

* 恢复重放 16 条已持久化事件，逐字段比对：`replayed=16, verified=true`。
* 篡改摘要后重开：报 `E_SNAPSHOT` 且 `replaced=true`，游标重置为 0（整体替换）。
* 篡改单条结局后重放：`E_REPLAY` 定位到首个不一致的 seq、service_id 与字段。

## 3. 三类异常业务码（一次演示内全部出现）

| 业务码 | 触发方式（仅改输入/参数） | 观测 |
| --- | --- | --- |
| E_CYCLE | 注入成环边 | 拓扑排序失败，列出环上服务 |
| E_CAPACITY | 放大容量上限 | 容量×配额×F 越过 2^63-1 |
| E_SNAPSHOT | 篡改账本 meta 摘要 | 报码后整体替换，非增量合并 |
| E_REPLAY | 篡改一条持久化结局 | 定位 seq/service_id/field |

结构化日志为单行 JSON，业务错误字段以 `biz_` 前缀写入，避免与 logging
保留键冲突；日志同时输出到 stdout 与可选 `--log-file`。

## 4. 复现命令

```bash
python3 -m unittest discover -s tests          # 28 个测试
bash scripts/demo.sh                           # 三场景 + 异常注入
python3 -m degprop --mode validate-only        # 仅强校验
python3 scripts/run_mock_service.py \
  --topology samples/topology.json --port-map artifacts/port_map.json
```
