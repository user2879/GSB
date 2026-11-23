# 本地网关 JSONL 协议

## 启动

```bat
py -3 -m gray_merge_lab serve --gateway-port 28190
```

网关只绑定 `127.0.0.1:28190`，并按声明启动四个后端子进程，分别绑定 `127.0.0.1:28101..28104`。所有端口可先由 `run_doctor.bat` 检查。

## 样本请求

TCP 连接上每行一个 JSON 对象：

```json
{"sample_id":"ss-A-01","sample_seq":1,"client_key":"tenant-search-A","rule_id":"search","request_path":"/search/a/01","input_seed":11001}
```

服务端逐行返回：

```json
{"ok":true,"op":"accepted","sample_id":"ss-A-01","pending_samples":1}
```

## 刷新批次

发送：

```json
{"op":"flush"}
```

服务端处理当前接收纪元所有待处理样本，返回：

```json
{
  "ok": true,
  "op": "flush",
  "samples": 8,
  "batches": 2,
  "batch_rows": [
    {
      "batch_id": "b-search-canary-000001-000004-37c878b825",
      "rule_id": "search",
      "destination": "canary",
      "client_bucket": 1465,
      "client_key": "tenant-search-A",
      "sample_ids": ["ss-A-01", "ss-A-02", "ss-A-03", "ss-A-04"],
      "output_digest": "sha256:...",
      "backend_param_digest": "sha256:...",
      "backend_param_version": 1,
      "merge_seq": 2,
      "duration_ms": 1
    }
  ],
  "sample_to_batch": {
    "ss-A-01": {"batch_id": "b-search-canary-000001-000004-37c878b825", "destination": "canary"}
  }
}
```

错误行：

```json
{"ok":false,"error":"E_SPEC","message":"...","field":"samples[1].sample_seq","batch_id":null,"details":{}}
```

## 后端子进程协议

网关到后端同样使用 JSONL：

```json
{"op":"infer","batch_id":"...","param_version":1,"samples":[{"sample_id":"...","input_seed":11001}]}
{"op":"update","version":2,"spec":{"backend_id":"canary-search","host":"127.0.0.1","port":28102,"model_dim":12,"hidden_dim":28,"seed":7727,"quant_profile":"int8-sym-v1"}}
{"op":"ping"}
```

后端只接受 `127.0.0.1` 回环字面量为声明 host，并校验更新不能改变后端标识或端口。网关校验后端返回的参数摘要/版本与自身 ModelBank 一致，避免热更新撕裂。

