# Edge Ledger Construction Report

## Summary

- Global version: `sha256:893b0f8c9757f77178e97befee107880d429c34cbb6905359ae2166b11e68128`
- Config hash: `sha256:aa422e2bc4172106b574f4765b35b531a39fc335ad53470f1f658358af2c4a64`
- Canonical digest: `sha256:51c095ba581e5cfb56692b6a3ea5f7eb0e0fb1d8624452be1150a3e66b1b5a18`
- End-to-end: 173.212 ms
- Snapshot generation: 3
- Recovery action: `resume`
- Input samples: 20
- Rendered samples: 17
- Kept samples: 13
- Dedupe removed: 3
- Clusters: 13
- Render points consumed: 17
- Physical render points recomputed: 0
- Pending samples: 0
- Quota-stopped samples: 0
- Quick view matches ledger: True

## Per-template counts

| Template | Input | Rendered | Kept | Removed | Failed | Pending | Quota stopped |
|---|---:|---:|---:|---:|---:|---:|---:|
| tpl_broken | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| tpl_note | 4 | 3 | 3 | 0 | 1 | 0 | 0 |
| tpl_summary | 11 | 10 | 6 | 3 | 2 | 0 | 0 |
| tpl_summary_answer | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tpl_task | 4 | 4 | 4 | 0 | 0 | 0 | 0 |

## Business codes

| Code | Count |
|---|---:|
| E_SAMPLE_INVALID | 3 |
| E_SCORE_NONFINITE | 1 |
| E_TEMPLATE_INVALID | 1 |

## Injection hits

| Injection | Target | Hit | Stages | Codes |
|---|---|---|---|---|
| INJ-CYCLE-001 | template:tpl_broken | True | load_templates | E_TEMPLATE_INVALID |
| INJ-MISSING-002 | sample:S-FAIL-MISSING | True | load_samples | E_SAMPLE_INVALID |
| INJ-NONFINITE-004 | sample:S-FAIL-NONFINITE | True | score | E_SCORE_NONFINITE |
| INJ-REVISION-003 | sample:S-FAIL-REV | True | load_samples | E_SAMPLE_INVALID |

## Cache stage summary

| Stage | Reuse | Recompute |
|---|---:|---:|
| dedupe | 1 | 0 |
| render | 17 | 0 |
| score | 16 | 1 |
| stats | 1 | 0 |
| template_resolution | 9 | 0 |

## Sample ledger

| Sample | Shard | Template revision | Score | Residual | Cluster | Dedupe | Render digest |
|---|---|---|---:|---:|---|---|---|
| S0001 | shard-a | tpl_summary@2 | 956000 | -400 | cluster:S0001 | kept | `sha256:a6afcacfa4982ad7678755bc62733fa84bf7d95679b0b4e79b84ab4528939941` |
| S0002 | shard-a | tpl_summary@2 | 956000 | -400 | cluster:S0001 | removed | `sha256:af935394c63f5126cece98660db49442ed3920d52cb0941e1361302661bfc85a` |
| S0003 | shard-a | tpl_summary@2 | 953000 | 200 | cluster:S0001 | removed | `sha256:013bd87b4a163524096c3ae0051b845de1cca5715d69e870325969599cc473fc` |
| S0004 | shard-a | tpl_summary@1 | 962000 | -400 | cluster:S0004 | kept | `sha256:2c70f4e598d06d90af7e3a2c74820a9eb2f36988e60096aa43ee84162bb9ffed` |
| S0005 | shard-a | tpl_summary@3 | 968000 | -400 | cluster:S0005 | kept | `sha256:e8a2776dc2e9e2790fd8af00d4a092416fc352377b442045ea3e0d952d517615` |
| S0006 | shard-b | tpl_task@2 | 968000 | -400 | cluster:S0006 | kept | `sha256:853700b144202fe540176707fe06c627bde3dac115e57cf40b2f5738f70ac51f` |
| S0007 | shard-b | tpl_task@1 | 962000 | -400 | cluster:S0007 | kept | `sha256:07b075a1a6b266c8b99bbf20da0049fcb17fbce8eb45d7ac3e2cf5e13233e95c` |
| S0008 | shard-b | tpl_task@1 | 963000 | -200 | cluster:S0008 | kept | `sha256:2f340598593dfb7b508b8e5a617852e21c2af62dac9d7d53a8dfeb58d9f986e7` |
| S0009 | shard-b | tpl_note@1 | 971000 | 200 | cluster:S0009 | kept | `sha256:b427b436c6226d70e5a8c7d45ade8b03d2cd911fd3f66871e3b22905f4804d27` |
| S0010 | shard-b | tpl_note@1 | 969000 | -200 | cluster:S0010 | kept | `sha256:7f9bd5f8023ef0c9a9789fa62019c6e384c3341ec4193049ca3f7ea87f9e016c` |
| S0011 | shard-c | tpl_summary@1 | 964000 | 0 | cluster:S0011 | kept | `sha256:d44c2cdc73712dab781408b7160ce7bb7eb4fa36970556ebd0d3aa6ecfa941a2` |
| S0012 | shard-c | tpl_summary@1 | 959000 | 200 | cluster:S0011 | removed | `sha256:2a74ab754ca8a41e5123524a75b4203f793c4cca1f521d09694e0ac73c3a8595` |
| S0013 | shard-c | tpl_summary@3 | 963000 | -200 | cluster:S0013 | kept | `sha256:e17222134af5ebe10b6607d053a4f32738b67573a1f4aff796d8e770a2a833e3` |
| S0014 | shard-c | tpl_task@2 | 960000 | 400 | cluster:S0014 | kept | `sha256:e4bf81d325b9679352dc04ec6871f131b4cf85e6be0fd7b94266a90754ff97a8` |
| S0015 | shard-c | tpl_note@1 | 954000 | 400 | cluster:S0015 | kept | `sha256:8296ca482877af9ca425f83dba87b2eab3e4f5a9dd0732063fa46f36484c8177` |
| S0016 | shard-c | tpl_summary@2 | 964000 | 0 | cluster:S0016 | kept | `sha256:9ceafd0effb40e0eaafeef668ab17de587540ebf62168308490e7b894151e382` |
| S-FAIL-NONFINITE | shard-failure | tpl_summary@2 |  |  |  | failed | `sha256:6240bf783133f5de232151dec478c337de1f347bbff38fe5ccc053bd1f3da356` |

## Pending queue

| Sample | Shard | Code | Sort key |
|---|---|---|---|

## Failures

| Stage | Code | Target | Injection | Message |
|---|---|---|---|---|
| load_templates | E_TEMPLATE_INVALID | template:tpl_broken | INJ-CYCLE-001 | template revision parent chain contains a cycle |
| load_samples | E_SAMPLE_INVALID | sample:S-FAIL-CYCLE |  | instruction template revision does not exist or is invalid: tpl_broken@2 |
| load_samples | E_SAMPLE_INVALID | sample:S-FAIL-MISSING | INJ-MISSING-002 | slot fills do not match the effective slot set |
| load_samples | E_SAMPLE_INVALID | sample:S-FAIL-REV | INJ-REVISION-003 | instruction template revision does not exist or is invalid: tpl_summary@99 |
| score | E_SCORE_NONFINITE | sample:S-FAIL-NONFINITE | INJ-NONFINITE-004 | quality computation produced a non-finite value |

