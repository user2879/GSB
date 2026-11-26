# IntentLab Diagnostic Report `gen-20261126T222058Z-298034-acf6e01f3b`

## Summary

- Queries loaded/processed/pending: `24` / `24` / `0`
- Merged segments: `14`
- Boundary intents: `1`
- Cache reuse hits: `62`
- Business codes: `{}`
- Snapshot generation: `gen-20261126T222058Z-298034-acf6e01f3b`

## Invalidation by revision

| revision | match | score | merge | total |
|---|---:|---:|---:|---:|
| `r00` | 0 | 0 | 0 | 0 |
| `r01` | 0 | 0 | 0 | 0 |
| `r02` | 0 | 0 | 0 | 0 |
| `r03` | 0 | 0 | 0 | 0 |
| `r04` | 0 | 0 | 0 | 0 |
| `r06` | 0 | 0 | 0 | 0 |
| `r07` | 0 | 0 | 0 | 0 |

## Query conclusions

| query | source | intent | score | margin | boundary | high-width | first error stage | segment |
|---|---|---|---:|---:|---|---|---|---|
| `q001` | source-a | `weather` | 18022400 | 8847360 | False | `weather` | None | `seg-190cae7bd3c86814` |
| `q002` | source-b | `weather` | 21845344 | 12670304 | False | `weather` | None | `seg-190cae7bd3c86814` |
| `q003` | source-c | `weather` | 18022400 | 8847360 | False | `weather` | None | `seg-190cae7bd3c86814` |
| `q004` | source-a | `nearby` | 15728640 | 5898240 | False | `nearby` | None | `seg-93dae0078f61780a` |
| `q005` | source-b | `nearby` | 13107200 | 3276800 | False | `nearby` | None | `seg-93dae0078f61780a` |
| `q006` | source-c | `music` | 24029856 | 14527136 | False | `music` | None | `seg-62b9789f329ae7d1` |
| `q007` | source-a | `navigation` | 12014944 | 2184544 | False | `navigation` | None | `seg-2463b4fb5fe0d8b9` |
| `q008` | source-b | `navigation` | 14417920 | 4587520 | False | `navigation` | None | `seg-2463b4fb5fe0d8b9` |
| `q009` | source-c | `hotel` | 18022400 | 8519680 | False | `hotel` | None | `seg-51770a697e80a868` |
| `q010` | source-a | `music` | 12014944 | 2512224 | False | `music` | None | `seg-2601ad1a409542ac` |
| `q011` | source-b | `hotel` | 14417920 | 4915200 | False | `hotel` | None | `seg-38dfe8e0684039b6` |
| `q012` | source-c | `hotel` | 18022400 | 8519680 | False | `hotel` | None | `seg-38dfe8e0684039b6` |
| `q013` | source-a | `news` | 19660800 | 10158080 | False | `news` | None | `seg-5a1293b1b30a369e` |
| `q014` | source-b | `news` | 15728640 | 6225920 | False | `news` | None | `seg-5a1293b1b30a369e` |
| `q015` | source-c | `translate` | 13107200 | 3932160 | False | `translate` | None | `seg-913d997f48e6c3bc` |
| `q016` | source-a | `nearby` | 13107200 | 3276800 | False | `nearby` | None | `seg-eb42f57d27f383f4` |
| `q017` | source-b | `nearby` | 21845344 | 12014944 | False | `nearby` | None | `seg-eb42f57d27f383f4` |
| `q018` | source-c | `nearby` | 9830400 | 0 | True | `nearby` | None | `seg-eb42f57d27f383f4` |
| `q019` | source-a | `music` | 18022400 | 8519680 | False | `music` | None | `seg-90df7abd482d9310` |
| `q020` | source-b | `music` | 24029856 | 14527136 | False | `music` | None | `seg-90df7abd482d9310` |
| `q021` | source-c | `weather` | 18022400 | 8847360 | False | `weather` | None | `seg-619d5d958dc02ab9` |
| `q022` | source-a | `translate` | 21845344 | 12670304 | False | `translate` | None | `seg-f41aeeb1afd3d8f3` |
| `q023` | source-b | `translate` | 13107200 | 3932160 | False | `translate` | None | `seg-f41aeeb1afd3d8f3` |
| `q024` | source-c | `navigation` | 12014944 | 2184544 | False | `navigation` | None | `seg-d71b734bd7d5d9e0` |

## Segments

- `seg-190cae7bd3c86814` session `s01` intent `weather` queries `q001, q002, q003` slots `6` conflicts `0`
- `seg-93dae0078f61780a` session `s02` intent `nearby` queries `q004, q005` slots `2` conflicts `0`
- `seg-62b9789f329ae7d1` session `s02` intent `music` queries `q006` slots `2` conflicts `0`
- `seg-2463b4fb5fe0d8b9` session `s03` intent `navigation` queries `q007, q008` slots `4` conflicts `0`
- `seg-51770a697e80a868` session `s03` intent `hotel` queries `q009` slots `2` conflicts `0`
- `seg-2601ad1a409542ac` session `s04` intent `music` queries `q010` slots `2` conflicts `0`
- `seg-38dfe8e0684039b6` session `s04` intent `hotel` queries `q011, q012` slots `4` conflicts `0`
- `seg-5a1293b1b30a369e` session `s05` intent `news` queries `q013, q014` slots `2` conflicts `0`
- `seg-913d997f48e6c3bc` session `s05` intent `translate` queries `q015` slots `2` conflicts `0`
- `seg-eb42f57d27f383f4` session `s06` intent `nearby` queries `q016, q017, q018` slots `4` conflicts `1`
- `seg-90df7abd482d9310` session `s07` intent `music` queries `q019, q020` slots `4` conflicts `0`
- `seg-619d5d958dc02ab9` session `s07` intent `weather` queries `q021` slots `2` conflicts `0`
- `seg-f41aeeb1afd3d8f3` session `s08` intent `translate` queries `q022, q023` slots `4` conflicts `0`
- `seg-d71b734bd7d5d9e0` session `s08` intent `navigation` queries `q024` slots `2` conflicts `0`

## Pending queue

- _empty_

## Business issues

- _none_
