# Validation Record

执行日期：2026-11-19  
工作区：`/home/zsl/feedback/uidfc5909d3-default-212`  
说明：本次自动验收在任务提供的 Linux/WSL 工作区执行；代码仅使用跨平台路径与 CLI，Windows 运行命令见 README。未访问远程服务、游戏引擎或外部素材。

## Environment

| Item | Actual value |
| --- | --- |
| Python | 3.11.0rc1 |
| Platform | Linux |
| numpy | 1.26.4 |
| PyYAML | 5.4.1 |
| Thread controls | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` |

## Automated Tests

```text
PYTHONHASHSEED=0 python -m unittest discover -s tests -q
----------------------------------------------------------------------
Ran 27 tests in 0.842s
OK
```

Coverage includes:

- `E_SPEC_FIELD`, `E_SPEC_WEIGHT`, `E_SPEC_BOUNDS` and no partial output.
- Same input/seed canonical level bytes and grid hash.
- Purpose-specific terrain sub-seed offset reproducibility.
- Connectivity paths and `E_CONNECT_UNSAT` conflict evidence.
- Forbidden-adjacency local repair and skipped-repair candidate preservation.
- Resource local repair with seeded removal plus skip-repair violation.
- Independent verifier ignoring stale generator evidence.
- Row-major first violation, artifact-tamper `E_VERIFIER_MISMATCH`.
- Minimal forbidden pair, resource over-count and connectivity cut counterexamples.
- Counterexample replay.
- CLI generate/verify and two full demo trees byte-for-byte identical.

## Demo Runs

Run 1 used `PYTHONHASHSEED=1`; run 2 used `PYTHONHASHSEED=987`.

```text
python -m level_workbench demo --out artifacts/demo-run-1
python -m level_workbench demo --out artifacts/demo-run-2
python -m level_workbench compare --a artifacts/demo-run-1 --b artifacts/demo-run-2
```

Comparison:

```json
{"differences":[],"file_count_a":32,"file_count_b":32,"identical":true,"schema":"level-workbench/comparison-v1","tree_sha256_a":"0192e9f65f11bd9159dacce4a138f829e76a7b9e5b7b8864b1fa582bcac837ae","tree_sha256_b":"0192e9f65f11bd9159dacce4a138f829e76a7b9e5b7b8864b1fa582bcac837ae"}
```

Report and run hashes:

| Artifact | SHA-256 |
| --- | --- |
| `report.json` | `3c7b938b3a1ff9c85e9449a09d57eb047f2e31b781d0a17a3059f67478fa4dda` |
| In-process case reports | `a65e0d4c7cb436b6c3095198e99c292dcc7c29631cde94798a78c02f013b0200` |
| Manifest `run_sha256` | `488ddedd2e8c00706a9380ef895cec758f39015f7b7901c36040769278eedeee` |

## Case Results

| Case | Status | Code | First violation | Minimal counterexample |
| --- | --- | --- | --- | --- |
| `normal` | pass | `OK` | none | not needed |
| `connectivity_unsat` | fail | `E_CONNECT_UNSAT` | `[13,5]` target, conflict `[spawn,target]` | not applicable (unsat before candidate) |
| `resource_spec_overflow` | error | `E_SPEC_BOUNDS` | path `resources.count`, max `320 > 280` | no grid generated |
| `resource_pipeline_overflow` | fail | `E_RESOURCE_RANGE` | `[2,0]`; count `20` outside `[10,14]` | 15 resources; every single removal returns count 14 |
| `illegal_adjacent` | fail | `E_REPAIR_EXHAUST` | `[8,6]`, neighbor `[9,6]`, pair `[rock,water]` | two tiles; every single removal removes violation |
| `verifier_mismatch` | fail | `E_VERIFIER_MISMATCH` | `[8,6]`, neighbor `[9,6]` | two tiles; generator `pass`, verifier `fail` for `forbidden_adjacent` |

### Normal Case Hashes

| Hash | Value |
| --- | --- |
| Normalized spec | `2e2cd03610b2eacb99a38c863fba540183c7283438183763982b396ef6aad533` |
| Effective spec | `25f8d96aa10af45520706bc3e1d57b2a426e98c1de418eafdd693d043856f976` |
| Grid | `877615790334d1cb4f9427cdb32f79c28d83dc96db70940a7c7259cf3ffc9aad` |
| Level | `96a0650c7a604c0fbaa79ac48bfe8206f56d4ede14e57bafe57635455f6ca8a9` |
| Generator evidence | `72721b6841a86f6c62e0d34bb397cf1fdb73c316b2713fba574ad1eea94e34c6` |
| Verifier result | `9d35b615fc0bd22cc7bc2b31ff42fa1739e30ca96f181b2379c2a31b932c90ba` |

Normal generation fixed stage order:

`terrain -> endpoint_normalization -> spawn_neighborhood -> connectivity_carving -> forbidden_adjacent_repair -> resource_placement`.

The normal sample started with 14 forbidden adjacent edges; the deterministic local repair completed in 1 round and the independent verifier passed all six constraint groups.

## Connectivity Unsatisfiable Evidence

The immutable water wall at `x=8` for `y=0..9` separates spawn `[2,5]` from target `[13,5]`.

```json
{
  "conflict_pair": ["spawn", "target"],
  "coord": [13, 5],
  "goal_kind": "target",
  "reason": "no path exists through mutable cells; immutable blocked cells separate spawn from the goal",
  "blocking_cell_count": 10
}
```

The generator returned `E_CONNECT_UNSAT` and did not write a partial level/grid.

## Counterexample Replay

Forbidden adjacency:

```text
PYTHONHASHSEED=0 python -m level_workbench replay-counterexample --spec artifacts/demo-run-1/specs/normal.yaml --counterexample artifacts/demo-run-1/counterexamples/illegal_adjacent.counterexample.json
```

Actual replay result: `violation_code=E_FORBIDDEN_ADJACENT`, `kind=standalone_tile_patch`, `reproduced=true`, `minimal=true`; removing `[8,6]` or `[9,6]` both makes the violation disappear.

Resource over-count:

```text
reproduced=true, minimal=true, kind=resource_count, code=E_RESOURCE_RANGE
```

## Other CLI Checks

```text
python -m level_workbench validate-spec --spec artifacts/demo-run-1/specs/normal.yaml
pass 2e2cd03610b2eacb99a38c863fba540183c7283438183763982b396ef6aad533

python -m level_workbench generate --spec artifacts/demo-run-1/specs/normal.yaml --out artifacts/cli-check
status=pass code=OK

python -m level_workbench verify --spec artifacts/cli-check/effective_spec.yaml --level artifacts/cli-check/level.json
pass 6 None
```

## Reproducibility Notes

- Different `PYTHONHASHSEED` values produced byte-identical 32-file output trees.
- The demo performs each fixed sequence twice in-process and records equality; it was also compared as two separate process outputs.
- The grid hash is computed over canonical row-major grid text; level content hashes exclude their own hash field.
- The resource underflow counterexample is explicitly marked non-minimizable by element removal, because deleting resources cannot repair an underflow.
