from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any

from .canonical import sha256_hex
from .errors import E_INJECTED_FAULT, E_OP_QUOTA_EXCEEDED, E_POINT_BUDGET_PENDING, InjectedFault
from .fixedpoint import FixedContext, finite_number, fraction_to_float_text
from .kernel import embedding_lookup, layer_normalize, multi_head_attention, segment_pool
from .ledger import CacheStore, classify_payload_change, operator_fingerprint, token_fingerprint
from .scheduler import BudgetScheduler, Task


STAGE_ORDER = {"embedding": 0, "segment_pool": 1, "attention": 2, "layer_norm": 3, "final_result": 4}


class ChainEngine:
    def __init__(self, config: dict[str, Any], operators: list[dict[str, Any]],
                 tokens: list[dict[str, Any]], boundaries: list[int], *,
                 injections: list[dict[str, Any]] | None = None,
                 exchange_version: str | None = None,
                 store: CacheStore | None = None,
                 serial: bool = False):
        self.config = config
        self.operators = operators
        self.tokens = tokens
        self.boundaries = boundaries
        self.injections = injections or []
        self.exchange_version = (exchange_version or config.get("exchange_version", "e2")).lower()
        self.serial = serial
        self.ctx = FixedContext(config["frac_bits"], config["quant_tier"], config.get("tie_rule", "half_away_from_zero"))
        self.store = store or CacheStore()
        self.old_index = copy.deepcopy(self.store.node_index)

        self.by_seq = {op["seq"]: op for op in operators}
        self.topo_ops = self._topological_order(operators)
        self.segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        for index, token in enumerate(tokens):
            token["seq"] = index
            token["segment_index"] = self._segment_index(index)

        for op in self.operators:
            fraction = self._scale_fraction(op)
            op["scale_rational"] = f"{fraction.numerator}/{fraction.denominator}"
            op["_scale_num"] = fraction.numerator
            op["_scale_den"] = fraction.denominator
            op["fingerprint"] = operator_fingerprint(op)

        self.candidates: dict[str, dict[str, Any]] = {}
        self.ordered_nodes: list[dict[str, Any]] = []
        self.ledger_entries: list[dict[str, Any]] = []
        self.records_by_node: dict[str, dict[str, Any]] = {}
        self.node_status: dict[str, str] = {}
        self.node_faults: dict[str, dict[str, Any]] = {}
        self.fault_hits: dict[str, dict[str, Any]] = {}
        self.new_index: dict[str, dict[str, Any]] = {}
        self.scheduler_report = None

    @staticmethod
    def _topological_order(operators: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_seq = {op["seq"]: op for op in operators}
        visited: set[int] = set()
        result: list[dict[str, Any]] = []

        def visit(seq: int, stack: list[int]) -> None:
            if seq in visited:
                return
            if seq in stack:
                raise ValueError(f"cycle at {seq}")
            for ref in by_seq[seq]["inputs"]:
                visit(ref, stack + [seq])
            visited.add(seq)
            result.append(by_seq[seq])

        for seq in sorted(by_seq):
            visit(seq, [])
        return result

    @staticmethod
    def _scale_fraction(op: dict[str, Any]) -> Fraction:
        if op.get("scale_float") is not None:
            return Fraction(str(op["scale_float"]))
        return Fraction(int(op.get("scale_permille", 1000)), 1000)

    @staticmethod
    def node_id(kind: str, op_seq: int, uid: str) -> str:
        return f"{op_seq}:{kind}:{uid}"

    @staticmethod
    def stage_label(kind: str, op_seq: int) -> str:
        return f"{op_seq}:{kind}"

    def _segment_index(self, token_index: int) -> int:
        for index, (start, end) in enumerate(self.segments):
            if start <= token_index < end:
                return index
        raise IndexError(f"token {token_index} is outside boundaries {self.boundaries}")

    def _segment_members(self, token_index: int) -> list[int]:
        segment_index = self._segment_index(token_index)
        start, end = self.segments[segment_index]
        return list(range(start, end))

    def _embedding_source_seq(self, op: dict[str, Any]) -> int:
        if op["kind"] == "embedding":
            return op["seq"]
        stack = list(op.get("inputs", []))
        seen: set[int] = set()
        while stack:
            ref = stack.pop(0)
            if ref in seen or ref not in self.by_seq:
                continue
            seen.add(ref)
            source = self.by_seq[ref]
            if source["kind"] == "embedding":
                return ref
            stack.extend(source.get("inputs", []))
        raise ValueError(f"{op['seq']}:{op['kind']} needs an embedding input")

    def _candidate_key(self, kind: str, op_seq: int, uid: str) -> str:
        return self.candidates[self.node_id(kind, op_seq, uid)]["key"]

    def _segment_payload(self, op: dict[str, Any], token_index: int) -> dict[str, Any]:
        segment_index = self._segment_index(token_index)
        start, end = self.segments[segment_index]
        members = list(range(start, end))
        embedding_seq = self._embedding_source_seq(op)
        embedding_op = self.by_seq[embedding_seq]
        return {
            "segment_index": segment_index,
            "start": start,
            "end": end,
            "member_uids": [self.tokens[index]["uid"] for index in members],
            "embedding_keys": [
                self._candidate_key(embedding_op["kind"], embedding_seq, self.tokens[index]["uid"])
                for index in members
            ],
        }

    def _upstream_payload(self, op: dict[str, Any], token_index: int) -> Any:
        uid = self.tokens[token_index]["uid"]
        kind = op["kind"]
        if kind == "embedding":
            return {}
        input_keys = {
            str(ref): self._candidate_key(self.by_seq[ref]["kind"], ref, uid)
            for ref in op["inputs"]
        }
        if kind == "segment_pool":
            return {"input_keys": input_keys, "segment": self._segment_payload(op, token_index)}
        if kind == "attention":
            return {"input_keys": input_keys, "segment": self._segment_payload(op, token_index)}
        if kind == "layer_norm":
            return {"input_keys": input_keys}
        return {"input_keys": input_keys}

    def plan(self) -> None:
        self.candidates.clear()
        self.ordered_nodes.clear()
        for op in sorted(self.operators, key=lambda item: item["seq"]):
            for token_index, token in enumerate(self.tokens):
                uid = token["uid"]
                node_id = self.node_id(op["kind"], op["seq"], uid)
                payload = {
                    "schema": "chaindiag-cache-v1",
                    "stage": op["kind"],
                    "identity": {
                        "node_id": node_id,
                        "op_seq": op["seq"],
                        "token_seq": token_index,
                        "token_uid": uid,
                        "token_fingerprint": token_fingerprint(uid, token["token"]),
                    },
                    "upstream": self._upstream_payload(op, token_index),
                    "operator_fingerprint": op["fingerprint"],
                    "fixed_point": {"frac_bits": self.config["frac_bits"], "quant_tier": self.config["quant_tier"]},
                    "exchange_version": self.exchange_version,
                }
                key = sha256_hex(payload)
                node = {
                    "node_id": node_id,
                    "key": key,
                    "payload": payload,
                    "kind": op["kind"],
                    "op_seq": op["seq"],
                    "token_seq": token_index,
                    "token_uid": uid,
                    "op": op,
                }
                self.candidates[node_id] = node
                self.ordered_nodes.append(node)

        norm_ops = [op for op in self.topo_ops if op["kind"] == "layer_norm"]
        terminal_op = norm_ops[-1] if norm_ops else self.topo_ops[-1]
        for token_index, token in enumerate(self.tokens):
            uid = token["uid"]
            upstream_key = self._candidate_key(terminal_op["kind"], terminal_op["seq"], uid)
            node_id = self.node_id("final_result", terminal_op["seq"], uid)
            payload = {
                "schema": "chaindiag-cache-v1",
                "stage": "final_result",
                "identity": {
                    "node_id": node_id,
                    "op_seq": terminal_op["seq"],
                    "token_seq": token_index,
                    "token_uid": uid,
                    "token_fingerprint": token_fingerprint(uid, token["token"]),
                },
                "upstream": {"layer_norm_key": upstream_key},
                "operator_fingerprint": sha256_hex(
                    {"stage": "final_result", "tie_rule": self.config.get("tie_rule"), "schema": "final-v1"}
                ),
                "fixed_point": {"frac_bits": self.config["frac_bits"], "quant_tier": self.config["quant_tier"]},
                "exchange_version": self.exchange_version,
            }
            key = sha256_hex(payload)
            node = {
                "node_id": node_id,
                "key": key,
                "payload": payload,
                "kind": "final_result",
                "op_seq": terminal_op["seq"],
                "token_seq": token_index,
                "token_uid": uid,
                "op": terminal_op,
            }
            self.candidates[node_id] = node
            self.ordered_nodes.append(node)

        self.ledger_entries = []
        for ordinal, node in enumerate(self.ordered_nodes):
            old = self.old_index.get(node["node_id"])
            old_payload = old.get("payload") if old else None
            reasons = classify_payload_change(old_payload, node["payload"])
            armed_injection = self._injection_for(node["op_seq"], node["kind"], node["token_seq"], node["token_uid"])
            if node["key"] in self.store.records and armed_injection is None:
                action = "reused"
                self.node_status[node["node_id"]] = "completed"
                self.records_by_node[node["node_id"]] = self.store.records[node["key"]]
                self.new_index[node["node_id"]] = {"cache_key": node["key"], "payload": node["payload"]}
                if old and old.get("cache_key") == node["key"]:
                    reasons = ["cache-key-match"]
                else:
                    reasons = reasons + ["content-addressed-cache-hit"]
            else:
                action = "computed"
                self.node_status[node["node_id"]] = "waiting"
                if not old:
                    reasons = ["initial_run"]
                if armed_injection is not None:
                    reasons = ["fault-injection-armed"] + reasons
            entry = {
                "ordinal": ordinal,
                "action": action,
                "kind": node["kind"],
                "op_seq": node["op_seq"],
                "token_seq": node["token_seq"],
                "token_uid": node["token_uid"],
                "cache_key": node["key"],
                "reasons": reasons,
            }
            self.ledger_entries.append(entry)

    def _injection_for(self, op_seq: int, kind: str, token_index: int, uid: str) -> dict[str, Any] | None:
        for injection in self.injections:
            target = injection.get("target", {}) if isinstance(injection.get("target"), dict) else {}
            inj_op = injection.get("op_seq", target.get("op_seq"))
            inj_uid = injection.get("token_uid", target.get("token_uid", target.get("uid")))
            inj_seq = injection.get("token_seq", target.get("token_seq"))
            inj_stage = injection.get("stage", target.get("stage"))
            if inj_op is not None and int(inj_op) != op_seq:
                continue
            if inj_stage is not None and inj_stage not in (kind, STAGE_ORDER and kind):
                # Accept either kernel kind or numeric/string stage labels.
                if str(inj_stage) != kind and str(inj_stage) != self.stage_label(kind, op_seq):
                    continue
            if inj_uid is not None and inj_uid != uid:
                continue
            if inj_seq is not None and int(inj_seq) != token_index:
                continue
            return injection
        return None

    def _maybe_inject(self, injection: dict[str, Any] | None, op_seq: int, kind: str,
                      token_index: int, uid: str) -> None:
        if not injection:
            return
        injection_id = injection.get("id", injection.get("injection_id", "INJ-UNKNOWN"))
        fault_type = injection.get("type", "injected_fault")
        expected_dim = int(self.config["embedding_dim"])
        if fault_type == "dimension_mismatch":
            actual_dim = expected_dim + int(injection.get("delta", 1))
            raise InjectedFault(
                injection_id,
                fault_type,
                f"注入维度不符: expected {expected_dim}, got {actual_dim}",
                op_seq=op_seq,
                stage=kind,
                token_uid=uid,
                token_seq=token_index,
                expected_dim=expected_dim,
                actual_dim=actual_dim,
            )
        if fault_type == "non_finite_feature":
            features = [0, 1, float("nan")]
            finite = all(finite_number(value) for value in features)
            raise InjectedFault(
                injection_id,
                fault_type,
                "注入特征包含非有限值 NaN/Inf，校验层已隔离该令牌",
                op_seq=op_seq,
                stage=kind,
                token_uid=uid,
                token_seq=token_index,
                finite=finite,
                observed_feature="float('nan')",
            )
        if fault_type == "segment_out_of_bounds":
            requested = len(self.segments) + int(injection.get("delta", 1))
            raise InjectedFault(
                injection_id,
                fault_type,
                f"注入段边界越界: requested segment {requested}, valid [0,{len(self.segments) - 1}]",
                op_seq=op_seq,
                stage=kind,
                token_uid=uid,
                token_seq=token_index,
                requested_segment=requested,
                segment_count=len(self.segments),
                boundaries=self.boundaries,
            )
        raise InjectedFault(injection_id, fault_type, injection.get("message", "unknown injected fault"), token_uid=uid)

    def _record_for_node(self, node: dict[str, Any], kernel_result: dict[str, Any]) -> dict[str, Any]:
        details = dict(kernel_result.get("details", {}))
        details["rounding_steps"] = kernel_result.get("rounding_steps", [])
        if "ties" in kernel_result:
            details["ties"] = kernel_result["ties"]
        record = {
            "cache_key": node["key"],
            "kind": node["kind"],
            "op_seq": node["op_seq"],
            "token_seq": node["token_seq"],
            "token_uid": node["token_uid"],
            "vector": kernel_result["vector"],
            "residual": int(kernel_result.get("residual", 0)),
            "boundary_token": bool(kernel_result.get("boundary_token", False)),
            "details": details,
        }
        return record

    def execute_node(self, node: dict[str, Any]) -> dict[str, Any]:
        op = node["op"]
        i = node["token_seq"]
        uid = node["token_uid"]
        injection = self._injection_for(op["seq"], node["kind"], i, uid)
        record: dict[str, Any] | None = None
        try:
            kind = node["kind"]
            if kind == "embedding":
                result = embedding_lookup(
                    token_id=self.tokens[i]["token"],
                    dim=int(self.config["embedding_dim"]),
                    seed=int(self.config["seed"]),
                    ctx=self.ctx,
                    scale_num=op["_scale_num"],
                    scale_den=op["_scale_den"],
                    int_bits=op["int_bits"],
                )
            elif kind == "segment_pool":
                embedding_seq = self._embedding_source_seq(op)
                members = self._segment_members(i)
                member_records = [
                    self.records_by_node[self.node_id(self.by_seq[embedding_seq]["kind"], embedding_seq, self.tokens[m]["uid"])]
                    for m in members
                ]
                result = segment_pool(
                    member_records=member_records,
                    ctx=self.ctx,
                    scale_num=op["_scale_num"],
                    scale_den=op["_scale_den"],
                    int_bits=op["int_bits"],
                )
            elif kind == "attention":
                embedding_seq = self._embedding_source_seq(op)
                members = self._segment_members(i)
                emb_op = self.by_seq[embedding_seq]
                embedding_records = [
                    self.records_by_node[self.node_id(emb_op["kind"], embedding_seq, self.tokens[m]["uid"])]
                    for m in members
                ]
                pool_ref = next((ref for ref in op["inputs"] if self.by_seq[ref]["kind"] == "segment_pool"), None)
                pool_record = None
                if pool_ref is not None:
                    pool_record = self.records_by_node[self.node_id("segment_pool", pool_ref, uid)]
                snapshot = op.get("attention_snapshot") or op.get("params", {}).get("attention_snapshot") or {}
                heads = int(snapshot.get("heads", self.config["heads"]))
                head_dim = int(snapshot.get("dim", snapshot.get("head_dim", self.config["head_dim"])))
                temperature = int(
                    snapshot.get("temperature_permille", op.get("params", {}).get("temperature_permille", 1000))
                )
                result = multi_head_attention(
                    query_token=self.tokens[i]["token"],
                    query_uid=uid,
                    member_tokens=[self.tokens[m]["token"] for m in members],
                    member_uids=[self.tokens[m]["uid"] for m in members],
                    embedding_records=embedding_records,
                    pool_record=pool_record,
                    heads=heads,
                    head_dim=head_dim,
                    seed=int(self.config["seed"]),
                    temperature_permille=temperature,
                    ctx=self.ctx,
                    scale_num=op["_scale_num"],
                    scale_den=op["_scale_den"],
                    int_bits=op["int_bits"],
                )
            elif kind == "layer_norm":
                ref = op["inputs"][0]
                upstream = self.records_by_node[self.node_id(self.by_seq[ref]["kind"], ref, uid)]
                params = op.get("params", {})
                result = layer_normalize(
                    values=upstream["vector"],
                    ctx=self.ctx,
                    gamma_int=params.get("gamma_int"),
                    beta_int=int(params.get("bias_int", params.get("beta_int", 0))),
                    int_bits=op["int_bits"],
                    op_scale_num=op["_scale_num"],
                    op_scale_den=op["_scale_den"],
                )
            elif kind == "final_result":
                norm_key = node["payload"]["upstream"]["layer_norm_key"]
                upstream = self.store.records.get(norm_key) or self.records_by_node.get(
                    next(nid for nid, candidate in self.candidates.items() if candidate["key"] == norm_key)
                )
                quantized, steps, residual, ties = self.ctx.quantize(
                    list(upstream["vector"]), int_bits=int(op["int_bits"]), phase="final_result.quantize"
                )
                result = {
                    "vector": quantized,
                    "residual": residual,
                    "rounding_steps": steps,
                    "ties": ties,
                    "boundary_token": upstream.get("boundary_token", False),
                    "details": {"source_cache_key": norm_key, "finalized": True},
                }
            else:
                raise ValueError(f"unsupported kernel stage: {kind}")
            record = self._record_for_node(node, result)
            # Run the injected validator after the deterministic shadow value
            # exists.  The record remains quarantined for dependencies, which
            # prevents an injection from perturbing unrelated tokens.
            self._maybe_inject(injection, op["seq"], node["kind"], i, uid)
            return {"ok": True, "record": record}
        except InjectedFault as fault:
            return {"ok": False, "fault": {
                "injection_id": fault.injection_id,
                "fault_type": fault.fault_type,
                "code": E_INJECTED_FAULT,
                "message": str(fault),
                "context": fault.context,
            }, "record": record}

    def _record_fault(self, node: dict[str, Any], fault: dict[str, Any]) -> None:
        injection_id = fault["injection_id"]
        injection = self._injection_for(node["op_seq"], node["kind"], node["token_seq"], node["token_uid"]) or {"id": injection_id}
        self.fault_hits.setdefault(injection_id, {"injection": injection, "hits": []})
        self.fault_hits[injection_id]["hits"].append(
            {
                "node_id": node["node_id"],
                "op_seq": node["op_seq"],
                "kind": node["kind"],
                "token_seq": node["token_seq"],
                "token_uid": node["token_uid"],
            }
        )

    def _dependency_nodes_ready(self, node: dict[str, Any]) -> bool:
        payload = node["payload"].get("upstream", {})
        if node["kind"] == "final_result":
            norm_key = payload["layer_norm_key"]
            return any(candidate["key"] == norm_key and self.node_status.get(nid) == "completed"
                       for nid, candidate in self.candidates.items())
        op = node["op"]
        uid = node["token_uid"]
        token_index = node["token_seq"]
        for ref in op.get("inputs", []):
            ref_kind = self.by_seq[ref]["kind"]
            if self.node_status.get(self.node_id(ref_kind, ref, uid)) not in ("completed", "quarantined"):
                return False
        if op["kind"] in ("segment_pool", "attention"):
            embedding_seq = self._embedding_source_seq(op)
            emb_kind = self.by_seq[embedding_seq]["kind"]
            for member in self._segment_members(token_index):
                member_uid = self.tokens[member]["uid"]
                if self.node_status.get(self.node_id(emb_kind, embedding_seq, member_uid)) not in ("completed", "quarantined"):
                    return False
        return True

    def run(self, *, with_width_reference: bool = True) -> dict[str, Any]:
        self.plan()
        scheduler = BudgetScheduler(
            point_budget=int(self.config["point_budget"]),
            operator_quota=int(self.config["operator_quota"]),
            max_workers=1 if self.serial else int(self.config["max_workers"]),
        )
        self.scheduler_report = scheduler.run([])[1]

        depths = self._op_depths()
        max_depth = max(depths.values(), default=0)
        for depth in range(max_depth + 1):
            op_seqs = sorted(seq for seq, value in depths.items() if value == depth)
            tasks: list[Task] = []
            level_nodes: list[dict[str, Any]] = []
            for op_seq in op_seqs:
                op = self.by_seq[op_seq]
                for token_index in range(len(self.tokens)):
                    uid = self.tokens[token_index]["uid"]
                    node = self.candidates[self.node_id(op["kind"], op_seq, uid)]
                    level_nodes.append(node)
                    if self.node_status[node["node_id"]] != "computed" and self.ledger_entry(node["node_id"])["action"] != "computed":
                        continue
                    if not self._dependency_nodes_ready(node):
                        self.node_status[node["node_id"]] = "blocked"
                        self.scheduler_report.pending.append(
                            {
                                "code": E_POINT_BUDGET_PENDING,
                                "task_id": node["node_id"],
                                "op_seq": op_seq,
                                "token_seq": token_index,
                                "reason": "upstream-not-completed",
                            }
                        )
                        continue
                    self.node_status[node["node_id"]] = "admitted"
                    tasks.append(
                        Task(
                            task_id=node["node_id"],
                            ordinal=node["token_seq"],
                            op_seq=op_seq,
                            token_seq=token_index,
                            kind=node["kind"],
                            work=lambda node=node: self.execute_node(node),
                        )
                    )
            results, self.scheduler_report = scheduler.run(tasks, serial=self.serial, report=self.scheduler_report)
            self._consume_results(tasks, results)

        self._run_final_wave(scheduler)
        self.store.node_index = self.new_index
        report = self.build_report()
        if with_width_reference and self.scheduler_report.status == "ok":
            report["width_reference"] = self.build_width_reference()
        else:
            report["width_reference"] = {"available": False, "reason": "primary-run-not-complete"}
        return report

    def ledger_entry(self, node_id: str) -> dict[str, Any]:
        return next(entry for entry in self.ledger_entries if entry["ordinal"] == self.candidates[node_id]["payload"]["identity"]["node_id"] or False) if False else next(
            entry for entry in self.ledger_entries if entry["cache_key"] == self.candidates[node_id]["key"] and entry["op_seq"] == self.candidates[node_id]["op_seq"] and entry["token_uid"] == self.candidates[node_id]["token_uid"]
        )

    def _op_depths(self) -> dict[int, int]:
        depths: dict[int, int] = {}

        def depth(seq: int) -> int:
            if seq in depths:
                return depths[seq]
            inputs = self.by_seq[seq]["inputs"]
            depths[seq] = 0 if not inputs else 1 + max(depth(ref) for ref in inputs)
            return depths[seq]

        for op in self.operators:
            depth(op["seq"])
        return depths

    def _consume_results(self, tasks: list[Task], results: dict[str, Any]) -> None:
        pending_ids = {item["task_id"] for item in self.scheduler_report.pending}
        quota_ids = {item["task_id"] for item in self.scheduler_report.quota_aborted}
        for task in sorted(tasks, key=lambda item: (item.op_seq, item.token_seq, item.ordinal)):
            node = self.candidates[task.task_id]
            if task.task_id in results:
                result = results[task.task_id]
                if result.get("ok"):
                    record = result["record"]
                    self.records_by_node[task.task_id] = record
                    self.store.put(record["cache_key"], record)
                    self.node_status[task.task_id] = "completed"
                    self.new_index[task.task_id] = {"cache_key": record["cache_key"], "payload": node["payload"]}
                else:
                    fault = result["fault"]
                    self.node_faults[task.task_id] = fault
                    quarantined = result.get("record")
                    if quarantined is not None:
                        self.records_by_node[task.task_id] = quarantined
                        self.store.put(quarantined["cache_key"], quarantined)
                        self.new_index[task.task_id] = {"cache_key": quarantined["cache_key"], "payload": node["payload"]}
                        self.node_status[task.task_id] = "quarantined"
                    else:
                        self.node_status[task.task_id] = "fault"
                    self._record_fault(node, fault)
                    if quarantined is None and task.task_id in self.old_index:
                        self.new_index[task.task_id] = self.old_index[task.task_id]
            elif task.task_id in pending_ids:
                self.node_status[task.task_id] = "pending"
                if task.task_id in self.old_index:
                    self.new_index[task.task_id] = self.old_index[task.task_id]
            elif task.task_id in quota_ids:
                self.node_status[task.task_id] = "quota_aborted"
            else:
                self.node_status[task.task_id] = "not_admitted"

    def _run_final_wave(self, scheduler: BudgetScheduler) -> None:
        terminal_seq = self.ordered_nodes[-1]["op_seq"]
        tasks: list[Task] = []
        for token_index in range(len(self.tokens)):
            uid = self.tokens[token_index]["uid"]
            node_id = self.node_id("final_result", terminal_seq, uid)
            node = self.candidates[node_id]
            action = next(entry for entry in self.ledger_entries if entry["cache_key"] == node["key"])["action"]
            if action == "reused":
                continue
            norm_status = self._terminal_norm_status_for_final(node)
            if norm_status != "completed":
                self.node_status[node_id] = norm_status
                if norm_status == "pending":
                    self.scheduler_report.pending.append(
                        {"code": E_POINT_BUDGET_PENDING, "task_id": node_id, "op_seq": terminal_seq,
                         "token_seq": token_index, "reason": "upstream-not-completed"}
                    )
                continue
            self.node_status[node_id] = "admitted"
            tasks.append(
                Task(task_id=node_id, ordinal=99, op_seq=terminal_seq, token_seq=token_index,
                     kind="final_result", work=lambda node=node: self.execute_node(node))
            )
        results, self.scheduler_report = scheduler.run(tasks, serial=self.serial, report=self.scheduler_report)
        self._consume_results(tasks, results)

    def _terminal_norm_status_for_final(self, final_node: dict[str, Any]) -> str:
        norm_key = final_node["payload"]["upstream"]["layer_norm_key"]
        for node_id, candidate in self.candidates.items():
            if candidate["key"] == norm_key and node_id.startswith(f"{final_node['op_seq']}:layer_norm:"):
                return self.node_status.get(node_id, "waiting")
        return "missing"

    def build_report(self) -> dict[str, Any]:
        token_results = [self._token_result(index) for index in range(len(self.tokens))]
        cache_records = []
        for node in self.ordered_nodes:
            record = self.records_by_node.get(node["node_id"])
            if record is not None:
                cache_records.append(record)
        stage_summaries = self._stage_summaries()
        fault_report = self._fault_report()
        status = "ok"
        if self.scheduler_report and self.scheduler_report.status != "ok":
            status = self.scheduler_report.status
        elif any(result["status"] == "fault" for result in token_results):
            status = "fault_injected"
        elif any(result["status"] == "pending" for result in token_results):
            status = E_POINT_BUDGET_PENDING
        return {
            "status": status,
            "config_hash": sha256_hex({k: v for k, v in self.config.items() if k != "faults"}),
            "chain_fingerprint": sha256_hex([op["fingerprint"] for op in sorted(self.operators, key=lambda x: x["seq"])]),
            "exchange_version": self.exchange_version,
            "token_results": token_results,
            "cache_records": cache_records,
            "stage_summaries": stage_summaries,
            "reuse_ledger": self.ledger_entries,
            "execution_order": [
                {
                    "ordinal": entry["ordinal"],
                    "cache_key": entry["cache_key"],
                    "op_seq": entry["op_seq"],
                    "token_seq": entry["token_seq"],
                    "token_uid": entry["token_uid"],
                    "kind": entry["kind"],
                    "action": entry["action"],
                }
                for entry in self.ledger_entries
            ],
            "scheduler": self.scheduler_report.to_dict() if self.scheduler_report else {},
            "diagnostics": list(self.scheduler_report.diagnostics if self.scheduler_report else []),
            "fault_report": fault_report,
            "ledger_context": self._ledger_context(),
        }

    def _token_result(self, index: int) -> dict[str, Any]:
        token = self.tokens[index]
        uid = token["uid"]
        cache_keys: dict[str, str] = {}
        actions: dict[str, str] = {}
        residuals: dict[str, int | None] = {}
        fault_ids: list[str] = []
        statuses: list[str] = []
        boundary = False
        vector: list[int] = []

        for op in sorted(self.operators, key=lambda item: item["seq"]):
            label = self.stage_label(op["kind"], op["seq"])
            node_id = self.node_id(op["kind"], op["seq"], uid)
            node = self.candidates[node_id]
            cache_keys[label] = node["key"]
            entry = next(item for item in self.ledger_entries if item["cache_key"] == node["key"])
            actions[label] = "injected_fault" if node_id in self.node_faults else entry["action"]
            record = self.records_by_node.get(node_id)
            residuals[label] = record["residual"] if record else None
            if record and record.get("boundary_token"):
                boundary = True
            fault = self.node_faults.get(node_id)
            if fault:
                fault_ids.append(fault["injection_id"])
            statuses.append(self.node_status.get(node_id, "unknown"))

        terminal_seq = self.topo_ops[-1]["seq"]
        final_id = self.node_id("final_result", terminal_seq, uid)
        final_node = self.candidates[final_id]
        cache_keys["final_result"] = final_node["key"]
        final_entry = next(item for item in self.ledger_entries if item["cache_key"] == final_node["key"])
        final_status = self.node_status.get(final_id, "unknown")
        actions["final_result"] = "injected_fault" if final_id in self.node_faults else final_entry["action"]
        final_record = self.records_by_node.get(final_id)
        residuals["final_result"] = final_record["residual"] if final_record else None
        if final_record:
            vector = final_record["vector"]
            boundary = bool(final_record.get("boundary_token", boundary))
        final_fault = self.node_faults.get(final_id)
        if final_fault:
            fault_ids.append(final_fault["injection_id"])

        if fault_ids:
            status = "fault"
        elif final_status == "completed":
            status = "ok"
        elif "pending" in statuses or final_status == "pending":
            status = "pending"
        else:
            status = "incomplete"
        return {
            "token_seq": index,
            "token_uid": uid,
            "token_value": token["token"],
            "segment_index": token["segment_index"],
            "segment_count": len(self.segments),
            "status": status,
            "vector": vector,
            "boundary_token": boundary,
            "cache_keys": cache_keys,
            "stage_actions": actions,
            "stage_residuals": residuals,
            "fault_ids": sorted(set(fault_ids)),
        }

    def _stage_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        groups = [(op["kind"], op["seq"], op["fingerprint"]) for op in sorted(self.operators, key=lambda x: x["seq"])]
        terminal_seq = self.topo_ops[-1]["seq"]
        groups.append(("final_result", terminal_seq, sha256_hex({"final": True})))
        for kind, op_seq, op_fingerprint in groups:
            node_ids = [
                node_id
                for node_id, node in self.candidates.items()
                if node["kind"] == kind and node["op_seq"] == op_seq
            ]
            entries = [entry for entry in self.ledger_entries if entry["kind"] == kind and entry["op_seq"] == op_seq]
            computed = sum(entry["action"] == "computed" for entry in entries)
            reused = sum(entry["action"] == "reused" for entry in entries)
            faulted = sum(self.node_status.get(node_id) == "fault" for node_id in node_ids)
            pending = sum(self.node_status.get(node_id) in ("pending", "blocked") for node_id in node_ids)
            residual_sum = 0
            steps_recorded = 0
            directions = {"positive": 0, "negative": 0, "zero": 0}
            keys: list[str] = []
            for node_id in sorted(node_ids, key=lambda nid: self.candidates[nid]["token_seq"]):
                record = self.records_by_node.get(node_id)
                keys.append(self.candidates[node_id]["key"])
                if not record:
                    continue
                residual_sum += int(record.get("residual", 0))
                steps = record.get("details", {}).get("rounding_steps", [])
                steps_recorded += len(steps)
                for step in steps:
                    direction = step.get("direction", 0)
                    if direction > 0:
                        directions["positive"] += 1
                    elif direction < 0:
                        directions["negative"] += 1
                    else:
                        directions["zero"] += 1
            summaries.append(
                {
                    "op_seq": op_seq,
                    "kind": kind,
                    "operator_fingerprint": op_fingerprint,
                    "stage_key": sha256_hex(keys),
                    "planned_tokens": len(self.tokens),
                    "computed_tokens": computed,
                    "reused_tokens": reused,
                    "completed_tokens": computed + reused,
                    "faulted_tokens": faulted,
                    "pending_tokens": pending,
                    "residual_sum": residual_sum,
                    "rounding_steps_recorded": steps_recorded,
                    "rounding_directions": directions,
                }
            )
        return summaries

    def _fault_report(self) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        for injection_id, hit in sorted(self.fault_hits.items()):
            hits = hit["hits"]
            token_indices = sorted({item["token_seq"] for item in hits})
            report.append(
                {
                    "injection_id": injection_id,
                    "type": hit["injection"].get("type"),
                    "hit": True,
                    "code": E_INJECTED_FAULT,
                    "target": hit["injection"].get("target", {}),
                    "op_seq": hit["injection"].get("op_seq", hit["injection"].get("target", {}).get("op_seq")),
                    "token_uids": [self.tokens[index]["uid"] for index in token_indices],
                    "hit_count": len(hits),
                    "hits": hits,
                    "isolation": "only listed token/node results carry fault_ids; all other tokens are independently computed",
                }
            )
        return report

    def _ledger_context(self) -> dict[str, Any]:
        return {
            "seed": self.config["seed"],
            "vocab_size": self.config["vocab_size"],
            "exchange_version": self.exchange_version,
            "fixed": {"frac_bits": self.config["frac_bits"], "quant_tier": self.config["quant_tier"]},
            "attention": {
                "heads": self.config["heads"],
                "dim": self.config["head_dim"],
                "temperature_permille": self.config.get("temperature_permille", 900),
            },
            "tokens": {token["uid"]: token["token"] for token in self.tokens},
            "token_segment": {token["uid"]: token["segment_index"] for token in self.tokens},
            "segment_members": {
                token["uid"]: [self.tokens[index]["uid"] for index in range(*self.segments[token["segment_index"]])]
                for token in self.tokens
            },
            "operators": {
                str(op["seq"]): {"kind": op["kind"], "inputs": op["inputs"], "fingerprint": op["fingerprint"]}
                for op in sorted(self.operators, key=lambda x: x["seq"])
            },
            "chain_fingerprint": sha256_hex([op["fingerprint"] for op in sorted(self.operators, key=lambda x: x["seq"])]),
        }

    def build_width_reference(self) -> dict[str, Any]:
        reference_config = copy.deepcopy(self.config)
        reference_config["frac_bits"] = int(self.config["frac_bits"]) + 1
        reference_engine = ChainEngine(
            reference_config,
            copy.deepcopy(self.operators),
            copy.deepcopy(self.tokens),
            list(self.boundaries),
            injections=[],
            exchange_version=self.exchange_version,
            store=CacheStore(),
            serial=True,
        )
        reference_report = reference_engine.run(with_width_reference=False)
        tolerance = Fraction(str(self.config.get("width_tolerance", "0.0005")))
        actual_bits = int(self.config["frac_bits"])
        reference_bits = actual_bits + 1
        stages: list[dict[str, Any]] = []
        first_exceeded: dict[str, Any] | None = None

        compare_groups = [(op["kind"], op["seq"]) for op in sorted(self.operators, key=lambda x: x["seq"])]
        for kind, op_seq in compare_groups:
            token_reports = []
            stage_exceeds = False
            token_count = 0
            max_overall = Fraction(0)
            sum_overall = Fraction(0)
            element_count = 0
            for token_index, token in enumerate(self.tokens):
                uid = token["uid"]
                node_id = self.node_id(kind, op_seq, uid)
                actual_record = self.records_by_node.get(node_id)
                reference_record = reference_engine.records_by_node.get(node_id)
                if not actual_record or not reference_record:
                    continue
                token_count += 1
                errors: list[Fraction] = []
                for dim, actual_value in enumerate(actual_record["vector"]):
                    reference_value = reference_record["vector"][dim]
                    error = abs(Fraction(2 * int(actual_value) - int(reference_value), 1 << reference_bits))
                    errors.append(error)
                token_max = max(errors)
                token_mean = sum(errors, Fraction(0)) / len(errors)
                exceeds = token_max > tolerance
                stage_exceeds = stage_exceeds or exceeds
                max_overall = max(max_overall, token_max)
                sum_overall += sum(errors, Fraction(0))
                element_count += len(errors)
                first_token_exceeded = None
                if exceeds:
                    for dim, error in enumerate(errors):
                        if error > tolerance:
                            first_token_exceeded = {"token_uid": uid, "token_seq": token_index, "dim": dim}
                            if first_exceeded is None:
                                first_exceeded = {"op_seq": op_seq, "kind": kind, "stage": kind, **first_token_exceeded}
                            break
                token_reports.append(
                    {
                        "token_uid": uid,
                        "exceeds_tolerance": exceeds,
                        "max_abs": fraction_to_float_text(token_max),
                        "mean_abs": fraction_to_float_text(token_mean),
                        "first_exceeded_dim": first_token_exceeded["dim"] if first_token_exceeded else None,
                    }
                )
            stages.append(
                {
                    "op_seq": op_seq,
                    "kind": kind,
                    "token_count": token_count,
                    "exceeds_tolerance": stage_exceeds,
                    "max_abs": fraction_to_float_text(max_overall),
                    "mean_abs": fraction_to_float_text(sum_overall / max(1, element_count)),
                    "tokens": token_reports,
                }
            )
        return {
            "available": True,
            "actual_frac_bits": actual_bits,
            "reference_frac_bits": reference_bits,
            "tolerance": fraction_to_float_text(tolerance),
            "reference_status": reference_report["status"],
            "first_exceeded_stage": first_exceeded,
            "stages": stages,
        }
