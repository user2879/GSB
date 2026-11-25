"""Run orchestration: validation, budget scheduling, cached deterministic
pipeline, fault injection, attribution, receipts, edits, and commits."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import __version__
from .core import (BizError, CrashFault, E_DUP_REQUEST, E_POINT_BUDGET_PENDING,
                   TIERS, config_from_dict, request_to_dict, validate_request)
from .formats import build_sidecar_entry, downgrade_entries, export_v1, export_v2
from .pipeline import (attribute_request, group_by_channel, layer_quotas,
                       normalize_channel, shuffle_layer, truncate)
from .store import Store, fp


def config_params_fp(config_raw):
    """Fingerprint of only the result-relevant config; operational knobs
    (threads, budgets) never change conclusions or shuffle seeds."""
    keys = ("topk", "sample_ratio_permille", "quant_tier", "norm_method",
            "channel_weights")
    return fp({k: config_raw.get(k) for k in keys})


def environment_check(cfg):
    cpus = os.cpu_count() or 1
    mem_mb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_mb = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    est_mb = round(cfg.batch_budget * 64 / 1e6, 3)
    return {"cpus": cpus, "threads": cfg.threads, "threads_ok": cfg.threads <= cpus,
            "mem_available_mb": mem_mb, "batch_estimate_mb": est_mb,
            "mem_ok": mem_mb is None or mem_mb > est_mb}


def schedule(valid, cfg, prev_pending):
    """Consume the batch budget in stable request-id order; requests over the
    per-request quota are parked with E_POINT_BUDGET_PENDING and get dedicated
    follow-up batches (the pending queue has priority on the next run)."""
    events, batches, deferred = [], [], []
    prev = [r for r in prev_pending if r in valid]
    order = prev + sorted(r for r in valid if r not in set(prev))
    cur, cost = [], 0
    for rid in order:
        n = len(valid[rid].candidates)
        if n > cfg.per_request_quota:
            events.append({"request_id": rid, "code": E_POINT_BUDGET_PENDING,
                           "cost": n, "quota": cfg.per_request_quota})
            deferred.append(rid)
            continue
        if cur and cost + n > cfg.batch_budget:
            batches.append(cur)
            cur, cost = [], 0
        cur.append(rid)
        cost += n
    if cur:
        batches.append(cur)
    for rid in deferred:
        batches.append([rid])
        events.append({"request_id": rid, "code": "BUDGET_DEDICATED_BATCH",
                       "cost": len(valid[rid].candidates)})
    return batches, events, deferred


def _apply_fault(req, fault, cfg):
    """Apply a real input fault, then let normal validation map the E-code."""
    if fault == "corrupt_score":
        req.candidates[0].score = float("nan")
    elif fault == "unknown_channel":
        req.candidates[0].channel = "ghost_channel"
    elif fault == "drop_candidates":
        req.candidates = []
    validate_request(request_to_dict(req), cfg)
    return req


def process_request(req, cfg, config_fp, cache, faults_for_req):
    """Run one request through all cached stages. Returns conclusion, reuse
    map, cache keys, new cache entries and the attribution report."""
    reuse = {"norm": {}, "trunc": None, "layer": {}, "ser": None}
    keys = {"norm": {}, "trunc": None, "layer": {}, "ser": None}
    puts = {}

    fault = faults_for_req.get("normalize")
    if fault == "crash":
        raise CrashFault("injected crash at normalize")
    if fault:
        req = _apply_fault(req, fault, cfg)

    by_ch = group_by_channel(req)
    scores = {}
    for ch, cands in by_ch.items():
        key = "norm:" + fp([[(c.candidate_id, c.score, c.aux) for c in cands],
                            cfg.channel_weights[ch], cfg.quant_tier, cfg.norm_method])
        keys["norm"][ch] = key
        hit = cache.get(key)
        if hit is None:
            res = normalize_channel(cands, cfg.channel_weights[ch], cfg.frac, cfg.norm_method)
            puts[key] = res
            reuse["norm"][ch] = "recompute"
        else:
            res = hit
            reuse["norm"][ch] = "reuse"
        scores.update({int(k): v for k, v in res.items()})

    if faults_for_req.get("truncate") == "crash":
        raise CrashFault("injected crash at truncate")
    tkey = "trunc:" + fp([keys["norm"], cfg.topk, sorted(scores)])
    keys["trunc"] = tkey
    hit = cache.get(tkey)
    if hit is None:
        trunc = truncate(scores, cfg.topk, cfg.quant_tier)
        puts[tkey] = trunc
        reuse["trunc"] = "recompute"
    else:
        trunc = hit
        reuse["trunc"] = "reuse"

    ratio = cfg.sample_ratio_permille
    if faults_for_req.get("sample") == "bad_ratio":
        ratio = -5  # real invalid input, mapped by stratified validation
    if not (0 <= ratio <= 1000):
        raise BizError("E_INVALID_RATIO", "injected bad_ratio", req.request_id)
    if faults_for_req.get("sample") == "crash":
        raise CrashFault("injected crash at sample")

    table = {c.candidate_id: c for c in req.candidates}
    selected = trunc["selected"]
    layer_sizes = {}
    for cid in selected:
        ch = table[cid].channel
        layer_sizes[ch] = layer_sizes.get(ch, 0) + 1
    total_quota, base_quota = layer_quotas(layer_sizes, ratio)
    layers = {}
    for ch in sorted(layer_sizes):
        members = sorted(cid for cid in selected if table[cid].channel == ch)
        lkey = "layer:" + fp([tkey, ch, ratio, config_fp, req.request_id,
                              [(cid, table[cid].aux) for cid in members]])
        keys["layer"][ch] = lkey
        hit = cache.get(lkey)
        if hit is None:
            order = shuffle_layer(members, lambda cid: table[cid].aux,
                                  config_fp, req.request_id, ch)
            puts[lkey] = {"order": order}
            reuse["layer"][ch] = "recompute"
        else:
            order = hit["order"]
            reuse["layer"][ch] = "reuse"
        quota = base_quota[ch]
        layer = {"quota": quota, "order": order,
                 "picked": order[:quota], "dropped": order[quota:]}
        layers[ch] = layer

    if faults_for_req.get("serialize") == "crash":
        raise CrashFault("injected crash at serialize")
    skey = "ser:" + fp([sorted(keys["layer"].values()), cfg.output_version,
                        req.request_id, tkey])
    keys["ser"] = skey
    conclusion = {
        "request_id": req.request_id,
        "quant_tier": cfg.quant_tier,
        "scores": {str(cid): s for cid, s in scores.items()},
        "truncated": list(trunc["selected"]),
        "truncation_order": list(trunc["order"]),
        "boundary": [dict(b) for b in trunc["boundary"]],
        "layers": layers,
        "total_quota": total_quota,
        "version": cfg.output_version,
    }
    hit = cache.get(skey)
    if hit is None:
        artifact = ({"v1_lines": export_v1(conclusion)[0]} if cfg.output_version == "v1"
                    else {"v2_record": export_v2(conclusion)})
        puts[skey] = artifact
        reuse["ser"] = "recompute"
    else:
        artifact = hit
        reuse["ser"] = "reuse"

    attribution = attribute_request(req, cfg, config_fp)
    return {"conclusion": conclusion, "reuse": reuse, "keys": keys,
            "puts": puts, "artifact": artifact, "attribution": attribution}


class Runner:
    """Stateful runner over a Store; each full_run ends in one atomic commit."""

    def __init__(self, store_dir):
        self.store = Store(store_dir)

    def set_inputs(self, requests_raw, config_raw):
        config_from_dict(config_raw)  # validate eagerly
        self.store.state["requests"] = requests_raw
        self.store.state["config"] = config_raw

    def config(self):
        return config_from_dict(self.store.state["config"])

    # ------------------------------------------------------------- edits
    def apply_edit(self, op):
        state, cfg_raw = self.store.state, self.store.state["config"]
        kind = op.get("op")
        if kind == "append_request":
            state["requests"].append(op["request"])
        elif kind == "replace_candidate":
            rid, cand = str(op["request_id"]), op["candidate"]
            for raw in state["requests"]:
                if str(raw.get("request_id")) == rid:
                    for i, c in enumerate(raw["candidates"]):
                        if c.get("candidate_id") == cand.get("candidate_id"):
                            raw["candidates"][i] = cand
                            break
                    else:
                        raise BizError("E_BAD_ID", "candidate to replace not found", rid)
                    break
            else:
                raise BizError("E_BAD_ID", "request to edit not found", rid)
        elif kind == "set_channel_weight":
            if op["channel"] not in cfg_raw["channel_weights"]:
                raise BizError("E_UNKNOWN_CHANNEL", "unknown channel %r" % op["channel"])
            cfg_raw["channel_weights"][op["channel"]] = op["weight"]
        elif kind == "set_sample_ratio":
            cfg_raw["sample_ratio_permille"] = op["permille"]
        elif kind == "set_topk":
            cfg_raw["topk"] = op["topk"]
        elif kind == "set_version":
            cfg_raw["output_version"] = op["version"]
        elif kind == "set_quant_tier":
            cfg_raw["quant_tier"] = op["tier"]
        else:
            raise BizError("E_BAD_CONFIG", "unknown edit op %r" % kind)
        config_from_dict(cfg_raw)  # re-validate edited config
        return self.full_run(note="edit:" + kind)

    # ------------------------------------------------------------- run
    def full_run(self, faults=None, note=""):
        cfg = self.config()
        state = self.store.state
        config_fp = config_params_fp(state["config"])
        data_fp = fp(state["requests"])
        manifest = self.store.manifest
        decision = "rebuild"
        if manifest and manifest.get("config_fp") == config_fp \
                and manifest.get("data_fp") == data_fp \
                and manifest.get("version") == __version__:
            decision = "resume"
        self.store.log({"type": "run_decision", "decision": decision, "note": note,
                        "config_fp": config_fp, "data_fp": data_fp})

        valid, failures = {}, []
        for idx, raw in enumerate(state["requests"]):
            rid = str(raw.get("request_id")) if isinstance(raw, dict) else None
            try:
                req = validate_request(raw, cfg)
                if req.request_id in valid:
                    raise BizError(E_DUP_REQUEST, "duplicate request_id", rid)
                valid[req.request_id] = req
            except BizError as exc:
                failures.append({"index": idx, "request_id": rid,
                                 "code": exc.code, "message": exc.message})
        for entry in failures:
            self.store.log({"type": "failure", **entry})

        fault_map = {}
        for f in faults or []:
            fault_map.setdefault(str(f["request_id"]), {})[f["stage"]] = f["fault"]

        batches, budget_events, deferred = schedule(valid, cfg, state.get("pending", []))
        for ev in budget_events:
            self.store.log({"type": "budget", **ev})

        conclusions, reuse_map, keys_map, attributions, artifacts = {}, {}, {}, {}, {}
        timings = {}
        for batch_no, batch in enumerate(batches):
            results = self._run_batch(batch, valid, cfg, config_fp, fault_map, timings)
            for rid in sorted(results):
                res = results[rid]
                if "error" in res:
                    err = {"index": None, "request_id": rid, "code": res["error"],
                           "message": res["message"]}
                    failures.append(err)
                    self.store.log({"type": "failure", **err})
                    continue
                conclusions[rid] = res["conclusion"]
                reuse_map[rid] = res["reuse"]
                keys_map[rid] = res["keys"]
                attributions[rid] = res["attribution"]
                artifacts[rid] = res["artifact"]
                self.store.cache.update(res["puts"])
                self.store.log({"type": "reuse_map", "request_id": rid,
                                "batch": batch_no, "stages": res["reuse"],
                                "cache_keys": res["keys"]})
                self.store.log({"type": "attribution", **res["attribution"]})

        arbitrations = self._update_receipts(conclusions, cfg, timings)
        outputs, sidecar = self._build_outputs(conclusions, artifacts, cfg, failures)
        self.store.outputs = outputs
        env = environment_check(cfg)
        self.store.report = {
            "note": note, "decision": decision, "environment": env,
            "batches": [list(b) for b in batches], "budget_events": budget_events,
            "deferred": deferred, "reuse_map": reuse_map, "cache_keys": keys_map,
            "attributions": attributions, "arbitrations": arbitrations,
            "failures": failures, "timings_ms": timings,
        }
        state["pending"] = deferred
        state["run_seq"] = state.get("run_seq", 0) + 1
        self.store.manifest = {"version": __version__, "config_fp": config_fp,
                               "data_fp": data_fp, "run_seq": state["run_seq"]}
        self.store.commit()
        return {"conclusions": len(conclusions), "failures": len(failures),
                "batches": len(batches), "decision": decision}

    def _run_batch(self, batch, valid, cfg, config_fp, fault_map, timings):
        results = {}

        def work(rid):
            start = time.perf_counter()
            req = valid[rid]
            faults_for_req = fault_map.get(rid, {})
            try:
                out = process_request(req, cfg, config_fp, self.store.cache,
                                      faults_for_req)
            except BizError as exc:
                out = {"error": exc.code, "message": exc.message}
            timings[rid] = int((time.perf_counter() - start) * 1000)
            return rid, out

        if cfg.threads > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=cfg.threads) as pool:
                for rid, out in pool.map(work, batch):
                    results[rid] = out
        else:
            for rid in batch:
                _, out = work(rid)
                results[rid] = out
        return results

    def _update_receipts(self, conclusions, cfg, timings):
        receipts = self.store.state.setdefault("receipts", {})
        for rid, concl in conclusions.items():
            receipts.setdefault(rid, {})[cfg.output_version] = {
                "elapsed_ms": int(timings.get(rid, 0)), "digest": fp(concl)}
        arbitrations = []
        for rid in sorted(receipts):
            versions = receipts[rid]
            if len(versions) > 1:
                winner = max(versions)  # higher version overrides
                arb = {"type": "arbitration", "request_id": rid, "winner": winner,
                       "basis": "higher_version",
                       "overridden": sorted(v for v in versions if v != winner),
                       "elapsed_ms": {v: versions[v]["elapsed_ms"] for v in versions}}
                arbitrations.append(arb)
                self.store.log(arb)
        return arbitrations

    def _build_outputs(self, conclusions, artifacts, cfg, failures):
        sidecar = {}
        downgrade_logs = []
        if cfg.output_version == "v1":
            lines = []
            for rid in sorted(conclusions):
                lines.extend(artifacts[rid]["v1_lines"])
                sidecar[rid] = build_sidecar_entry(conclusions[rid])
                downgrade_logs.append(downgrade_entries(conclusions[rid], "v2->v1"))
            artifact_view = {"format": "v1", "lines": lines}
        else:
            records = [artifacts[rid]["v2_record"] for rid in sorted(conclusions)]
            for rid in sorted(conclusions):
                sidecar[rid] = build_sidecar_entry(conclusions[rid])
            artifact_view = {"format": "v2", "records": records}
        for entry in downgrade_logs:
            self.store.log(entry)
        outputs = {"version": cfg.output_version, "artifacts": artifact_view,
                   "sidecar": sidecar,
                   "conclusions": {rid: conclusions[rid] for rid in sorted(conclusions)},
                   "failures": failures}
        return outputs, sidecar
