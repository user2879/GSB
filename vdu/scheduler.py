"""Evidence budget/quota scheduling with persistent todo retention."""

from .errors import EvidenceBudgetPending, EvidenceQuotaError


class BatchScheduler:
    """Consumes segments in stable order under point-budget constraints."""

    def __init__(self, config, ledger):
        sched = config["scheduler"]
        self.max_workers = sched["max_workers"]
        self.batch_budget = sched["batch_budget"]
        self.segment_quota = sched["segment_quota"]
        self.ledger = ledger
        self.todo = []
        self.batch_reports = []

    def enqueue(self, segment_ids, points):
        for seg_id in segment_ids:
            if all(item["segment_id"] != seg_id for item in self.todo):
                self.todo.append({
                    "segment_id": seg_id,
                    "points": points[seg_id],
                    "status": "pending",
                    "error": None,
                })

    def _stable_order(self, items, segment_meta=None):
        def key(item):
            meta = (segment_meta or {}).get(item["segment_id"], {})
            return (meta.get("start", 0), meta.get("speaker", ""),
                    item["segment_id"])
        return sorted(items, key=key)

    def next_batch(self, segment_meta):
        """Return runnable segment ids; record quota and budget markers."""
        self.todo = self._stable_order(self.todo, segment_meta)
        runnable = []
        remaining = []
        budget = self.batch_budget
        report = {"budget": budget, "ran": [], "quota_blocked": [],
                  "budget_pending": []}
        for item in self.todo:
            seg_id = item["segment_id"]
            if item["points"] > self.segment_quota:
                item["status"] = "quota_exceeded"
                item["error"] = "E_EVIDENCE_QUOTA_EXCEEDED"
                report["quota_blocked"].append(seg_id)
                self.ledger.record(
                    f"scheduler:{seg_id}", "scheduler",
                    {"segment_id": seg_id}, "quota_exceeded",
                    f"points {item['points']} > quota "
                    f"{self.segment_quota}")
                remaining.append(item)
                continue
            if item["points"] > budget:
                item["status"] = "budget_pending"
                item["error"] = "E_EVIDENCE_BUDGET_PENDING"
                report["budget_pending"].append(seg_id)
                self.ledger.record(
                    f"scheduler:{seg_id}", "scheduler",
                    {"segment_id": seg_id}, "budget_pending",
                    f"needs {item['points']} points, {budget} remain")
                remaining.append(item)
                continue
            item["status"] = "running"
            budget -= item["points"]
            runnable.append(seg_id)
            report["ran"].append(seg_id)
        self.todo = [item for item in remaining
                     if item["status"] != "quota_exceeded"]
        report["budget_remaining"] = budget
        self.batch_reports.append(report)
        return runnable, report

    def mark_done(self, segment_ids):
        done = set(segment_ids)
        self.todo = [item for item in self.todo
                     if item["segment_id"] not in done]

    def has_todo(self):
        return bool(self.todo)

    def quota_errors(self):
        if not self.batch_reports:
            return []
        return [EvidenceQuotaError(
            f"segment {seg_id} exceeded quota")
            for seg_id in self.batch_reports[-1]["quota_blocked"]]

    def budget_errors(self):
        latest = self.batch_reports[-1] if self.batch_reports else {}
        return [EvidenceBudgetPending(
            f"segment {seg_id} waits for next batch")
            for seg_id in latest.get("budget_pending", [])]
