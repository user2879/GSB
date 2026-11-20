"""独立异常注入：从 YAML 读取注入计划，命中情况写入诊断报告。"""

import yaml


class InjectionPlan:
    def __init__(self, injections):
        self.injections = list(injections)
        self._hits = {inj["id"]: False for inj in self.injections}

    @classmethod
    def load(cls, path):
        if path is None:
            return cls([])
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(raw.get("injections", []) or [])

    def for_sample(self, sample_id, inj_type=None):
        out = []
        for inj in self.injections:
            if inj.get("sample_id") != sample_id:
                continue
            if inj_type is not None and inj.get("type") != inj_type:
                continue
            out.append(inj)
        return out

    def mark_hit(self, injection_id):
        if injection_id in self._hits:
            self._hits[injection_id] = True

    def report(self):
        return [
            {
                "id": inj["id"],
                "type": inj.get("type"),
                "sample_id": inj.get("sample_id"),
                "stage": inj.get("stage", "load_validate"),
                "hit": self._hits.get(inj["id"], False),
            }
            for inj in sorted(self.injections, key=lambda i: i["id"])
        ]
