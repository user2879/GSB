"""独立注入清单：加载、段范围校验与按音频/段/阶段路由。"""

import json
import os

from .errors import PipelineError, E_INJECT_SEGMENT_OUT_OF_RANGE


class InjectionBook:
    def __init__(self, items):
        self.items = [item for item in items if item.get("enabled", True)]

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path):
            return cls([])
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def for_audio(self, audio_id):
        return [item for item in self.items if item.get("audio") == audio_id]

    def for_target(self, audio_id, segment_index, stage):
        for item in self.items:
            if item.get("audio") != audio_id or item.get("stage") != stage:
                continue
            if int(item.get("segment", -1)) == int(segment_index):
                return item
        return None

    def validate_segment_range(self, audio_id, segment_count):
        """注入段超出音频范围时返回显式故障记录，不静默忽略。"""
        records = []
        for item in self.for_audio(audio_id):
            index = int(item.get("segment", -1))
            if index < 0 or index >= segment_count:
                records.append(PipelineError(
                    E_INJECT_SEGMENT_OUT_OF_RANGE,
                    "注入段 {} 超出音频 {} 段范围 [0,{})".format(index, audio_id, segment_count),
                    item.get("stage"),
                    "{}#s{:06d}".format(audio_id, max(index, 0)),
                    extra={"injection_id": item["id"], "segment_index": index,
                           "segment_count": segment_count},
                ).to_record())
        return records

    def threshold_item(self, audio_id):
        for item in self.items:
            if item.get("audio") == audio_id and item.get("type") == "threshold_out_of_range":
                return item
        return None
