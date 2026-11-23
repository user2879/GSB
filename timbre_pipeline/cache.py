"""基于内容指纹的缓存键、上游引用账本与编辑命令。"""

from .utils import canonical_json, sha256_text


def make_cache_key(stage, *parts):
    return "{}:{}".format(stage, sha256_text(canonical_json(list(parts)))[:20])


class CacheLedger:
    """阶段产物缓存 + 每段/每说话人上游引用 + 复用/重算顺序账本。"""

    def __init__(self):
        self.entries = {}
        self.references = []
        self.events = []
        self.command_cursor = 0

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        self.entries[key] = value

    def record_reference(self, target, stage, key, upstream_keys):
        self.references.append({
            "target": target,
            "stage": stage,
            "cache_key": key,
            "upstream": list(upstream_keys),
        })

    def record_event(self, command_id, target, stage, action, key, reason=""):
        self.events.append({
            "seq": len(self.events),
            "command_id": command_id,
            "target": target,
            "stage": stage,
            "action": action,
            "cache_key": key,
            "reason": reason,
        })

    def invalidate_targets(self, command_id, targets, stages, reason):
        """按目标与阶段失效引用，并从缓存删除对应键。"""
        target_set = set(targets)
        removed = []
        kept_references = []
        for reference in self.references:
            if reference["target"] in target_set and reference["stage"] in stages:
                self.entries.pop(reference["cache_key"], None)
                removed.append(reference)
            else:
                kept_references.append(reference)
        self.references = kept_references
        for reference in removed:
            self.record_event(command_id, reference["target"], reference["stage"],
                              "invalidate", reference["cache_key"], reason)
        return removed

    def downstream_targets(self, targets):
        """沿上游引用图收集全部下游目标。"""
        result = set(targets)
        changed = True
        while changed:
            changed = False
            for reference in self.references:
                upstream_targets = {
                    ref["target"] for ref in self.references
                    if ref["cache_key"] in reference["upstream"]
                }
                if upstream_targets & result and reference["target"] not in result:
                    result.add(reference["target"])
                    changed = True
        return result

    def snapshot(self):
        return {
            "entries": self.entries,
            "references": self.references,
            "events": self.events,
            "command_cursor": self.command_cursor,
        }

    def restore(self, payload):
        self.entries = dict(payload["entries"])
        self.references = list(payload["references"])
        self.events = list(payload["events"])
        self.command_cursor = int(payload["command_cursor"])


def validate_command_index(segments, index):
    return 0 <= index < len(segments)


def apply_split(config, segments, command):
    """在段内采样偏移处把一段切为两段，指纹与边界真实变化。"""
    index = int(command["segment_index"])
    if not validate_command_index(segments, index):
        raise IndexError("segment_index 越界")
    target = segments[index]
    offset = int(command.get("sample_offset", (target["end_sample"] - target["start_sample"]) // 2))
    split_sample = target["start_sample"] + offset
    if not target["start_sample"] + 1 < split_sample < target["end_sample"] - 1:
        raise ValueError("切分点必须落在段内部")
    hop = config.hop_length()
    frame_len = config.frame_length()
    left_end = split_sample
    right_start = split_sample
    left_frame_end = target["frame_start"] + max(0, (left_end - target["start_sample"] - frame_len) // hop)
    right_frame_start = target["frame_start"] + max(0, (right_start - target["start_sample"]) // hop)
    if right_frame_start <= target["frame_start"] or left_frame_end >= target["frame_end"]:
        raise ValueError("切分后至少一侧不足一帧")
    left = dict(target)
    left.update({
        "end_sample": int(split_sample),
        "frame_end": int(left_frame_end),
        "frame_count": int(left_frame_end - target["frame_start"] + 1),
        "origin": "split",
        "edit_history": target["edit_history"] + [command["command_id"]],
    })
    right = dict(target)
    right.update({
        "start_sample": int(split_sample),
        "frame_start": int(right_frame_start),
        "frame_count": int(target["frame_end"] - right_frame_start + 1),
        "origin": "split",
        "edit_history": target["edit_history"] + [command["command_id"]],
    })
    segments[index:index + 1] = [left, right]
    _reindex(segments)
    return [left["segment_id"], right["segment_id"]]


def apply_merge(config, segments, command):
    """合并相邻两段；采样与帧范围取并集。"""
    index = int(command["segment_index"])
    if index < 0 or index + 1 >= len(segments):
        raise IndexError("merge 需要相邻的两个段")
    left = segments[index]
    right = segments[index + 1]
    if left["end_sample"] > right["start_sample"]:
        raise ValueError("只能合并时间上相邻的段")
    merged = dict(left)
    merged.update({
        "end_sample": right["end_sample"],
        "frame_end": right["frame_end"],
        "frame_count": right["frame_end"] - left["frame_start"] + 1,
        "origin": "merge",
        "edit_history": left["edit_history"] + right["edit_history"] + [command["command_id"]],
    })
    segments[index:index + 2] = [merged]
    _reindex(segments)
    return [merged["segment_id"]]


def apply_feature_modify(segments, command):
    """修改段特征输入：通过内容版本号使内容指纹真实变化。"""
    index = int(command["segment_index"])
    if not validate_command_index(segments, index):
        raise IndexError("segment_index 越界")
    target = segments[index]
    target["feature_override"] = {
        "dimension": command["dimension"],
        "value": command["value"],
        "command_id": command["command_id"],
    }
    target["edit_history"] = target["edit_history"] + [command["command_id"]]
    target["content_fingerprint"] = sha256_text(canonical_json({
        "base": target["content_fingerprint"],
        "override": target["feature_override"],
    }))
    return [target["segment_id"]]


def apply_command(config, segments, command):
    """命令分发，返回受影响的新段 ID 集合。"""
    kind = command["type"]
    if kind == "split":
        return apply_split(config, segments, command)
    if kind == "merge":
        return apply_merge(config, segments, command)
    if kind == "modify_feature":
        return apply_feature_modify(segments, command)
    raise ValueError("未知编辑命令: {}".format(kind))


def _reindex(segments):
    audio_id = segments[0]["segment_id"].split("#")[0]
    for index, segment in enumerate(segments):
        segment["segment_index"] = index
        segment["segment_id"] = "{}#s{:06d}".format(audio_id, index)
