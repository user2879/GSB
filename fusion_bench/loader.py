"""样本与配置加载层：PNG 容器头校验、尺寸一致性、样本 ID 唯一性、
断言字段完整性、分块整除性校验。不合法样本进入失败清单，其余继续。"""

import hashlib
import io
import json
import os

import numpy as np
from PIL import Image

from .errors import E_IMAGE_FORMAT, E_SAMPLE_INVALID, PipelineError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class Sample:
    def __init__(self, sample_id, image, assertions, block_size):
        self.sample_id = sample_id
        self.image = image              # uint8 HxWx3
        self.assertions = assertions    # list[dict] 或 None（缺失 -> 默认补齐）
        self.block_size = block_size

    @property
    def n_blocks(self):
        bs = self.block_size
        n = self.image.shape[0] // bs
        return n * n


def input_fingerprint(samples_dir):
    """输入指纹：排序后的文件名 + 内容哈希 + 描述文件哈希。"""
    h = hashlib.sha256()
    images_dir = os.path.join(samples_dir, "images")
    for name in sorted(os.listdir(images_dir)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(images_dir, name), "rb") as fh:
            h.update(name.encode())
            h.update(hashlib.sha256(fh.read()).digest())
    desc_path = os.path.join(samples_dir, "descriptions.jsonl")
    with open(desc_path, "rb") as fh:
        h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _read_image_bytes(images_dir, sample_id, plan):
    path = os.path.join(images_dir, f"{sample_id}.png")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as fh:
        data = fh.read()
    inj_id = None
    for inj in plan.for_sample(sample_id, "truncated_block"):
        n = int(inj.get("truncate_bytes", 0))
        data = data[: max(0, len(data) - n)]
        plan.mark_hit(inj["id"])
        inj_id = inj["id"]
    return data, inj_id


def _decode_png(data, sample_id):
    if data is None or not data.startswith(PNG_SIGNATURE):
        raise PipelineError(E_IMAGE_FORMAT, "load_validate",
                            f"{sample_id}: invalid PNG container header")
    try:
        with Image.open(io.BytesIO(data)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise PipelineError(E_IMAGE_FORMAT, "load_validate",
                            f"{sample_id}: PNG decode failed: {exc}")
    return arr


def _validate_assertions(sample_id, entry, plan):
    if "sample_id" not in entry or not isinstance(entry.get("sample_id"), str):
        raise PipelineError(E_SAMPLE_INVALID, "load_validate",
                            f"{sample_id}: missing or invalid sample_id field")
    assertions = entry.get("assertions")
    if assertions is None:
        return None  # 断言缺失 -> 后续按默认向量补齐
    corrupt_id = None
    for inj in plan.for_sample(sample_id, "corrupt_assertion"):
        assertions = ["corrupted"]
        plan.mark_hit(inj["id"])
        corrupt_id = inj["id"]
    if not isinstance(assertions, list):
        raise PipelineError(E_SAMPLE_INVALID, "load_validate",
                            f"{sample_id}: assertions must be a list",
                            injection_id=corrupt_id)
    for i, item in enumerate(assertions):
        if not isinstance(item, dict):
            raise PipelineError(E_SAMPLE_INVALID, "load_validate",
                                f"{sample_id}: assertion #{i} must be an object",
                                injection_id=corrupt_id)
        for field in ("entity", "attribute"):
            if field not in item or not isinstance(item[field], str):
                raise PipelineError(
                    E_SAMPLE_INVALID, "load_validate",
                    f"{sample_id}: assertion #{i} missing field {field}",
                    injection_id=corrupt_id)
        if "value" not in item:
            raise PipelineError(E_SAMPLE_INVALID, "load_validate",
                                f"{sample_id}: assertion #{i} missing field value",
                                injection_id=corrupt_id)
    return assertions


def load_samples(samples_dir, cfg, plan):
    """返回 (有效样本列表(按 ID 排序), 失败清单, 输入指纹)。"""
    images_dir = os.path.join(samples_dir, "images")
    desc_path = os.path.join(samples_dir, "descriptions.jsonl")
    failures = []
    fingerprint = input_fingerprint(samples_dir)

    entries = {}
    with open(desc_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append({
                    "sample_id": f"<line {line_no}>",
                    "stage": "load_validate",
                    "code": E_SAMPLE_INVALID,
                    "message": f"invalid JSONL line {line_no}: {exc}",
                    "injection_id": None,
                })
                continue
            sid = entry.get("sample_id") if isinstance(entry, dict) else None
            if not isinstance(sid, str) or not sid:
                failures.append({
                    "sample_id": f"<line {line_no}>",
                    "stage": "load_validate",
                    "code": E_SAMPLE_INVALID,
                    "message": f"line {line_no}: missing sample_id",
                    "injection_id": None,
                })
                continue
            if sid in entries:
                failures.append({
                    "sample_id": sid,
                    "stage": "load_validate",
                    "code": E_SAMPLE_INVALID,
                    "message": f"duplicate sample_id: {sid}",
                    "injection_id": None,
                })
                entries[sid] = None  # 标记该 ID 失效
                continue
            entries[sid] = entry

    samples = []
    for sid in sorted(entries):
        entry = entries[sid]
        if entry is None:
            continue  # 重复 ID，已记录失败
        try:
            assertions = _validate_assertions(sid, entry, plan)
            data, trunc_inj_id = _read_image_bytes(images_dir, sid, plan)
            if data is None:
                raise PipelineError(E_SAMPLE_INVALID, "load_validate",
                                    f"{sid}: image file missing")
            try:
                image = _decode_png(data, sid)
            except PipelineError as exc:
                if exc.injection_id is None:
                    exc.injection_id = trunc_inj_id
                raise
            if image.shape[0] != cfg.image_height or image.shape[1] != cfg.image_width:
                raise PipelineError(
                    E_IMAGE_FORMAT, "load_validate",
                    f"{sid}: image size {image.shape[1]}x{image.shape[0]} "
                    f"!= expected {cfg.image_width}x{cfg.image_height}")
            block_size = cfg.block_size
            for inj in plan.for_sample(sid, "indivisible_block"):
                block_size = int(inj["block_size"])
                plan.mark_hit(inj["id"])
            if (image.shape[0] % block_size != 0
                    or image.shape[1] % block_size != 0):
                inj_ids = plan.for_sample(sid, "indivisible_block")
                raise PipelineError(
                    E_IMAGE_FORMAT, "load_validate",
                    f"{sid}: block size {block_size} does not divide image edge",
                    injection_id=inj_ids[0]["id"] if inj_ids else None)
            samples.append(Sample(sid, image, assertions, block_size))
        except PipelineError as exc:
            failures.append(exc.to_record(sid))

    # 图像无对应文本描述的样本
    described = set(entries)
    for name in sorted(os.listdir(images_dir)):
        if name.endswith(".png") and name[:-4] not in described:
            failures.append({
                "sample_id": name[:-4],
                "stage": "load_validate",
                "code": E_SAMPLE_INVALID,
                "message": f"{name[:-4]}: no text description entry",
                "injection_id": None,
            })

    failures.sort(key=lambda f: (f["sample_id"], f["stage"]))
    return samples, failures, fingerprint
