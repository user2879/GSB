"""v1 交换文件：定宽文本行。

行布局（每个字段固定宽度，零填充，整数有符号）：
  头行:
    #MVRB1
    #SIZE W=<4 位> H=<4 位>
    #VOX <10 位有符号定点> #FPB <4 位>
  记录行（总长度固定，长度不符或标签无法识别即 E_FIELD_TRUNCATED）:
    F| view(4) feature_id(6) x(6) y(6) landmark_id(6)
    M| view_a(4) feat_a(6) view_b(4) feat_b(6) score(4) metric_code(2)
    V| vx(7) vy(7) vz(7) rx(10) ry(10) rz(10) landmark_id(6)
"""
from __future__ import annotations

import os

from .errors import E_FIELD_TRUNCATED, E_PARSE, FailureList
from .ir import Feature, Match, ReconstructionIR, VoxelPoint
from .utils import fixed_to_float, float_to_fixed

WIDTHS = {
    "F": [4, 6, 6, 6, 6],                 # view, fid, x, y, lid
    "M": [4, 6, 4, 6, 4, 2],              # va, fa, vb, fb, score, metric
    "V": [7, 7, 7, 10, 10, 10, 6],        # vx, vy, vz, rx, ry, rz, lid
}
TAG_LEN = 2
LINE_LEN = {tag: TAG_LEN + sum(ws) for tag, ws in WIDTHS.items()}


def _slice_fields(line: str, widths: list) -> list:
    start = TAG_LEN
    out = []
    for w in widths:
        out.append(line[start:start + w])
        start += w
    return out


def _parse_int(token: str) -> int:
    token = token.strip()
    if token == "":
        raise ValueError("empty integer field")
    return int(token, 10)  # 支持 +0001 / -0001


def parse_v1(text: str) -> ReconstructionIR:
    """解析 v1 文本为统一 IR；字段错位/宽度不足进入失败清单。"""
    failures = FailureList()
    width = height = None
    voxel_fixed = None
    fixed_bits = None
    features, matches, voxels = [], [], []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if line.startswith("#"):
            if line.startswith("#SIZE"):
                try:
                    parts = line.split()
                    width = int(parts[1].split("=")[1])
                    height = int(parts[2].split("=")[1])
                except (IndexError, ValueError):
                    failures.fail(E_PARSE, f"bad #SIZE header at line {lineno}", stage="load")
            elif line.startswith("#VOX"):
                try:
                    parts = line.split()
                    voxel_fixed = int(parts[1], 10)
                    fixed_bits = int(parts[2].split("=")[1], 10)
                except (IndexError, ValueError):
                    failures.fail(E_PARSE, f"bad #VOX header at line {lineno}", stage="load")
            continue

        if len(line) < TAG_LEN:
            failures.fail(
                E_FIELD_TRUNCATED,
                f"line {lineno} shorter than tag width: {raw!r}",
                stage="load",
                sort_key=f"load|{lineno:06d}|{E_FIELD_TRUNCATED}",
            )
            continue
        tag = line[:TAG_LEN]
        kind = tag[0]
        if kind not in WIDTHS or tag[1] != "|":
            failures.fail(
                E_FIELD_TRUNCATED,
                f"line {lineno} unknown/misaligned tag {tag!r}",
                stage="load",
                sort_key=f"load|{lineno:06d}|{E_FIELD_TRUNCATED}",
            )
            continue
        if len(line) != LINE_LEN[kind]:
            failures.fail(
                E_FIELD_TRUNCATED,
                f"line {lineno} width {len(line)} != {LINE_LEN[kind]} for record {kind}",
                stage="load",
                sort_key=f"load|{lineno:06d}|{E_FIELD_TRUNCATED}",
            )
            continue
        try:
            tokens = _slice_fields(line, WIDTHS[kind])
            vals = [_parse_int(tok) for tok in tokens]
        except ValueError as exc:
            failures.fail(
                E_FIELD_TRUNCATED,
                f"line {lineno} field parse error: {exc}",
                stage="load",
                sort_key=f"load|{lineno:06d}|{E_FIELD_TRUNCATED}",
            )
            continue

        if kind == "F":
            view, fid, x, y, lid = vals
            features.append(
                Feature(view=view, feature_id=fid, x=float(x), y=float(y),
                        landmark_id=lid, precision="fixed:v1")
            )
        elif kind == "M":
            va, fa, vb, fb, score, metric_code = vals
            matches.append(
                Match(view_a=va, feature_a=fa, view_b=vb, feature_b=fb,
                      score_permille=float(score), metric="", metric_code=metric_code,
                      precision="fixed:v1")
            )
        else:
            vx, vy, vz, rx, ry, rz, lid = vals
            voxels.append(
                VoxelPoint(vx=vx, vy=vy, vz=vz,
                           x=fixed_to_float(rx, fixed_bits or 8),
                           y=fixed_to_float(ry, fixed_bits or 8),
                           z=fixed_to_float(rz, fixed_bits or 8),
                           landmark_id=lid, precision="fixed:v1")
            )

    if width is None or height is None or voxel_fixed is None or fixed_bits is None:
        failures.fail(E_PARSE, "missing required #SIZE/#VOX header", stage="load")
        ir = ReconstructionIR((0, 0), 0.0, fixed_bits or 8, [], [], [])
    else:
        voxel_size = fixed_to_float(voxel_fixed, fixed_bits)
        ir = ReconstructionIR(
            (width, height), voxel_size, fixed_bits, features, matches, voxels,
            source_version="v1",
        )
    ir.failures = failures
    return ir


def _signed(value: int, width: int) -> str:
    return f"{int(value):+0{width}d}"


def serialize_v1(ir: ReconstructionIR) -> str:
    """把统一 IR 序列化为定宽 v1 文本。"""
    width, height = ir.image_size
    bits = ir.fixed_point_bits
    lines = [
        "#MVRB1",
        f"#SIZE W={int(width):04d} H={int(height):04d}",
        f"#VOX {_signed(float_to_fixed(ir.voxel_size, bits), 10)} #FPB={bits:04d}",
    ]
    for f in ir.features:
        lines.append(
            "F|"
            + f"{f.view:04d}{f.feature_id:06d}{round(f.x):06d}{round(f.y):06d}{f.landmark_id:06d}"
        )
    for m in ir.matches:
        lines.append(
            "M|"
            + f"{m.view_a:04d}{m.feature_a:06d}{m.view_b:04d}{m.feature_b:06d}"
            + f"{round(m.score_permille):04d}{m.metric_code:02d}"
        )
    for p in ir.voxel_points:
        lines.append(
            "V|"
            + _signed(p.vx, 7) + _signed(p.vy, 7) + _signed(p.vz, 7)
            + _signed(float_to_fixed(p.x, bits), 10)
            + _signed(float_to_fixed(p.y, bits), 10)
            + _signed(float_to_fixed(p.z, bits), 10)
            + f"{p.landmark_id:06d}"
        )
    return "\n".join(lines) + "\n"


def write_v1(path: str, ir: ReconstructionIR) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(serialize_v1(ir))
    os.replace(tmp, path)


def read_v1(path: str) -> ReconstructionIR:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_v1(fh.read())
