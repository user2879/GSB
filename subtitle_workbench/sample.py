"""Deterministic synthetic sample generation.

A single :class:`numpy.random.Generator` created from a fixed seed drives every
random choice (cut jitter, frame jitter, text/speaker variants). The four
required situations are all present:

* 2-4 cues overlapping on the same frames (including a 4-way cluster);
* cues crossing virtual shot boundaries;
* accessibility-description (AD) cues, including a 3-way AD cluster that
  triggers the yielding rule;
* long latin/CJK texts that force font-size downgrade and wrapping.

Running twice with the same seed produces byte-identical output.
"""
from __future__ import annotations

import numpy as np

from .models import (FPS, FRAME_HEIGHT, FRAME_WIDTH, SampleProject, Shot,
                     SubtitleItem, Timeline, canonical_bytes)
from .layout import AD_MIN_DURATION

DEFAULT_SEED = 20261119
TOTAL_FRAMES = 90 * FPS  # 2700

SPEAKERS = ("ANNA", "BEN", "CHEN", "NARRATOR")
LANG_POOL = ("zh", "zh", "en")

DIALOGUE_ZH = [
    "我们需要在镜头切换前确认灯光与麦克风的位置。",
    "安全区边距会随轨道编号变化，请逐帧核对时间轴。",
    "这一段对白横跨两个镜头，排版引擎必须保持轨道稳定。",
    "轨道避让的顺序由阅读优先级、起始帧和条目编号共同决定。",
]
DIALOGUE_EN = [
    "The layout engine assigns lanes by reading priority before timeline order.",
    "Every committed snapshot is immutable and carries a content hash.",
    "Word boundary wrapping keeps each rendered line inside the safe width.",
    "A rejected drag never mutates the last committed layout state.",
]
AD_ZH = [
    "【音效描述】窗外传来低沉的雷声，房间灯光微微闪烁。",
    "【动作描述】她转身走向控制台，手指停在红色按钮上方。",
    "【环境描述】街道尽头的霓虹灯依次亮起，雨水沿玻璃滑落。",
]
AD_EN = [
    "[Audio description] A slow dolly move reveals the empty harbor at dawn.",
]
LONG_ZH = ("无障碍排版要求长文本在词边界优先折行，当默认字号无法在限定行数内"
           "容纳全部内容时，引擎按声明的字号档位逐级降档，并保证任何单行都不"
           "会超出安全区宽度，整个过程必须可复算。")
LONG_EN = ("This deliberately long English caption is used to prove that the "
           "wrapping engine prefers word boundaries, downgrades the font size "
           "tier when the line budget cannot be met, and never lets a single "
           "rendered line escape the declared safe area margins.")


def _build_shots(rng: np.random.Generator) -> tuple[Shot, ...]:
    # Base cuts every 10s with seeded +/-15 frame jitter; minimum gap 120f.
    cuts = [0]
    for base in range(300, TOTAL_FRAMES, 300):
        jitter = int(rng.integers(-15, 16))
        cut = max(cuts[-1] + 120, min(TOTAL_FRAMES - 1, base + jitter))
        cuts.append(cut)
    cuts.append(TOTAL_FRAMES)
    shots = tuple(
        Shot(shot_id=f"SH{index:02d}", start_frame=cuts[index],
             end_frame=cuts[index + 1])
        for index in range(len(cuts) - 1)
    )
    return shots


def _cue(counter: list[int], start: int, end: int, text: str, *,
         is_ad: bool, priority: int, rng: np.random.Generator) -> SubtitleItem:
    item_id = f"SUB-{counter[0]:03d}"
    counter[0] += 1
    language = LANG_POOL[int(rng.integers(0, len(LANG_POOL)))]
    speaker = ("AUDIO_DESC" if is_ad
               else SPEAKERS[int(rng.integers(0, len(SPEAKERS) - 1))])
    return SubtitleItem(item_id, start, end, text, speaker, language,
                        priority, is_ad)


def _pick(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


def generate_sample(seed: int = DEFAULT_SEED) -> SampleProject:
    rng = np.random.default_rng(seed)
    shots = _build_shots(rng)
    counter = [1]
    items: list[SubtitleItem] = []

    def jitter(span: int = 4) -> int:
        return int(rng.integers(-span, span + 1))

    def add(start: int, end: int, text: str, *, is_ad: bool,
            priority: int) -> None:
        # Accessibility descriptions obey the longer minimum display rule.
        if is_ad and end - start < AD_MIN_DURATION:
            end = min(TOTAL_FRAMES, start + AD_MIN_DURATION)
        items.append(_cue(counter, start, end, text, is_ad=is_ad,
                          priority=priority, rng=rng))

    # --- opening single cue ---------------------------------------------
    add(30 + jitter(), 120 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=2)
    # --- 2-way overlap ---------------------------------------------------
    base = 200
    add(base + jitter(), base + 90 + jitter(), _pick(rng, DIALOGUE_EN),
        is_ad=False, priority=2)
    add(base + 45 + jitter(), base + 140 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=1)
    # --- cue crossing the first shot cut (cut near frame 300) ------------
    add(260 + jitter(), 360 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=3)
    # --- first AD cue crossing a cut (~600) ------------------------------
    add(560 + jitter(), 660 + jitter(), _pick(rng, AD_ZH),
        is_ad=True, priority=2)
    # --- 4-way dialogue cluster (frames ~720-840) ------------------------
    cluster = 720
    add(cluster + jitter(), cluster + 120 + jitter(),
        _pick(rng, DIALOGUE_ZH), is_ad=False, priority=1)
    add(cluster + 20 + jitter(), cluster + 100 + jitter(),
        _pick(rng, DIALOGUE_EN), is_ad=False, priority=2)
    add(cluster + 50 + jitter(), cluster + 130 + jitter(),
        _pick(rng, DIALOGUE_ZH), is_ad=False, priority=2)
    add(cluster + 70 + jitter(), cluster + 110 + jitter(),
        _pick(rng, DIALOGUE_EN), is_ad=False, priority=3)
    # --- long Chinese text: wrapping + font downgrade --------------------
    add(900 + jitter(), 1040 + jitter(), LONG_ZH, is_ad=False, priority=2)
    # --- long English text ----------------------------------------------
    add(1080 + jitter(), 1230 + jitter(), LONG_EN, is_ad=False, priority=2)
    # --- AD + dialogue conflict, dialogue priority 1 (AD yields) --------
    add(1300 + jitter(), 1400 + jitter(), _pick(rng, AD_ZH),
        is_ad=True, priority=2)
    add(1320 + jitter(), 1390 + jitter(), _pick(rng, DIALOGUE_EN),
        is_ad=False, priority=1)
    # --- 3-way AD cluster (only 2 declared AD lanes -> one yields) ------
    ad_cluster = 1500
    add(ad_cluster + jitter(), ad_cluster + 110 + jitter(),
        _pick(rng, AD_ZH), is_ad=True, priority=2)
    add(ad_cluster + 30 + jitter(), ad_cluster + 120 + jitter(),
        _pick(rng, AD_EN), is_ad=True, priority=1)
    add(ad_cluster + 60 + jitter(), ad_cluster + 140 + jitter(),
        _pick(rng, AD_ZH), is_ad=True, priority=3)
    # --- cross-cut dialogue near frame 1800 ------------------------------
    add(1760 + jitter(), 1860 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=2)
    # --- mixed 3-way overlap with an AD cue ------------------------------
    add(1960 + jitter(), 2080 + jitter(), _pick(rng, DIALOGUE_EN),
        is_ad=False, priority=1)
    add(2000 + jitter(), 2090 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=3)
    add(2020 + jitter(), 2070 + jitter(), _pick(rng, AD_ZH),
        is_ad=True, priority=2)
    # --- second long text after another cut (~2100) ----------------------
    add(2140 + jitter(), 2290 + jitter(), LONG_EN, is_ad=False, priority=2)
    # --- closing cues ----------------------------------------------------
    add(2380 + jitter(), 2470 + jitter(), _pick(rng, DIALOGUE_ZH),
        is_ad=False, priority=2)
    add(2520 + jitter(), 2620 + jitter(), _pick(rng, AD_EN),
        is_ad=True, priority=2)
    add(2640 + jitter(), 2690 + jitter(), _pick(rng, DIALOGUE_EN),
        is_ad=False, priority=1)

    items_tuple = tuple(sorted(items, key=lambda it: it.item_id))

    manifest = tuple(
        {
            "item_id": item.item_id,
            "start_frame": item.start_frame,
            "end_frame": item.end_frame,
            "is_ad": item.is_ad,
            "priority": item.priority,
            "crosses_cut": any(item.start_frame < shot.start_frame <
                               item.end_frame for shot in shots),
            "overlaps_with": sorted(
                other.item_id for other in items_tuple
                if other.item_id != item.item_id
                and item.start_frame < other.end_frame
                and other.start_frame < item.end_frame
            ),
        }
        for item in items_tuple
    )

    timeline = Timeline(FPS, FRAME_WIDTH, FRAME_HEIGHT, TOTAL_FRAMES, shots)
    return SampleProject(timeline, items_tuple, seed, manifest)


def write_sample(project: SampleProject, path: str | os.PathLike) -> None:
    """Write the sample project as deterministic canonical JSON bytes."""
    with open(path, "wb") as handle:
        handle.write(canonical_bytes(project.to_dict()))
