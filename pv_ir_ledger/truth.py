from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import is_int, load_json
from .errors import ErrorCode, make_issue


def grid_intervals(width: int, height: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    intervals: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        y0 = (row * height) // rows
        y1 = ((row + 1) * height) // rows
        for col in range(cols):
            x0 = (col * width) // cols
            x1 = ((col + 1) * width) // cols
            intervals.append((x0, y0, x1, y1))
    return intervals


def validate_truth(
    raw: Any, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    truth_by_image: dict[str, dict[str, Any]] = {}
    width = config.get("fixed_width", 320)
    height = config.get("fixed_height", 240)

    if not isinstance(raw, dict):
        return {}, [make_issue(ErrorCode.TRUTH_INVALID, "truth document must be a JSON object")]
    entries = raw.get("arrays", raw.get("images"))
    if not isinstance(entries, list):
        return {}, [make_issue(ErrorCode.TRUTH_INVALID, "truth document must contain arrays or images list")]

    for entry_index, entry in enumerate(entries):
        entry_issues_before = len(issues)
        if not isinstance(entry, dict):
            issues.append(
                make_issue(ErrorCode.TRUTH_INVALID, "truth array entry must be an object", index=entry_index)
            )
            continue
        image = entry.get("image")
        if not isinstance(image, str) or not image:
            issues.append(
                make_issue(ErrorCode.TRUTH_INVALID, "truth entry image must be a non-empty string", index=entry_index)
            )
            image = f"<entry-{entry_index}>"

        rows = entry.get("grid_rows")
        cols = entry.get("grid_cols")
        if not is_int(rows) or rows <= 0:
            issues.append(make_issue(ErrorCode.TRUTH_INVALID, "truth grid_rows must be positive", image=image, value=rows))
            rows = None
        if not is_int(cols) or cols <= 0:
            issues.append(make_issue(ErrorCode.TRUTH_INVALID, "truth grid_cols must be positive", image=image, value=cols))
            cols = None

        components = entry.get("components")
        if not isinstance(components, list):
            issues.append(make_issue(ErrorCode.TRUTH_INVALID, "truth components must be a list", image=image))
            continue
        if rows and cols and len(components) != rows * cols:
            issues.append(
                make_issue(
                    ErrorCode.TRUTH_INVALID,
                    "truth component count does not equal grid rows times columns",
                    image=image,
                    component_count=len(components),
                    expected=rows * cols,
                )
            )

        intervals = grid_intervals(width, height, rows, cols) if rows and cols else []
        previous_id: int | None = None
        normalized_components: list[dict[str, Any]] = []
        for component_index, component in enumerate(components):
            if not isinstance(component, dict):
                issues.append(
                    make_issue(
                        ErrorCode.TRUTH_INVALID,
                        "truth component must be an object",
                        image=image,
                        index=component_index,
                    )
                )
                continue
            component_id = component.get("component_id")
            if not is_int(component_id) or component_id <= 0:
                issues.append(
                    make_issue(
                        ErrorCode.TRUTH_INVALID,
                        "component_id must be a positive integer",
                        image=image,
                        index=component_index,
                        value=component_id,
                    )
                )
            elif previous_id is not None and component_id <= previous_id:
                issues.append(
                    make_issue(
                        ErrorCode.TRUTH_INVALID,
                        "component_id values must be strictly increasing without duplicates",
                        image=image,
                        component_id=component_id,
                        previous_component_id=previous_id,
                    )
                )
            else:
                previous_id = component_id

            bbox = component.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(is_int(v) for v in bbox):
                issues.append(
                    make_issue(
                        ErrorCode.TRUTH_INVALID,
                        "bbox must be four integers [x1,y1,x2,y2]",
                        image=image,
                        component_id=component_id,
                        value=bbox,
                    )
                )
                continue
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                issues.append(
                    make_issue(
                        ErrorCode.TRUTH_INVALID,
                        "bbox is outside the fixed image or has reversed endpoints",
                        image=image,
                        component_id=component_id,
                        bbox=bbox,
                        image_width=width,
                        image_height=height,
                    )
                )
            elif component_index < len(intervals):
                cx0, cy0, cx1, cy1 = intervals[component_index]
                if not (cx0 <= x1 and x2 <= cx1 and cy0 <= y1 and y2 <= cy1):
                    issues.append(
                        make_issue(
                            ErrorCode.TRUTH_INVALID,
                            "bbox crosses its declared grid cell",
                            reason="BBOX_OUTSIDE_GRID",
                            image=image,
                            component_id=component_id,
                            bbox=bbox,
                            grid_cell=[cx0, cy0, cx1, cy1],
                        )
                    )
            normalized_components.append({**component, "bbox": bbox})

        if len(issues) == entry_issues_before:
            truth_by_image[image] = {
                "image": image,
                "grid_rows": rows,
                "grid_cols": cols,
                "components": normalized_components,
            }
    return truth_by_image, issues


def load_truth(
    path: str | Path, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw, issues = load_json(path)
    if issues:
        return {}, issues
    return validate_truth(raw, config)
