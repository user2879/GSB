"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import canonical_json, run_demo
from .model import loads_glyphs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glyphforge", description="Glyph outline and atlas workbench"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo", help="run the reproducible end-to-end ledger demo"
    )
    demo_parser.add_argument("--out", default="artifacts/demo")
    demo_parser.add_argument("--threads", type=int, default=3)
    demo_parser.add_argument("--no-repeat-check", action="store_true")

    gui_parser = subparsers.add_parser("gui", help="open the PyQt5 diagnostic editor")
    gui_parser.add_argument("--samples", help="optional deterministic glyph JSON")

    validate_parser = subparsers.add_parser(
        "validate", help="validate and canonicalize glyph JSON"
    )
    validate_parser.add_argument("path")

    args = parser.parse_args(argv)
    if args.command == "demo":
        summary = run_demo(
            args.out, threads=args.threads, verify_repeat=not args.no_repeat_check
        )
        print(canonical_json(summary))
        return 0
    if args.command == "gui":
        from .application import GlyphApplication
        from .gui import launch_gui

        if args.samples:
            document = loads_glyphs(Path(args.samples).read_bytes())
            glyphs = document.by_id
            application = GlyphApplication(
                glyphs,
                active_sizes={glyph_id: (12, 16, 24, 32, 48) for glyph_id in glyphs},
            )
            application.render_all(step="gui_initial_render")
            return launch_gui(application)
        return launch_gui()
    if args.command == "validate":
        data = Path(args.path).read_bytes()
        document = loads_glyphs(data)
        summary = {
            "valid": True,
            "format": document.format_version,
            "glyphs": [glyph.glyph_id for glyph in document.glyphs],
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
