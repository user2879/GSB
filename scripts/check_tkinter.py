from __future__ import annotations

import sys


def main() -> int:
    print(f"Python {sys.version}")
    try:
        import tkinter
    except Exception as exc:
        print("tkinter is not available in this interpreter:")
        print(f"  {type(exc).__name__}: {exc}")
        print("On Windows, reinstall/modify Python and enable tcl/tk, or run: python -m tkinter")
        return 1
    print(f"tkinter import OK; TkVersion={tkinter.TkVersion}")
    try:
        root = tkinter.Tk()
        root.title("Workbench Tk check")
        label = tkinter.Label(root, text="tkinter is available", padx=24, pady=16)
        label.pack()
        root.after(300, root.destroy)
        root.mainloop()
        print("Tk window smoke test OK")
    except Exception as exc:
        print(f"tkinter imports but no display/session could open Tk: {type(exc).__name__}: {exc}")
        print("On a Windows desktop this normally opens the small widget window.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
