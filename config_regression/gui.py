"""Optional Tkinter report viewer.

Tkinter is imported lazily so the headless pipeline and test suite work on
machines without the Tk system libraries.  On a standard Windows Python
install Tkinter ships with the interpreter and ``python -m config_regression.cli gui``
opens this viewer directly.
"""

from __future__ import annotations

import json
from tkinter import Tk, Text, Button, filedialog, Scrollbar, END


def launch_viewer() -> None:
    root = Tk()
    root.title("Config Regression Report Viewer")
    root.geometry("960x640")
    text = Text(root, wrap="none")
    scroll_y = Scrollbar(root, command=text.yview)
    scroll_x = Scrollbar(root, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    def open_report() -> None:
        path = filedialog.askopenfilename(
            title="Open regression report",
            filetypes=(("JSON files", "*.json"), ("all files", "*.*")))
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        text.delete("1.0", END)
        text.insert(END, json.dumps(payload, indent=2, sort_keys=True,
                                    ensure_ascii=False))

    button = Button(root, text="Open report JSON", command=open_report)
    button.pack(anchor="w")
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    text.pack(fill="both", expand=True)
    root.mainloop()
