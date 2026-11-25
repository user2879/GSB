"""可选 Tk 监控面板；系统缺少 Tk 运行库时降级为文本控制台。"""


class TextDashboard:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)
        print(f"[event] {event.get('event')} "
              f"{event.get('batch_id', '')} {event.get('worker_id', '')}")


def make_dashboard(prefer_tk=True):
    if not prefer_tk:
        return TextDashboard(), "text"
    try:
        import tkinter
    except Exception as exc:
        print(f"[dashboard] tkinter 不可用（{exc}），降级为文本控制台；"
              "可执行 sudo apt install python3-tk 启用图形面板。")
        return TextDashboard(), "text"

    class TkDashboard:
        def __init__(self):
            self.root = tkinter.Tk()
            self.root.title("批推理租约监控")
            self.text = tkinter.Text(self.root, width=96, height=24)
            self.text.pack()

        def emit(self, event):
            self.text.insert("end", str(event) + "\n")
            self.text.see("end")
            self.root.update_idletasks()

        def close(self):
            self.root.destroy()

    return TkDashboard(), "tk"
