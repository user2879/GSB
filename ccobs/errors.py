"""稳定可判的业务码与业务异常。

业务码:
  E_CONFIG    配置或脚本校验失败(整体拒绝, 指明首个非法条目)
  E_SCRIPT    脚本事件序号缺口(报告缺口起点与缺失条数)
  E_CWND      重复确认异常聚集超限(报告超限次数与所用上限)
  E_SNAPSHOT  快照版本或摘要一致性问题(整体替换)
  E_REPLAY    恢复后重放校验不一致(定位首个不一致事件序号与观测项)
"""

E_CONFIG = "E_CONFIG"
E_SCRIPT = "E_SCRIPT"
E_CWND = "E_CWND"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"

ALL_CODES = (E_CONFIG, E_SCRIPT, E_CWND, E_SNAPSHOT, E_REPLAY)


class BizError(Exception):
    """携带稳定业务码与结构化细节的异常。

    details 渲染为按键排序的 k=v 串, 保证同一问题在任意执行次数下
    产出逐字节一致的业务码行。
    """

    def __init__(self, code, message, **details):
        if code not in ALL_CODES:
            raise ValueError("unknown business code: %r" % (code,))
        self.code = code
        self.message = message
        self.details = dict(details)
        super().__init__(self.render())

    def render(self):
        parts = ["%s|%s" % (self.code, self.message)]
        for key in sorted(self.details):
            parts.append("%s=%s" % (key, self.details[key]))
        return "|".join(parts)

    def __str__(self):
        return self.render()
