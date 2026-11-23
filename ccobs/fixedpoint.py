"""拥塞窗口、慢启动阈值与平滑 RTT 的整数定点累积。

定点口径: 单位 U = 10 ** scale_factor, 所有窗口量与 RTT 量均以
"真实值 * U" 的整数保存。每次截断除法向对应观测项的量化误差上界
累加 1 个定点单位, 使每个观测项都能登记其误差上界。

观测项: CWND, SSTHRESH, FLIGHT, SRTT, RTO (RTTVAR 并入 RTO 的推导)。
"""
from dataclasses import dataclass, field, replace

ITEM_CWND = "CWND"
ITEM_SSTHRESH = "SSTHRESH"
ITEM_FLIGHT = "FLIGHT"
ITEM_SRTT = "SRTT"
ITEM_RTO = "RTO"

ALL_ITEMS = (ITEM_CWND, ITEM_SSTHRESH, ITEM_FLIGHT, ITEM_SRTT, ITEM_RTO)

# 各观测项累积所依赖的事件类别
ITEM_EVENT_KINDS = {
    ITEM_CWND: ("ack", "dup_ack", "timeout"),
    ITEM_SSTHRESH: ("ack", "dup_ack", "timeout"),
    ITEM_FLIGHT: ("send", "ack", "timeout"),
    ITEM_SRTT: ("ack",),
    ITEM_RTO: ("ack", "timeout"),
}

SLOW_START = "slow_start"
CONGESTION_AVOIDANCE = "congestion_avoidance"


@dataclass(frozen=True)
class CongestionState:
    """一次事件推进后的定点状态快照(不可变)。"""
    cwnd: int              # 定点字节
    ssthresh: int          # 定点字节
    flight: int            # 已发送未确认(字节, 精确整数)
    srtt: int              # 定点毫秒
    rttvar: int            # 定点毫秒
    mode: str = SLOW_START
    dup_run: int = 0       # 当前连续重复确认计数
    err_cwnd: int = 0      # 量化误差上界(定点单位)
    err_ssthresh: int = 0
    err_srtt: int = 0
    err_rttvar: int = 0

    @property
    def rto(self):
        return self.srtt + 4 * self.rttvar

    @property
    def err_rto(self):
        return self.err_srtt + 4 * self.err_rttvar


def initial_state(config):
    unit = config.unit
    return CongestionState(
        cwnd=config.mss * unit,
        ssthresh=config.cwnd_max * unit,
        flight=0,
        srtt=0,
        rttvar=0,
    )


def _halve(value):
    """定点减半, 截断误差 <= 1 个定点单位。"""
    return value // 2, 1


def apply_send(state, config):
    return replace(state, flight=state.flight + config.mss)


def apply_ack(state, config, rtt_ms):
    """正常确认: 慢启动每确认窗口增一(MSS), 拥塞避免每往返窗口增一。"""
    unit = config.unit
    mss = config.mss * unit
    err_cwnd = state.err_cwnd
    err_ssthresh = state.err_ssthresh
    err_srtt = state.err_srtt
    err_rttvar = state.err_rttvar

    cwnd = state.cwnd
    ssthresh = state.ssthresh
    mode = state.mode
    if state.dup_run >= 3:
        # 快速重传后的确认: 窗口回落到阈值
        cwnd = ssthresh
        mode = CONGESTION_AVOIDANCE
    elif mode == SLOW_START:
        cwnd = min(cwnd + mss, config.cwnd_max * unit)
        if cwnd >= ssthresh:
            mode = CONGESTION_AVOIDANCE
    else:
        inc = (mss * mss) // cwnd if cwnd > 0 else mss
        if inc == 0:
            inc = 1
        err_cwnd += 1  # 截断误差上界累加
        cwnd = min(cwnd + inc, config.cwnd_max * unit)

    # RTT 定点累积(RFC 6298 风格, 整数截断)
    sample = rtt_ms * unit
    if state.srtt == 0 and state.rttvar == 0:
        srtt = sample
        rttvar = sample // 2
        err_rttvar += 1
    else:
        delta = abs(state.srtt - sample)
        rttvar = (3 * state.rttvar + delta) // 4
        srtt = (7 * state.srtt + sample) // 8
        err_rttvar += 1
        err_srtt += 1

    flight = max(0, state.flight - config.mss)
    return CongestionState(
        cwnd=cwnd, ssthresh=ssthresh, flight=flight,
        srtt=srtt, rttvar=rttvar, mode=mode, dup_run=0,
        err_cwnd=err_cwnd, err_ssthresh=err_ssthresh,
        err_srtt=err_srtt, err_rttvar=err_rttvar,
    )


def apply_dup_ack(state, config):
    """重复确认: 第三次触发快速重传并调整阈值。"""
    unit = config.unit
    mss = config.mss * unit
    dup_run = state.dup_run + 1
    cwnd = state.cwnd
    ssthresh = state.ssthresh
    err_cwnd = state.err_cwnd
    err_ssthresh = state.err_ssthresh
    if dup_run == 3:
        ssthresh, err = _halve(cwnd)
        ssthresh = max(ssthresh, 2 * mss)
        err_ssthresh += err
        cwnd = min(ssthresh + 3 * mss, config.cwnd_max * unit)
    elif dup_run > 3:
        cwnd = min(cwnd + mss, config.cwnd_max * unit)
    return replace(state, cwnd=cwnd, ssthresh=ssthresh, dup_run=dup_run,
                   err_cwnd=err_cwnd, err_ssthresh=err_ssthresh)


def apply_timeout(state, config):
    """超时: 阈值减半, 窗口回到慢启动。"""
    unit = config.unit
    mss = config.mss * unit
    ssthresh, err = _halve(state.cwnd)
    ssthresh = max(ssthresh, 2 * mss)
    return CongestionState(
        cwnd=mss, ssthresh=ssthresh, flight=0,
        srtt=state.srtt, rttvar=state.rttvar, mode=SLOW_START, dup_run=0,
        err_cwnd=state.err_cwnd,
        err_ssthresh=state.err_ssthresh + err,
        err_srtt=state.err_srtt, err_rttvar=state.err_rttvar,
    )


def item_value(state, item):
    if item == ITEM_CWND:
        return state.cwnd
    if item == ITEM_SSTHRESH:
        return state.ssthresh
    if item == ITEM_FLIGHT:
        return state.flight
    if item == ITEM_SRTT:
        return state.srtt
    if item == ITEM_RTO:
        return state.rto
    raise ValueError("unknown item %r" % (item,))


def item_error_bound(state, item):
    if item == ITEM_CWND:
        return state.err_cwnd
    if item == ITEM_SSTHRESH:
        return state.err_ssthresh
    if item == ITEM_FLIGHT:
        return 0
    if item == ITEM_SRTT:
        return state.err_srtt
    if item == ITEM_RTO:
        return state.err_rto
    raise ValueError("unknown item %r" % (item,))
