"""
TWT schedules for the HaLow AP, and the limits of the MM6108 hardware.

A schedule is one line per station, written on the AP into
/sys/kernel/debug/ieee80211/<phy>/morse/twt_ap_update:

    <station MAC> <wake time us> <wake interval us> <wake duration us>

The limits below were measured on the lab's HaLow TWT bench:

- Duration is a one-byte field counted in 256 us units, so the longest window is
  255 x 256 = 65,280 us. Longer requests are clipped without any error.
- At 4.1 ms and below no traffic got through a window; 6.1 ms was the first duration
  that carried any.
- Wake intervals between 250 ms and 2.5 s were tested (worst error 1.2 %).
- The interval is stored as mantissa x 2^exponent (16-bit mantissa), so it is rounded
  to the nearest value of that form, e.g. 10,000,000 us is stored as 9,999,872 us.
"""

import math
import re
from dataclasses import dataclass, replace

DURATION_UNIT_US = 256
MAX_DURATION_US = 255 * DURATION_UNIT_US          # 65,280 us
MIN_USEFUL_DURATION_US = 24 * DURATION_UNIT_US    # 6,144 us: shortest window that carried traffic
TESTED_INTERVAL_US = (250_000, 2_500_000)
MAX_MANTISSA = 0xFFFF
MAX_EXPONENT = 31

_MAC_RE = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")


class ScheduleError(ValueError):
    """A schedule the AP would refuse, or that would not do what was asked."""


@dataclass(frozen=True)
class StationSchedule:
    mac: str
    wake_us: int
    interval_us: int
    duration_us: int
    name: str = ""

    def to_line(self):
        """The text written into twt_ap_update."""
        return f"{self.mac} {self.wake_us} {self.interval_us} {self.duration_us}"

    @property
    def duty_cycle(self):
        return self.duration_us / self.interval_us if self.interval_us else 0.0

    def label(self):
        return self.name or self.mac


def normalize_mac(mac):
    mac = mac.strip().lower().replace("-", ":")
    if not _MAC_RE.match(mac):
        raise ScheduleError(f"not a MAC address: {mac!r}")
    return mac


def quantize_duration_us(duration_us):
    """Round up to whole 256 us units and clip at 65,280 us (what the hardware can express)."""
    if duration_us <= 0:
        raise ScheduleError("duration must be positive")
    units = math.ceil(duration_us / DURATION_UNIT_US)
    return min(units, 255) * DURATION_UNIT_US


def quantize_interval_us(interval_us):
    """Nearest value of the form mantissa x 2^exponent, with a 16-bit mantissa."""
    if interval_us <= 0:
        raise ScheduleError("wake interval must be positive")
    exponent = 0
    while interval_us / (1 << exponent) > MAX_MANTISSA:
        exponent += 1
    mantissa = round(interval_us / (1 << exponent))
    if mantissa > MAX_MANTISSA:          # rounding pushed it over: use the next exponent
        exponent += 1
        mantissa = round(interval_us / (1 << exponent))
    if exponent > MAX_EXPONENT:
        raise ScheduleError(f"wake interval {interval_us} us is too long to encode")
    return mantissa << exponent


def quantize(schedule):
    """Return (the schedule the AP will actually hold, notes about what changed)."""
    notes = []
    interval = quantize_interval_us(schedule.interval_us)
    duration = quantize_duration_us(schedule.duration_us)
    if interval != schedule.interval_us:
        notes.append(f"{schedule.label()}: interval {schedule.interval_us} us -> {interval} us "
                     "(nearest encodable value)")
    if schedule.duration_us > MAX_DURATION_US:
        notes.append(f"{schedule.label()}: duration {schedule.duration_us} us clipped to "
                     f"{MAX_DURATION_US} us (hardware maximum)")
    elif duration != schedule.duration_us:
        notes.append(f"{schedule.label()}: duration {schedule.duration_us} us -> {duration} us "
                     "(whole 256 us units)")
    return replace(schedule, mac=normalize_mac(schedule.mac), interval_us=interval,
                   duration_us=duration), notes


def _windows_overlap(a, b):
    """Do two windows with the same interval overlap anywhere in the cycle?"""
    period = a.interval_us
    sa, sb = a.wake_us % period, b.wake_us % period
    for x, y, dy in ((sa, sb, b.duration_us), (sb, sa, a.duration_us)):
        # does y's window [y, y+dy) (taken around the cycle) contain the start x?
        if (x - y) % period < dy:
            return True
    return False


def validate(schedules):
    """
    Check a full schedule before it is sent.

    Returns (errors, warnings). Errors are things the AP would refuse or that are plainly
    wrong; warnings are legal but risky (untested range, windows too short to carry traffic,
    stations contending in overlapping windows).
    """
    errors, warnings = [], []
    seen = {}
    for s in schedules:
        try:
            mac = normalize_mac(s.mac)
        except ScheduleError as e:
            errors.append(str(e))
            continue
        if mac in seen:
            errors.append(f"{s.label()}: station appears twice ({mac})")
        seen[mac] = s
        if s.duration_us > s.interval_us:
            errors.append(f"{s.label()}: window {s.duration_us} us is longer than its "
                          f"interval {s.interval_us} us (the AP refuses this)")
        if not 0 <= s.wake_us < s.interval_us:
            errors.append(f"{s.label()}: wake time {s.wake_us} us must be in "
                          f"[0, interval {s.interval_us} us)")
        elif s.wake_us + s.duration_us > s.interval_us:
            warnings.append(f"{s.label()}: window wraps past the end of the interval; "
                            "the AP may refuse it")
        lo, hi = TESTED_INTERVAL_US
        if not lo <= s.interval_us <= hi:
            warnings.append(f"{s.label()}: interval {s.interval_us} us is outside the tested "
                            f"range {lo}-{hi} us")
        if s.duration_us < MIN_USEFUL_DURATION_US:
            warnings.append(f"{s.label()}: window {s.duration_us} us is below "
                            f"{MIN_USEFUL_DURATION_US} us; on the bench such windows carried "
                            "no traffic")

    items = list(seen.values())
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.interval_us != b.interval_us:
                warnings.append(f"{a.label()} and {b.label()} use different intervals; overlap "
                                "between them is not checked")
            elif a.wake_us == b.wake_us and a.duration_us == b.duration_us:
                continue                               # deliberately sharing one window
            elif _windows_overlap(a, b):
                warnings.append(f"{a.label()} and {b.label()}: windows partly overlap, so they "
                                "will contend")
    return errors, warnings
