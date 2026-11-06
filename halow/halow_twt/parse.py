"""Parsers for what the AP prints: its TWT agreement table and `iw dev wlan0 station dump`."""

import re
from dataclasses import dataclass

_REQUESTER = re.compile(r"Requester:\s*([0-9a-fA-F:]{17})")
_FLOW0 = re.compile(r"Flow ID:\s*0,\s*Wake Interval:\s*(\d+)\s*us,\s*Wake Time:\s*(\d+)\s*us,"
                    r"\s*Wake Duration:\s*(\d+)\s*us,\s*State:?\s*(\d+)")
_STATION = re.compile(r"^Station\s+([0-9a-fA-F:]{17})")
_FIELD = re.compile(r"^\s+([a-zA-Z][a-zA-Z /\[\]-]*?):\s*(.*?)\s*$")
_INT = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Agreement:
    """Flow-0 TWT agreement as the AP stores it."""
    wake_us: int
    interval_us: int
    duration_us: int
    state: int


def parse_agreements(text):
    """twt_sta_agreements -> {mac: Agreement} for every station whose flow 0 is set up."""
    out, mac = {}, None
    for line in text.replace("\r", "").splitlines():
        m = _REQUESTER.search(line)
        if m:
            mac = m.group(1).lower()
            continue
        m = _FLOW0.search(line)
        if m and mac:
            interval, wake, duration, state = (int(x) for x in m.groups())
            if interval > 0:
                out[mac] = Agreement(wake_us=wake, interval_us=interval,
                                     duration_us=duration, state=state)
    return out


# station-dump fields kept, and the name each is stored under
_NUMERIC = {
    "inactive time": "inactive_ms",
    "rx bytes": "rx_bytes",
    "rx packets": "rx_packets",
    "tx bytes": "tx_bytes",
    "tx packets": "tx_packets",
    "tx retries": "tx_retries",
    "tx failed": "tx_failed",
    "rx drop misc": "rx_drop_misc",
    "signal": "signal_dbm",
    "signal avg": "signal_avg_dbm",
}


def parse_station_dump(text):
    """`iw dev wlan0 station dump` -> {mac: {field: value}} with numeric fields as ints."""
    out, cur = {}, None
    for line in text.replace("\r", "").splitlines():
        m = _STATION.match(line)
        if m:
            cur = out.setdefault(m.group(1).lower(), {})
            continue
        if cur is None:
            continue
        m = _FIELD.match(line)
        if not m:
            continue
        key, value = m.group(1).strip().lower(), m.group(2)
        if key in _NUMERIC:
            n = _INT.search(value)
            if n:
                cur[_NUMERIC[key]] = int(n.group())
        elif key in ("authorized", "associated"):
            cur[key] = value.strip().lower() == "yes"
    return out
