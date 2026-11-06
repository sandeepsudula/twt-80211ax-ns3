"""
A stand-in for the HaLow AP, for trying the wrapper without hardware.

DryRunTransport answers the same shell commands as the real AP, in the same text formats, from
an in-memory table. Its traffic numbers come from a crude model (traffic grows with the usable
part of each window) and only exist so the pipeline has something to compute. They are not a
prediction of what the bench will measure.
"""

import re
import shlex

from .ap import DEBUGFS_GLOB
from .schedule import MAX_DURATION_US

_ECHO = re.compile(r"^echo (.+) > (\S+)/twt_ap_update$")
_DEBUGFS = DEBUGFS_GLOB.replace("*", "phy0")
_DEAD_TIME_US = 4096        # the bench carried no traffic in windows this short


class SimClock:
    """A clock that only moves when sleep() is called, so dry runs finish instantly."""

    def __init__(self, start=0.0):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, seconds)


class DryRunTransport:
    def __init__(self, stations, clock, offered_pps=50.0, peak_pps=400.0, payload_bytes=200):
        self.clock = clock
        self.offered_pps = offered_pps
        self.peak_pps = peak_pps
        self.payload_bytes = payload_bytes
        self.commands = []                  # every command received, for inspection
        n = max(1, len(stations))
        self.agreements = {st["mac"].lower(): [i * 1_000_000 // n, 1_000_000, MAX_DURATION_US]
                           for i, st in enumerate(stations)}
        self.counters = {mac: {"rx_packets": 0.0} for mac in self.agreements}
        self._last = clock.now()

    # ---- the traffic model
    def _rate_pps(self, mac):
        wake, interval, duration = self.agreements[mac]
        usable = max(0, duration - _DEAD_TIME_US) / interval
        return min(self.offered_pps, self.peak_pps * usable)

    def _advance(self):
        now = self.clock.now()
        dt, self._last = now - self._last, now
        for mac, c in self.counters.items():
            c["rx_packets"] += self._rate_pps(mac) * dt

    # ---- the AP's text formats
    def _render_agreements(self):
        lines = ["wlan0:"]
        for mac, (wake, interval, duration) in self.agreements.items():
            lines.append(f"TWT Agreements for Requester: {mac}, Responder: 02:00:00:00:00:ff")
            lines.append(f"\tFlow ID: 0, Wake Interval: {interval} us, Wake Time: {wake} us, "
                         f"Wake Duration: {duration} us, State 5")
            for flow in range(1, 8):
                lines.append(f"\tFlow ID: {flow}, Wake Interval: 0 us, Wake Time: 0 us, "
                             "Wake Duration: 0 us, State 0")
        return "\n".join(lines) + "\n"

    def _render_station_dump(self):
        self._advance()
        lines = []
        for mac, c in self.counters.items():
            _, interval, _ = self.agreements[mac]
            pkts = int(c["rx_packets"])
            lines += [f"Station {mac} (on wlan0)",
                      f"\tinactive time:\t{interval // 2000} ms",
                      f"\trx bytes:\t{pkts * self.payload_bytes}",
                      f"\trx packets:\t{pkts}",
                      f"\ttx bytes:\t{pkts * 40}",
                      f"\ttx packets:\t{pkts // 4}",
                      "\ttx retries:\t0",
                      "\ttx failed:\t0",
                      "\tsignal:  \t-40 dBm",
                      "\tsignal avg:\t-40 dBm",
                      "\tauthorized:\tyes",
                      "\tassociated:\tyes"]
        return "\n".join(lines) + "\n"

    def run(self, command):
        self.commands.append(command)
        if command.startswith("ls -d "):
            return 0, _DEBUGFS + "\n"
        m = _ECHO.match(command)
        if m:
            fields = shlex.split(m.group(1))[0].split()
            mac, wake, interval, duration = fields[0].lower(), *map(int, fields[1:4])
            if mac not in self.agreements or duration > interval:
                return 1, ""                # the driver refuses it
            self._advance()
            self.agreements[mac] = [wake, interval, duration]
            return 0, ""
        if command.endswith("/twt_sta_agreements"):
            return 0, self._render_agreements()
        if "station dump" in command:
            return 0, self._render_station_dump()
        return 1, ""
