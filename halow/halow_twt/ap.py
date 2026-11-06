"""
Talking to the HaLow access point.

The AP is a Raspberry Pi running OpenWrt. Everything goes through one shell command at a time,
over ssh in real use or through DryRunTransport for testing without hardware.
"""

import shlex
import subprocess

from .parse import parse_agreements, parse_station_dump

DEBUGFS_GLOB = "/sys/kernel/debug/ieee80211/*/morse"


class APError(RuntimeError):
    """The AP could not be reached or refused a command."""


class SSHTransport:
    """Runs commands on the AP with ssh. host is anything ssh accepts, e.g. the alias "mmap"."""

    def __init__(self, host, ssh_options=(), timeout_s=25):
        self.host = host
        self.ssh_options = list(ssh_options)
        self.timeout_s = timeout_s

    def run(self, command):
        args = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                *self.ssh_options, self.host, command]
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return 255, ""
        return res.returncode, res.stdout


class HaLowAP:
    def __init__(self, transport, interface="wlan0"):
        self.transport = transport
        self.interface = interface
        self._debugfs = None

    def debugfs_dir(self):
        """The morse driver's debugfs folder; failing to find it means the AP is unreachable."""
        if self._debugfs is None:
            rc, out = self.transport.run(f"ls -d {DEBUGFS_GLOB} 2>/dev/null | head -1")
            path = out.strip()
            if rc != 0 or not path:
                raise APError("AP unreachable, or the morse driver's debugfs folder is missing")
            self._debugfs = path
        return self._debugfs

    def push(self, schedules):
        """
        Write one twt_ap_update line per station. Returns [(schedule, ok)].
        The driver refuses a window that does not fit its interval; that shows as ok=False.
        The driver itself resends each update over the air until the station acknowledges it.
        """
        target = f"{self.debugfs_dir()}/twt_ap_update"
        results = []
        for s in schedules:
            rc, _ = self.transport.run(f"echo {shlex.quote(s.to_line())} > {target}")
            results.append((s, rc == 0))
        return results

    def agreements(self):
        rc, out = self.transport.run(f"cat {self.debugfs_dir()}/twt_sta_agreements")
        if rc != 0:
            raise APError("could not read twt_sta_agreements")
        return parse_agreements(out)

    def stations(self):
        rc, out = self.transport.run(f"iw dev {self.interface} station dump")
        if rc != 0:
            raise APError("could not read the station dump")
        return parse_station_dump(out)
