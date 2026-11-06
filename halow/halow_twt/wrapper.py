"""
HaLowWrapper: the real HaLow bench behind the same reset() / step() interface as a simulator.

    env = HaLowWrapper(load_config("bench.json"))
    obs = env.reset()
    obs, done = env.step(action)      # an int (this repo's DQN) or a group-format dict

One step:
  1. translate the action into one schedule line per station, rounded to what the
     hardware can express, and check it (refuse errors, report warnings)
  2. write the lines on the AP (twt_ap_update)
  3. wait until the AP's agreement table shows every new schedule, i.e. each station
     acknowledged its update
  4. wait settle_s more: stations switch 4-23 s after the write, so no measurement may
     straddle the change
  5. read the AP's per-station counters, wait measure_s, read them again
  6. return the differences as the observation
"""

import csv
import datetime
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .actions import dqn_action_to_schedules, group_action_to_schedules
from .ap import APError, HaLowAP, SSHTransport
from .dryrun import DryRunTransport, SimClock
from .schedule import (
    ScheduleError,
    StationSchedule,
    normalize_mac,
    quantize,
    validate,
)


@dataclass
class BenchConfig:
    stations: list = field(default_factory=list)   # [{"name": "sta1", "mac": "..."}], in order
    ap_host: str = "mmap"                          # anything ssh accepts
    ssh_options: list = field(default_factory=list)
    interface: str = "wlan0"
    settle_s: float = 35.0       # discard the 4-23 s schedule transition
    measure_s: float = 25.0      # measurement window per step
    verify_timeout_s: float = 30.0
    poll_s: float = 2.0
    max_steps: Optional[int] = None
    allow_warnings: bool = True
    # Radio currents used ONLY to estimate power from the duty cycle. The defaults are ns-3's
    # WifiRadioEnergyModel values used in this repository's simulator, not MM6108 measurements;
    # replace them once the stations' current draw has been measured.
    energy_model: dict = field(default_factory=lambda: {
        "idle_ma": 273.0, "sleep_ma": 33.0, "voltage_v": 3.0})
    log_csv: Optional[str] = None


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}   # "_comment" keys
    cfg = BenchConfig(**raw)
    cfg.stations = [{"name": st.get("name", ""), "mac": normalize_mac(st["mac"])}
                    for st in cfg.stations]
    if not cfg.stations:
        raise ValueError(f"{path}: no stations listed")
    return cfg


def read_schedule_file(path, stations):
    """A bench config file: one 'name wake_us interval_us duration_us' line per station."""
    by_name = {st["name"]: st["mac"] for st in stations}
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            name, wake, interval, duration = parts[:4]
            if name not in by_name:
                raise ScheduleError(f"{path}: {name} is not in the station list")
            out.append(StationSchedule(mac=by_name[name], name=name, wake_us=int(wake),
                                       interval_us=int(interval), duration_us=int(duration)))
    return out


class HaLowWrapper:
    def __init__(self, config, transport=None, dry_run=False, sleep=None, clock=None,
                 verbose=False):
        self.config = config
        self.verbose = verbose
        if dry_run and (sleep is None or clock is None):
            sim = SimClock()
            sleep, clock = sim.sleep, sim.now
        self.sleep = sleep or time.sleep
        self.clock = clock or time.monotonic
        if transport is None:
            transport = (DryRunTransport(config.stations, clock=_ClockAdapter(self.clock))
                         if dry_run else
                         SSHTransport(config.ap_host, config.ssh_options))
        self.ap = HaLowAP(transport, config.interface)
        self.step_count = 0
        self.last_schedules = None

    # ---- agent interface
    def reset(self):
        """Check the bench is ready and return a first observation (all deltas zero)."""
        self.step_count = 0
        self.ap.debugfs_dir()
        dump = self.ap.stations()
        missing = [st["name"] or st["mac"] for st in self.config.stations
                   if not dump.get(st["mac"], {}).get("authorized", st["mac"] in dump)]
        if missing:
            raise APError("not associated with the AP: " + ", ".join(missing))
        return self._observation(dump, dump, 0.0, self.ap.agreements())

    def step(self, action):
        schedules, notes = self.translate(action)
        final = []
        for s in schedules:
            q, n = quantize(s)
            final.append(q)
            notes += n
        errors, warnings = validate(final)
        if errors:
            raise ScheduleError("; ".join(errors))
        if warnings and not self.config.allow_warnings:
            raise ScheduleError("warnings not allowed: " + "; ".join(warnings))
        self._say(*notes, *warnings)

        refused = [s.label() for s, ok in self.ap.push(final) if not ok]
        if refused:
            raise APError("the AP refused the schedule for: " + ", ".join(refused))
        self.last_schedules = final
        installed = self._wait_installed(final)

        self._say(f"waiting {self.config.settle_s:.0f} s for stations to switch")
        self.sleep(self.config.settle_s)
        before, t0 = self.ap.stations(), self.clock()
        self._say(f"measuring for {self.config.measure_s:.0f} s")
        self.sleep(self.config.measure_s)
        after, t1 = self.ap.stations(), self.clock()

        obs = self._observation(before, after, t1 - t0, installed)
        obs["notes"], obs["warnings"] = notes, warnings
        self.step_count += 1
        self._log(obs)
        done = self.config.max_steps is not None and self.step_count >= self.config.max_steps
        return obs, done

    def close(self):
        """Nothing to release. The last schedule stays installed on the AP."""

    # ---- helpers
    def translate(self, action):
        """Action -> (list of StationSchedule, notes)."""
        stations = self.config.stations
        if isinstance(action, bool):
            raise ScheduleError("an action cannot be a bool")
        if hasattr(action, "__index__"):           # int, or a numpy integer from the agent
            return dqn_action_to_schedules(int(action), stations), []
        if isinstance(action, dict):
            return group_action_to_schedules(action, stations)
        if isinstance(action, (list, tuple)) and all(isinstance(s, StationSchedule) for s in action):
            return list(action), []
        raise ScheduleError(f"unknown action type: {type(action).__name__}")

    def _wait_installed(self, schedules):
        """Poll the agreement table until it matches what was sent (stations acknowledged)."""
        deadline = self.clock() + self.config.verify_timeout_s
        while True:
            agreements = self.ap.agreements()
            pending = [s for s in schedules if not _matches(agreements.get(s.mac), s)]
            if not pending:
                return agreements
            if self.clock() >= deadline:
                detail = []
                for s in pending:
                    a = agreements.get(s.mac)
                    have = (f"{a.wake_us}/{a.interval_us}/{a.duration_us}" if a else "none")
                    detail.append(f"{s.label()} wants {s.wake_us}/{s.interval_us}/"
                                  f"{s.duration_us}, AP has {have}")
                raise APError("stations did not acknowledge in time: " + "; ".join(detail))
            self.sleep(self.config.poll_s)

    def _observation(self, before, after, window_s, agreements):
        em = self.config.energy_model
        rows = []
        for st in self.config.stations:
            mac = st["mac"]
            b, a = before.get(mac, {}), after.get(mac, {})
            d = {k: a.get(k, 0) - b.get(k, 0)
                 for k in ("rx_packets", "rx_bytes", "tx_packets", "tx_retries", "tx_failed")}
            ag = agreements.get(mac)
            duty = ag.duration_us / ag.interval_us if ag else None
            power = (em["voltage_v"] * (duty * em["idle_ma"] + (1 - duty) * em["sleep_ma"]) / 1000
                     if duty is not None else None)
            rx_kbps = d["rx_bytes"] * 8 / window_s / 1000 if window_s > 0 else 0.0
            rows.append({
                "name": st["name"], "mac": mac, "associated": mac in after,
                "wake_us": ag.wake_us if ag else None,
                "interval_us": ag.interval_us if ag else None,
                "duration_us": ag.duration_us if ag else None,
                "duty_cycle": duty, **d, "rx_kbps": rx_kbps,
                "signal_dbm": a.get("signal_dbm"), "inactive_ms": a.get("inactive_ms"),
                "est_power_w": power,
                # same meaning as the "realistic" (AP-observable) metrics of the ns-3 wrapper
                "realistic": {
                    "rx_fragment_count": d["rx_packets"],
                    "bytes_received_at_ap": d["rx_bytes"],
                    "rssi_dbm": a.get("signal_dbm"),
                    "last_rx_ms_ago": a.get("inactive_ms"),
                    "duty_cycle": duty,
                    "awake_time_ms": duty * window_s * 1000 if duty is not None else None,
                    "sleep_time_ms": (1 - duty) * window_s * 1000 if duty is not None else None,
                },
            })
        duties = [r["duty_cycle"] for r in rows if r["duty_cycle"] is not None]
        powers = [r["est_power_w"] for r in rows if r["est_power_w"] is not None]
        return {
            "step": self.step_count,
            "window_s": window_s,
            "stations": rows,
            "summary": {
                "total_rx_kbps": sum(r["rx_kbps"] for r in rows),
                "stations_with_traffic": sum(1 for r in rows if r["rx_packets"] > 0),
                "mean_duty_cycle": sum(duties) / len(duties) if duties else None,
                "mean_est_power_w": sum(powers) / len(powers) if powers else None,
            },
        }

    def _log(self, obs):
        path = self.config.log_csv
        if not path:
            return
        new = not os.path.exists(path)
        cols = ["time", "step", "name", "mac", "wake_us", "interval_us", "duration_us",
                "duty_cycle", "rx_packets", "rx_bytes", "rx_kbps", "tx_retries", "tx_failed",
                "signal_dbm", "inactive_ms", "est_power_w"]
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in obs["stations"]:
                w.writerow({"time": now, "step": obs["step"], **r})

    def _say(self, *lines):
        if self.verbose:
            for line in lines:
                print("  " + line)


def _matches(agreement, schedule):
    if agreement is None:
        return False
    return (agreement.wake_us == schedule.wake_us
            and agreement.duration_us == schedule.duration_us
            and abs(agreement.interval_us - schedule.interval_us) <= 0.005 * schedule.interval_us)


class _ClockAdapter:
    """Lets DryRunTransport read the wrapper's clock function."""

    def __init__(self, fn):
        self.now = fn
