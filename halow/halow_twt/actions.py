"""
Turn an agent's action into one StationSchedule per station.

Two action formats are supported:

1. This repository's DQN (dqn/env.py): an integer index into 22 actions, either always-on
   ("cam") or one (wake interval, service period) pair used by every station. As in twt-sim,
   the stations' wake times are spread evenly over the interval.

2. The group format of the SHINe Lab ICCCN'26 scheduler (TXST-SHINe-Lab/802.11ax-DRL-TWT-ICCCN26):
   a dict with "group_configs" (interval, duration and offset per group) and "sta_assignments"
   (which group each station is in). Stations in one group share one window.
"""

from .schedule import (
    MIN_USEFUL_DURATION_US,
    TESTED_INTERVAL_US,
    ScheduleError,
    StationSchedule,
)

# Same list, in the same order, as ACTIONS in dqn/env.py (a test checks they match).
_WAKE_INTERVALS_MS = [10, 20, 50, 100, 200, 500]
_SP_DURATIONS_MS = [2, 5, 10, 20]
DQN_ACTIONS = [("cam", 0, 0)] + [
    ("twt", wi, sp) for wi in _WAKE_INTERVALS_MS for sp in _SP_DURATIONS_MS if sp < wi
]


class UnsupportedAction(ScheduleError):
    """The action has no equivalent on the HaLow AP's control interface."""


def dqn_action_name(index):
    mode, wi, sp = DQN_ACTIONS[index]
    return "cam" if mode == "cam" else f"twt {wi}/{sp} ms"


def dqn_action_hardware_notes(index):
    """Why a DQN action may not behave on the bench as it did in ns-3 (empty list = fine)."""
    mode, wi, sp = DQN_ACTIONS[index]
    if mode == "cam":
        return ["always-on means no TWT agreement; the AP's control file can only change "
                "agreements, not remove them"]
    notes = []
    lo, hi = TESTED_INTERVAL_US
    if not lo <= wi * 1000 <= hi:
        notes.append(f"interval {wi} ms is outside the tested {lo // 1000}-{hi // 1000} ms range")
    if sp * 1000 < MIN_USEFUL_DURATION_US:
        notes.append(f"window {sp} ms is below {MIN_USEFUL_DURATION_US / 1000:.1f} ms and "
                     "carried no traffic on the bench")
    return notes


def dqn_action_to_schedules(index, stations):
    """
    index: DQN action index. stations: list of {"name": ..., "mac": ...} in roster order.
    Every station gets the same interval and window; wake times are spread evenly.
    """
    if not 0 <= index < len(DQN_ACTIONS):
        raise ScheduleError(f"DQN action {index} does not exist (0-{len(DQN_ACTIONS) - 1})")
    mode, wi, sp = DQN_ACTIONS[index]
    if mode == "cam":
        raise UnsupportedAction(dqn_action_hardware_notes(index)[0])
    interval_us, duration_us = wi * 1000, sp * 1000
    n = len(stations)
    return [StationSchedule(mac=st["mac"], name=st.get("name", ""),
                            wake_us=(i * interval_us) // n, interval_us=interval_us,
                            duration_us=duration_us)
            for i, st in enumerate(stations)]


def group_action_to_schedules(action, stations):
    """
    action: {"group_configs": [{"group_id", "twt_wake_interval_ms", "twt_wake_duration_ms",
                                "twt_sp_offset_ms"}, ...],
             "sta_assignments": [{"sta_id", "assigned_twt_group", "enable_twt"}, ...]}
    sta_id indexes into stations. Stations with enable_twt false get no line.
    Returns (schedules, notes).
    """
    groups = {int(g["group_id"]): g for g in action.get("group_configs", [])}
    schedules, notes = [], []
    for a in action.get("sta_assignments", []):
        sid = int(a["sta_id"])
        if sid >= len(stations):
            notes.append(f"sta_id {sid} has no station on the bench; skipped")
            continue
        st = stations[sid]
        if not a.get("enable_twt", True):
            notes.append(f"{st.get('name') or st['mac']}: TWT off in the action; its current "
                         "agreement is left unchanged")
            continue
        gid = int(a["assigned_twt_group"])
        if gid not in groups:
            raise ScheduleError(f"sta_id {sid} is assigned to group {gid}, which has no config")
        g = groups[gid]
        schedules.append(StationSchedule(
            mac=st["mac"], name=st.get("name", ""),
            wake_us=round(float(g["twt_sp_offset_ms"]) * 1000),
            interval_us=round(float(g["twt_wake_interval_ms"]) * 1000),
            duration_us=round(float(g["twt_wake_duration_ms"]) * 1000)))
    return schedules, notes
