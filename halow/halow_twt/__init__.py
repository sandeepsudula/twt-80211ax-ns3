"""
halow_twt: run TWT schedules chosen by an RL agent on a Wi-Fi HaLow (Morse Micro MM6108) bench.

The agent-facing class is HaLowWrapper. It has the same reset() / step() shape as the ns-3
wrappers, so a policy trained in simulation can drive the real access point unchanged.
"""

from .schedule import (
    DURATION_UNIT_US,
    MAX_DURATION_US,
    MIN_USEFUL_DURATION_US,
    TESTED_INTERVAL_US,
    ScheduleError,
    StationSchedule,
    quantize,
    quantize_duration_us,
    quantize_interval_us,
    validate,
)
from .actions import (
    DQN_ACTIONS,
    UnsupportedAction,
    dqn_action_hardware_notes,
    dqn_action_to_schedules,
    group_action_to_schedules,
)
from .wrapper import BenchConfig, HaLowWrapper, load_config

__all__ = [
    "BenchConfig", "DQN_ACTIONS", "DURATION_UNIT_US", "HaLowWrapper", "MAX_DURATION_US",
    "MIN_USEFUL_DURATION_US", "ScheduleError", "StationSchedule", "TESTED_INTERVAL_US",
    "UnsupportedAction", "dqn_action_hardware_notes", "dqn_action_to_schedules",
    "group_action_to_schedules", "load_config", "quantize", "quantize_duration_us",
    "quantize_interval_us", "validate",
]
