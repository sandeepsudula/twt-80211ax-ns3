import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from halow_twt import (  # noqa: E402
    DQN_ACTIONS, MAX_DURATION_US, BenchConfig, HaLowWrapper, ScheduleError, StationSchedule,
    UnsupportedAction, dqn_action_hardware_notes, dqn_action_to_schedules,
    group_action_to_schedules, quantize, quantize_duration_us, quantize_interval_us, validate,
)
from halow_twt.ap import APError  # noqa: E402
from halow_twt.parse import parse_agreements, parse_station_dump  # noqa: E402

STATIONS = [{"name": f"sta{i + 1}", "mac": f"02:00:00:00:00:0{i + 1}"} for i in range(4)]

AGREEMENTS_TEXT = """wlan0:
TWT Agreements for Requester: 02:00:00:00:00:01, Responder: 02:00:00:00:00:ff
\tFlow ID: 0, Wake Interval: 9999872 us, Wake Time: 0 us, Wake Duration: 65280 us, State 5
\tFlow ID: 1, Wake Interval: 0 us, Wake Time: 0 us, Wake Duration: 0 us, State 0
TWT Agreements for Requester: 02:00:00:00:00:02, Responder: 02:00:00:00:00:ff
\tFlow ID: 0, Wake Interval: 0 us, Wake Time: 0 us, Wake Duration: 0 us, State 0
"""

DUMP_TEXT = """Station 02:00:00:00:00:01 (on wlan0)
\tinactive time:\t3430 ms
\trx bytes:\t6032
\trx packets:\t110
\ttx retries:\t2
\ttx failed:\t0
\tsignal:  \t-1 dBm
\tsignal avg:\t-1 dBm
\ttx bitrate:\t130.0 MBit/s VHT-MCS 1 160MHz short GI VHT-NSS 1
\tauthorized:\tyes
Station 02:00:00:00:00:02 (on wlan0)
\trx packets:\t7
\tauthorized:\tno
"""


# ---- encoding limits
def test_interval_encoding_matches_bench():
    assert quantize_interval_us(10_000_000) == 9_999_872     # the value the AP really stored
    assert quantize_interval_us(1_000_000) == 1_000_000
    assert quantize_interval_us(102_400) == 102_400
    assert quantize_interval_us(65_535) == 65_535


def test_duration_rounds_up_and_clips():
    assert quantize_duration_us(45_000) == 45_056             # 176 x 256
    assert quantize_duration_us(65_280) == 65_280
    assert quantize_duration_us(200_000) == MAX_DURATION_US


def test_quantize_reports_changes():
    q, notes = quantize(StationSchedule("02:00:00:00:00:01", 0, 10_000_000, 200_000))
    assert (q.interval_us, q.duration_us) == (9_999_872, 65_280)
    assert len(notes) == 2


# ---- validation
def test_validate_errors():
    errors, _ = validate([StationSchedule("02:00:00:00:00:01", 0, 10_000, 20_000)])
    assert any("longer than its interval" in e for e in errors)
    errors, _ = validate([StationSchedule("02:00:00:00:00:01", 1_000_000, 1_000_000, 10_000)])
    assert any("wake time" in e for e in errors)
    errors, _ = validate([StationSchedule("bad-mac", 0, 1_000_000, 10_000)])
    assert errors


def test_validate_overlap_and_sharing():
    a = StationSchedule("02:00:00:00:00:01", 0, 1_000_000, 65_280)
    b = StationSchedule("02:00:00:00:00:02", 30_000, 1_000_000, 65_280)   # overlaps a
    c = StationSchedule("02:00:00:00:00:03", 0, 1_000_000, 65_280)        # shares a's window
    _, warnings = validate([a, b])
    assert any("overlap" in w for w in warnings)
    _, warnings = validate([a, c])
    assert not any("overlap" in w for w in warnings)


def test_validate_wraparound_overlap():
    a = StationSchedule("02:00:00:00:00:01", 990_000, 1_000_000, 20_000)
    b = StationSchedule("02:00:00:00:00:02", 5_000, 1_000_000, 10_000)
    _, warnings = validate([a, b])
    assert any("overlap" in w for w in warnings)


def test_validate_bench_warnings():
    _, warnings = validate([StationSchedule("02:00:00:00:00:01", 0, 100_000, 2_048)])
    assert any("outside the tested range" in w for w in warnings)
    assert any("carried no traffic" in w for w in warnings)


# ---- actions
def test_dqn_actions_match_repo_env():
    np = pytest.importorskip("numpy")  # noqa: F841  (env.py imports numpy)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "dqn"))
    from env import ACTIONS
    assert [tuple(a) for a in ACTIONS] == [tuple(a) for a in DQN_ACTIONS]


def test_dqn_action_staggers_stations():
    idx = DQN_ACTIONS.index(("twt", 500, 20))
    scheds = dqn_action_to_schedules(idx, STATIONS)
    assert [s.wake_us for s in scheds] == [0, 125_000, 250_000, 375_000]
    assert all(s.interval_us == 500_000 and s.duration_us == 20_000 for s in scheds)
    assert dqn_action_hardware_notes(idx) == []


def test_dqn_cam_is_unsupported():
    with pytest.raises(UnsupportedAction):
        dqn_action_to_schedules(0, STATIONS)


def test_hardware_legal_dqn_actions():
    legal = [DQN_ACTIONS[i] for i in range(len(DQN_ACTIONS)) if not dqn_action_hardware_notes(i)]
    assert legal == [("twt", 500, 10), ("twt", 500, 20)]


def test_group_action():
    action = {
        "group_configs": [
            {"group_id": 0, "twt_wake_interval_ms": 102.4, "twt_wake_duration_ms": 22,
             "twt_sp_offset_ms": 0},
            {"group_id": 1, "twt_wake_interval_ms": 102.4, "twt_wake_duration_ms": 32,
             "twt_sp_offset_ms": 24},
        ],
        "sta_assignments": [
            {"sta_id": 0, "assigned_twt_group": 0, "enable_twt": True},
            {"sta_id": 1, "assigned_twt_group": 0, "enable_twt": True},
            {"sta_id": 2, "assigned_twt_group": 1, "enable_twt": True},
            {"sta_id": 3, "assigned_twt_group": 1, "enable_twt": False},
        ],
    }
    scheds, notes = group_action_to_schedules(action, STATIONS)
    assert [(s.wake_us, s.duration_us) for s in scheds] == [(0, 22_000), (0, 22_000),
                                                            (24_000, 32_000)]
    assert len(notes) == 1
    _, warnings = validate([quantize(s)[0] for s in scheds])
    assert not any("overlap" in w for w in warnings)


# ---- parsers
def test_parse_agreements():
    ag = parse_agreements(AGREEMENTS_TEXT)
    assert list(ag) == ["02:00:00:00:00:01"]
    a = ag["02:00:00:00:00:01"]
    assert (a.wake_us, a.interval_us, a.duration_us, a.state) == (0, 9_999_872, 65_280, 5)


def test_parse_station_dump():
    d = parse_station_dump(DUMP_TEXT)
    s = d["02:00:00:00:00:01"]
    assert (s["rx_packets"], s["rx_bytes"], s["inactive_ms"], s["signal_dbm"]) == (110, 6032, 3430, -1)
    assert s["authorized"] is True
    assert d["02:00:00:00:00:02"]["authorized"] is False


# ---- end to end on the simulated AP
def _config(**kw):
    return BenchConfig(stations=STATIONS, **kw)


def test_dry_run_episode():
    env = HaLowWrapper(_config(max_steps=2), dry_run=True)
    obs = env.reset()
    assert obs["summary"]["total_rx_kbps"] == 0
    idx20 = DQN_ACTIONS.index(("twt", 500, 20))
    idx10 = DQN_ACTIONS.index(("twt", 500, 10))
    obs20, done = env.step(idx20)
    assert not done
    assert all(r["interval_us"] == 500_000 and r["duration_us"] == 20_224 for r in obs20["stations"])
    assert obs20["window_s"] == pytest.approx(25.0)
    obs10, done = env.step(idx10)
    assert done
    # a shorter window gives a lower duty cycle and (in the toy model) less traffic
    assert obs10["summary"]["mean_duty_cycle"] < obs20["summary"]["mean_duty_cycle"]
    assert obs10["summary"]["total_rx_kbps"] < obs20["summary"]["total_rx_kbps"]
    assert obs10["summary"]["mean_est_power_w"] < obs20["summary"]["mean_est_power_w"]


def test_dry_run_commands_sent_to_ap():
    env = HaLowWrapper(_config(), dry_run=True)
    env.reset()
    env.step(DQN_ACTIONS.index(("twt", 500, 20)))
    writes = [c for c in env.ap.transport.commands if "twt_ap_update" in c]
    assert writes[0] == ("echo '02:00:00:00:00:01 0 500000 20224' > "
                         "/sys/kernel/debug/ieee80211/phy0/morse/twt_ap_update")
    assert len(writes) == 4


def test_errors_are_refused_before_sending():
    env = HaLowWrapper(_config(), dry_run=True)
    env.reset()
    bad = [StationSchedule("02:00:00:00:00:01", 0, 10_000, 20_000)]
    with pytest.raises(ScheduleError):
        env.step(bad)
    assert not [c for c in env.ap.transport.commands if "twt_ap_update" in c]


def test_warnings_can_be_made_fatal():
    env = HaLowWrapper(_config(allow_warnings=False), dry_run=True)
    env.reset()
    with pytest.raises(ScheduleError):
        env.step(DQN_ACTIONS.index(("twt", 100, 10)))      # interval outside tested range


def test_unacknowledged_update_times_out():
    env = HaLowWrapper(_config(verify_timeout_s=6), dry_run=True)
    env.reset()
    env.ap.transport.agreements["02:00:00:00:00:02"] = [0, 1_000_000, 65_280]
    original = env.ap.transport.run

    def drop_sta2(command):                 # the AP accepts the write, the station never acks
        if "02:00:00:00:00:02" in command and "twt_ap_update" in command:
            return 0, ""
        return original(command)

    env.ap.transport.run = drop_sta2
    with pytest.raises(APError, match="did not acknowledge"):
        env.step(DQN_ACTIONS.index(("twt", 500, 20)))


def test_reset_requires_association():
    env = HaLowWrapper(_config(), dry_run=True)
    del env.ap.transport.counters["02:00:00:00:00:04"]
    with pytest.raises(APError, match="sta4"):
        env.reset()


def test_schedule_file(tmp_path):
    from halow_twt.wrapper import read_schedule_file
    f = tmp_path / "cfg.conf"
    f.write_text("# one station per window\nsta1 0 1000000 65280\nsta2 250000 1000000 65280\n")
    scheds = read_schedule_file(str(f), STATIONS)
    assert [(s.name, s.wake_us) for s in scheds] == [("sta1", 0), ("sta2", 250_000)]
