# HaLowWrapper: run learned TWT schedules on a Wi-Fi HaLow bench

The DQN in [`../dqn`](../dqn) picks TWT settings inside ns-3. This folder runs those choices on
real Wi-Fi HaLow (802.11ah) radios: a Morse Micro MM6108 access point that can change any
station's TWT schedule while it is connected, as built on the SHINe Lab HaLow TWT bench.

`HaLowWrapper` has the same `reset()` / `step(action)` shape as a simulator wrapper, so the
trained agent does not change; only what sits behind `step()` does.

```
agent ──action──▶ HaLowWrapper.step()
                    1. action → one schedule line per station, rounded to what the hardware can encode
                    2. check it: refuse errors, report warnings
                    3. write each line on the AP: <MAC> <wake us> <interval us> <duration us>
                    4. wait until the AP's agreement table shows every new schedule (station acknowledged)
                    5. wait settle_s (stations switch 4–23 s after the write)
                    6. read the AP's per-station counters, wait measure_s, read them again
agent ◀──obs───── 7. differences + installed schedule + estimated power
```

## Files

| File | What it does |
|---|---|
| `halow_twt/schedule.py` | Hardware limits, rounding (256 µs duration units, mantissa × 2^exp intervals), schedule checks |
| `halow_twt/actions.py` | DQN action index → per-station schedules; also the ICCCN'26 group format |
| `halow_twt/ap.py` | Commands to the AP over ssh: push schedules, read agreements, read the station dump |
| `halow_twt/parse.py` | Parsers for `twt_sta_agreements` and `iw dev wlan0 station dump` |
| `halow_twt/wrapper.py` | `HaLowWrapper` (`reset` / `step` / `close`), config loading, CSV log |
| `halow_twt/dryrun.py` | A simulated AP that answers the same commands, for testing without hardware |
| `halow_twt/__main__.py` | Command line: `actions`, `status`, `apply`, `deploy` |
| `bench.example.json` | Config template (placeholder MACs) |
| `tests/` | Unit tests and dry-run episodes (`python -m pytest -q tests`) |

The core package uses only the Python standard library. `deploy` also needs `torch` and `numpy`
(already in `../dqn/requirements.txt`); the tests need `pytest`.

## Hardware limits it enforces

Measured on the HaLow bench:

| Limit | Value | What the wrapper does |
|---|---|---|
| Longest window | 65,280 µs (one byte × 256 µs) | Rounds durations up to 256 µs steps, clips at 65,280, reports it |
| Shortest useful window | ~6.1 ms (≤ 4.1 ms carried no traffic) | Warning |
| Tested wake intervals | 250 ms – 2.5 s | Warning outside this range |
| Interval encoding | mantissa (16 bit) × 2^exp | Rounds to the nearest encodable value, reports it |
| Window longer than its interval | refused by the AP | Error, nothing is sent |
| Overlapping windows | the AP does not check | Warning (stations sharing one exact window are fine) |
| Schedule adoption | 4–23 s after the write | Waits for acknowledgement, then `settle_s` (35 s) |

**Only 2 of the DQN's 22 actions are inside these limits** (`twt 500/10` and `twt 500/20`).
Run `python -m halow_twt actions` for the full list. The DQN was trained for 802.11ax, with
intervals of 10–500 ms and windows of 2–20 ms. For example, for a 4-station, 100 ms-packet network
with a 500 ms delay target, its first choice is `twt 200/2`, which would carry no traffic on
HaLow. `deploy --legal-only` restricts the choice to legal actions. The real fix is retraining
with a HaLow-legal action set.

Always-on (`cam`, action 0) is not supported: the AP's control file changes agreements but cannot
remove them.

## Setup

1. Make the AP reachable with plain `ssh <host>` from the machine that runs this (key-based login;
   the bench uses the alias `mmap`). Test: `ssh mmap 'ls -d /sys/kernel/debug/ieee80211/*/morse'`.
2. `cp bench.example.json bench.json` and enter your stations' MAC addresses, in the order the
   agent should see them. `bench.json` is git-ignored so real MACs stay out of the repository.
3. **Stop anything else that changes schedules.** The bench dashboard's periodic health check
   (`twt_health.sh --repair`) re-applies the last schedule *it* applied and would undo the
   wrapper's changes within minutes.

## Use

```bash
cd halow
python -m halow_twt actions                                     # which DQN actions the bench can run
python -m halow_twt status --config bench.json                  # what the AP has installed now
python -m halow_twt apply  --config bench.json --action 21      # run DQN action 21 (twt 500/20) and measure
python -m halow_twt apply  --config bench.json --schedule cfg1.conf   # 'name wake interval duration' lines
python -m halow_twt deploy --config bench.json \
    --model ../dqn/runs/v2_delay_target/dqn_twt.pt \
    --n-sta 4 --pkt-interval-ms 100 --pkt-size 500 --target-ms 500 --legal-only
```

Add `--dry-run` to any command to use the simulated AP (no ssh, finishes instantly), and `--json`
to print the whole observation.

From Python:

```python
from halow_twt import HaLowWrapper, load_config

env = HaLowWrapper(load_config("bench.json"), verbose=True)
obs = env.reset()
obs, done = env.step(21)                 # DQN action index
obs, done = env.step({                   # or the ICCCN'26 group format
    "group_configs": [{"group_id": 0, "twt_wake_interval_ms": 500,
                       "twt_wake_duration_ms": 20, "twt_sp_offset_ms": 0}],
    "sta_assignments": [{"sta_id": i, "assigned_twt_group": 0, "enable_twt": True}
                        for i in range(4)],
})
print(obs["summary"])
```

## The observation

Per station, measured over `measure_s` after the switch:

- installed `wake_us`, `interval_us`, `duration_us` (read back from the AP, not what was asked for),
  and `duty_cycle` = duration / interval
- deltas of `rx_packets`, `rx_bytes`, `tx_packets`, `tx_retries`, `tx_failed`, and `rx_kbps`
- `signal_dbm`, `inactive_ms` (time since the AP last heard the station)
- `est_power_w`, **an estimate, not a measurement**: duty cycle × idle current + the rest × sleep
  current. The default currents are ns-3's model values used in `../ns3`; replace them in
  `energy_model` once the MM6108 stations' current draw is measured.
- `realistic`: the same values under the names of the AP-observable ("realistic") metrics in the
  ns-3 wrapper (`rx_fragment_count`, `bytes_received_at_ap`, `rssi_dbm`, `last_rx_ms_ago`,
  `duty_cycle`, `awake_time_ms`, `sleep_time_ms`).

Each step also returns `notes` (rounding applied) and `warnings` (limits crossed). With `log_csv`
set, every step appends one row per station to a CSV.

## Known gaps

- One step takes about a minute (acknowledgement + 35 s settle + 25 s measurement), so this is
  for evaluating a trained policy, not for training on hardware.
- Power is estimated from the schedule; it is not measured.
- Delay is not measured. The AP's counters give delivered traffic, not per-packet latency.
- Not yet run against the real bench; the tests use the simulated AP.
