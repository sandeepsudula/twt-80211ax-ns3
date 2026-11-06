# Target Wake Time (TWT) power saving for IEEE 802.11ax in ns-3

Simulates 802.11ax individual Target Wake Time in ns-3.48 and compares it with
legacy power save (PS-Poll) and always-on operation. The comparison covers
energy, throughput, latency and packet loss across TWT wake intervals, service
period durations and station counts.

> **Status:** the scenario builds and runs on ns-3.48, and the sweep and analysis pipeline works
> end to end. A quick sweep (IoT traffic, 10/40 STAs, 3 seeds) has run; the full sweep is next.

## Layout

| Path | Contents |
|---|---|
| `ns3/scratch/twt-sim/` | ns-3 program: copy into `<ns-3.48>/scratch/twt-sim/` |
| `ns3/scratch/twt-sim/twt-power-save-manager.{h,cc}` | Individual TWT power save manager (subclass of ns-3's `PowerSaveManager`) |
| `ns3/scratch/twt-sim/twt-sim.cc` | Scenario: 1 HE AP, N HE STAs, uplink UDP; modes `cam`, `psm`, `twt` (`--twtEarlySleep` for early sleep) |
| `sweep/` | Parallel sweep driver, analysis/plots, configs ([details](sweep/README.md)) |
| `dqn/` | Deep Q-Network that picks the TWT wake interval and SP per network ([details](dqn/README.md)) |
| `halow/` | `HaLowWrapper`: runs the DQN's chosen TWT schedule on a real Wi-Fi HaLow (MM6108) access point ([details](halow/README.md)) |

## How TWT is modeled

ns-3.48 ships a power-save framework and legacy power save, but not TWT
agreements. `TwtPowerSaveManager` adds an individual TWT schedule that is set up
in code, so no TWT Setup frames are exchanged:

- SP *k* of a STA starts at `offset + k · wakeInterval` on the shared simulation clock
  and lasts `spDuration`. By default, offsets are spread evenly over the interval.
- Inside an SP the STA radio is awake, and unicast frames can flow both ways.
- At SP end, unicast queues are frozen on the STA (uplink) and at the AP
  (downlink for that STA). Frames are held, not dropped. The radio sleeps as soon
  as no exchange is in progress.
- Outside an SP, uplink packets do not wake the radio; they wait for the next SP.
- Beacon-loss disassociation is disabled, because dozing STAs skip beacons by design.
- Optional `--twtEarlySleep`: the STA dozes before SP end once it has nothing left to send.

Association and Block Ack agreements are set up with `WifiStaticSetupHelper`.
Energy uses `WifiRadioEnergyModel` with ns-3's default currents
(TX 380 mA, RX 313 mA, idle 273 mA, sleep 33 mA, 3 V).

## First observations (quick sweep, IoT uplink, 100 B every 100 ms)

| Mode | Energy savings vs always-on | Mean latency |
|---|---|---|
| Legacy PS (PS-Poll) | ~87% | < 1 ms |
| TWT, awake for the whole SP | 66–79% (depends on SP share) | ≈ half the wake interval |
| TWT + early sleep | ~87–88% | ≈ half the wake interval |

For sparse uplink-only traffic, ns-3's legacy power save is already near-ideal. It
sleeps right after each transmission and wakes only for beacons, so TWT gains
nothing unless the STA dozes as soon as it is done. TWT's advantage shows up under
load: in a 20-STA, 2 Mb/s-per-STA test, legacy power save slept only ~14% of the time,
while TWT (50 ms interval, 25 ms SP) slept ~49%, used ~43% less energy than always-on, and
cut mean latency from ~32 ms to ~9 ms by spreading out contention.

These are preliminary single-scenario numbers; the full sweep will replace them.

## Learning the TWT settings with a DQN

`dqn/` trains a Deep Q-Network that reads a network's situation (number of stations, packet
interval and size, load, delay target) and picks one of 22 settings: always on, or a TWT
wake interval / SP pair. Each choice is scored by one `twt-sim` run. On 30 unseen networks the
DQN used 22% less power than the best single fixed TWT setting and 84% less than always-on,
with no delay-target violations. See [dqn/README.md](dqn/README.md).

## Quick start

```bash
cp -r ns3/scratch/twt-sim <ns-3.48>/scratch/
cd <ns-3.48>
./ns3 configure -d optimized --enable-modules="wifi;flow-monitor;applications;internet;energy"
./ns3 build twt-sim
./ns3 run "twt-sim --mode=twt --nSta=20 --wakeIntervalMs=100 --spDurationMs=10 --verbose=1"
# debug: print the PHY state timeline of one STA
./ns3 run "twt-sim --mode=twt --simTime=0.5 --traceSta=0"

cd <repo>/sweep
pip install pandas matplotlib numpy scipy
python3 run_sweep.py --ns3-dir <ns-3.48> --config sweep_quick.json --jobs 4
python3 analyze.py results/all_runs.csv --out figures
```
