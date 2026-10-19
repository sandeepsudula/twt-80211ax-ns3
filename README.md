# Target Wake Time (TWT) power saving for IEEE 802.11ax in ns-3

Simulates 802.11ax individual Target Wake Time in ns-3.48 and compares it with
legacy power save (PS-Poll) and always-on operation. The comparison covers
energy, throughput, latency and packet loss across TWT wake intervals, service
period durations and station counts.

> **Status:** work in progress. The ns-3 scenario has not yet been validated end to end.

## Layout

| Path | Contents |
|---|---|
| `ns3/scratch/twt-sim/` | ns-3 program: copy into `<ns-3.48>/scratch/twt-sim/` |
| `ns3/scratch/twt-sim/twt-power-save-manager.{h,cc}` | Individual TWT power save manager (subclass of ns-3's `PowerSaveManager`) |
| `ns3/scratch/twt-sim/twt-sim.cc` | Scenario: 1 HE AP, N HE STAs, uplink UDP; modes `cam`, `psm`, `twt` |
| `sweep/` | Parallel sweep driver, analysis/plots, configs ([details](sweep/README.md)) |

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

## Quick start

```bash
cp -r ns3/scratch/twt-sim <ns-3.48>/scratch/
cd <ns-3.48>
./ns3 configure -d optimized --enable-modules="wifi;flow-monitor;applications;internet;energy"
./ns3 build twt-sim
./ns3 run "twt-sim --mode=twt --nSta=20 --wakeIntervalMs=100 --spDurationMs=10 --verbose=1"

cd <repo>/sweep
pip install pandas matplotlib numpy scipy
python3 run_sweep.py --ns3-dir <ns-3.48> --config sweep_quick.json --jobs 4
python3 analyze.py results/all_runs.csv --out figures
```
