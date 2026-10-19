# TWT power-save sweep (ns-3)

Sweep runner and analysis for comparing 802.11ax Target Wake Time against
always-on (CAM) and legacy power save (PSM) in ns-3.

| File | What it does |
|---|---|
| `run_sweep.py` | Runs every configuration in parallel, can resume after an interruption, merges the results into `results/all_runs.csv` |
| `analyze.py` | Seed-averaged metrics with 95% CIs, savings paired by seed against CAM and PSM, plots, `report.txt` |
| `sweep.json` | Full grid: 2 traffic types × 5 station counts × (CAM + PSM + 2 TWT variants × 28 settings) × 10 seeds = 5,700 runs |
| `sweep_quick.json` | Small grid for checking that the scenario works |
| `mock_sim.py` | Fake simulator that produces **synthetic** numbers, used only to test the pipeline |

## Quick start

```bash
pip install pandas matplotlib numpy scipy

# 1. test the pipeline (no ns-3 needed)
python3 run_sweep.py --mock --config sweep_quick.json --out mock_results
python3 analyze.py mock_results/all_runs.csv --out mock_figures

# 2. real sweep (scenario lives in <ns-3>/scratch/twt-sim/)
python3 run_sweep.py --ns3-dir ~/ns-3-dev --config sweep_quick.json --dry-run
python3 run_sweep.py --ns3-dir ~/ns-3-dev --config sweep.json --jobs 8
python3 analyze.py results/all_runs.csv --out figures --max-loss 1 --max-delay 100
```

ns-3 should be built in optimized mode for sweeps:
`./ns3 configure -d optimized --enable-modules="wifi;flow-monitor;applications;internet;energy"`.
The sweep runs the program as `./ns3 run twt-sim` (override with `--program`).

## Contract for the ns-3 program (`scratch/twt-sim/twt-sim.cc`)

Command-line arguments (add them with `CommandLine::AddValue`; `RngRun` is built in):

| Argument | Meaning |
|---|---|
| `--mode` | `cam` (no power save), `psm` (legacy PS), or `twt` |
| `--twtEarlySleep` | `1` = doze before SP end once nothing is left to send (sweep mode `twtes`) |
| `--traffic` | `iot` (100 B every 100 ms uplink) or `video` (~2 Mb/s uplink); add more if you like |
| `--nSta` | number of stations |
| `--simTime` | seconds |
| `--RngRun` | seed run number |
| `--wakeIntervalMs`, `--spDurationMs` | TWT agreement (only passed for `twt`) |
| `--outFile` | CSV path to write |

Any `extraArgs` in the config JSON are passed through as `--key=value`.

Output: one row per station with this header, exactly:

```
staId,energyJ,txPackets,rxPackets,rxBytes,meanDelayMs,p95DelayMs,sleepFraction
```

- `energyJ`: `WifiRadioEnergyModel::GetTotalEnergyConsumption()` at the end of the run
- `txPackets`, `rxPackets`, `rxBytes`, `meanDelayMs`: from FlowMonitor, for that station's flow
- `p95DelayMs`: 95th percentile of per-packet delay (enable FlowMonitor's delay histogram, or record delays in a trace sink)
- `sleepFraction`: time in `WifiPhyState::SLEEP` ÷ simTime (track it with the PHY state trace)

Exclude the warm-up period (association, ADDBA, TWT setup) from energy and traffic,
for example by starting applications at 2 s and reading energy from that point.

## Things the analysis checks

`report.txt` warns when:
- TWT runs report almost no sleep: the PHY is never put to sleep, so savings can't show up
- CAM runs report sleep: the baseline is wrong
- a run sent no packets

The headline result is the best TWT saving that **also** stays within the loss and
latency limits you pass in, so it isn't a saving bought by dropping traffic.
