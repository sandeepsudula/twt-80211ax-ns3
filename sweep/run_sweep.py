#!/usr/bin/env python3
"""
Run an ns-3 TWT / power-save parameter sweep and collect every run into one CSV.

Each run calls the ns-3 program with command-line arguments (see README.md for the
contract the C++ scenario must follow), the program writes one CSV row per station
to --outFile, and this script appends the run's parameters to those rows.

Usage:
  python3 run_sweep.py --ns3-dir ~/ns-3-dev --config sweep.json --jobs 8
  python3 run_sweep.py --mock --config sweep_quick.json      # test the pipeline
  python3 run_sweep.py --ns3-dir ~/ns-3-dev --dry-run         # print commands only

Runs whose output file already exists are skipped, so an interrupted sweep can be
restarted with the same command.
"""
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Columns the simulation must write (one row per station).
STATION_FIELDS = [
    "staId", "energyJ", "txPackets", "rxPackets", "rxBytes",
    "meanDelayMs", "p95DelayMs", "sleepFraction",
]
PARAM_FIELDS = [
    "runId", "mode", "traffic", "nSta", "wakeIntervalMs", "spDurationMs",
    "dutyCycle", "run", "simTimeS",
]


def expand_grid(cfg):
    """Turn the config into a list of run-parameter dicts."""
    runs = []
    twt = cfg.get("twt", {})
    min_sp = float(twt.get("minSpMs", 0.0))
    for traffic in cfg["traffic"]:
        for n_sta in cfg["nSta"]:
            for run in range(1, int(cfg["seeds"]) + 1):
                for mode in cfg["modes"]:
                    base = dict(mode=mode, traffic=traffic, nSta=int(n_sta),
                                run=run, simTimeS=cfg["simTimeS"])
                    if mode == "twt":
                        for wi in twt["wakeIntervalMs"]:
                            for duty in twt["dutyCycle"]:
                                sp = round(float(wi) * float(duty), 3)
                                if sp < min_sp:
                                    continue
                                runs.append(dict(base, wakeIntervalMs=float(wi),
                                                 spDurationMs=sp, dutyCycle=float(duty)))
                    else:
                        runs.append(dict(base, wakeIntervalMs="", spDurationMs="",
                                         dutyCycle=""))
    for p in runs:
        p["runId"] = run_id(p)
    return runs


def run_id(p):
    s = f"{p['mode']}_{p['traffic']}_n{p['nSta']}"
    if p["mode"] == "twt":
        s += f"_wi{p['wakeIntervalMs']:g}_sp{p['spDurationMs']:g}"
    return s + f"_r{p['run']}"


def program_args(p, out_file, extra):
    args = [
        f"--mode={p['mode']}",
        f"--traffic={p['traffic']}",
        f"--nSta={p['nSta']}",
        f"--simTime={p['simTimeS']}",
        f"--RngRun={p['run']}",
        f"--outFile={out_file}",
    ]
    if p["mode"] == "twt":
        args += [f"--wakeIntervalMs={p['wakeIntervalMs']:g}",
                 f"--spDurationMs={p['spDurationMs']:g}"]
    args += [f"--{k}={v}" for k, v in extra.items()]
    return args


def build_command(opts, p, out_file, extra):
    pargs = program_args(p, out_file, extra)
    if opts.mock:
        return [sys.executable, str(HERE / "mock_sim.py")] + pargs, None
    ns3 = Path(opts.ns3_dir).expanduser().resolve()
    target = " ".join([opts.program] + [shlex.quote(a) for a in pargs])
    return [str(ns3 / "ns3"), "run", "--no-build", target], str(ns3)


def run_one(opts, p, raw_dir, log_dir, extra):
    out_file = (raw_dir / f"{p['runId']}.csv").resolve()
    if out_file.exists() and out_file.stat().st_size > 0:
        return p, "skipped", 0.0
    cmd, cwd = build_command(opts, p, out_file, extra)
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=opts.timeout)
    except subprocess.TimeoutExpired:
        (log_dir / f"{p['runId']}.log").write_text("TIMEOUT\n")
        return p, "timeout", time.time() - t0
    dt = time.time() - t0
    if res.returncode != 0 or not out_file.exists():
        (log_dir / f"{p['runId']}.log").write_text(
            " ".join(cmd) + "\n\n--- stdout ---\n" + res.stdout +
            "\n--- stderr ---\n" + res.stderr)
        if out_file.exists():
            out_file.unlink()  # don't let a partial file be "resumed"
        return p, "failed", dt
    return p, "ok", dt


def collect(runs, raw_dir, out_csv):
    """Merge per-run station CSVs into one file with parameter columns."""
    n_rows, missing = 0, 0
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARAM_FIELDS + STATION_FIELDS)
        w.writeheader()
        for p in runs:
            path = raw_dir / f"{p['runId']}.csv"
            if not path.exists():
                missing += 1
                continue
            with open(path, newline="") as rf:
                reader = csv.DictReader(rf)
                absent = set(STATION_FIELDS) - set(reader.fieldnames or [])
                if absent:
                    raise SystemExit(f"{path} is missing columns: {sorted(absent)}")
                for row in reader:
                    w.writerow({**{k: p[k] for k in PARAM_FIELDS},
                                **{k: row[k] for k in STATION_FIELDS}})
                    n_rows += 1
    return n_rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "sweep.json"))
    ap.add_argument("--ns3-dir", help="path to ns-3 (folder containing ./ns3)")
    ap.add_argument("--program", default="scratch/twt-sim",
                    help="ns-3 program to run (default: scratch/twt-sim)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--jobs", type=int, default=4, help="parallel runs")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per run")
    ap.add_argument("--mock", action="store_true",
                    help="use mock_sim.py instead of ns-3 (pipeline test)")
    ap.add_argument("--dry-run", action="store_true", help="print commands and exit")
    ap.add_argument("--no-build", action="store_true",
                    help="skip './ns3 build' before the sweep")
    opts = ap.parse_args()

    if not opts.mock and not opts.ns3_dir:
        ap.error("--ns3-dir is required unless --mock is given")

    cfg = json.loads(Path(opts.config).read_text())
    extra = cfg.get("extraArgs", {})
    runs = expand_grid(cfg)

    out = Path(opts.out)
    raw_dir, log_dir = out / "raw", out / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))

    if opts.dry_run:
        for p in runs[:20]:
            cmd, cwd = build_command(opts, p, (raw_dir / f"{p['runId']}.csv").resolve(), extra)
            print(f"(cwd={cwd}) " + " ".join(shlex.quote(c) for c in cmd))
        print(f"... {len(runs)} runs total")
        return

    if not opts.mock and not opts.no_build:
        print("Building ns-3 ...", flush=True)
        subprocess.run([str(Path(opts.ns3_dir).expanduser() / "ns3"), "build"],
                       cwd=str(Path(opts.ns3_dir).expanduser()), check=True)

    print(f"{len(runs)} runs, {opts.jobs} in parallel -> {out}/", flush=True)
    counts = {"ok": 0, "skipped": 0, "failed": 0, "timeout": 0}
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        futs = [pool.submit(run_one, opts, p, raw_dir, log_dir, extra) for p in runs]
        for i, fut in enumerate(as_completed(futs), 1):
            p, status, dt = fut.result()
            counts[status] += 1
            if status in ("failed", "timeout"):
                print(f"  {status.upper()}: {p['runId']} (see {log_dir}/{p['runId']}.log)",
                      flush=True)
            if i % 50 == 0 or i == len(runs):
                el = time.time() - t_start
                print(f"  [{i}/{len(runs)}] ok={counts['ok']} skipped={counts['skipped']} "
                      f"failed={counts['failed'] + counts['timeout']}  {el / 60:.1f} min",
                      flush=True)

    n_rows, missing = collect(runs, raw_dir, out / "all_runs.csv")
    print(f"Wrote {out / 'all_runs.csv'} ({n_rows} station rows; {missing} runs missing)")
    if counts["failed"] or counts["timeout"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
