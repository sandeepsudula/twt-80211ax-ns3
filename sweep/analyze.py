#!/usr/bin/env python3
"""
Analyze a TWT / power-save sweep: per-run metrics, seed-averaged summary with 95%
confidence intervals, paired energy savings vs CAM and legacy PSM, and plots.

Usage:
  python3 analyze.py results/all_runs.csv --out figures
  python3 analyze.py results/all_runs.csv --nsta-ref 20 --max-loss 1 --max-delay 100

Savings are paired by seed: each TWT run is compared against the CAM/PSM run with
the same traffic, station count and RngRun, then averaged.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:          # fall back to a t-table
    stats = None
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}

CFG = ["mode", "traffic", "nSta", "wakeIntervalMs", "spDurationMs", "dutyCycle"]
METRICS = ["energyPerStaJ", "nJPerBit", "throughputMbps", "lossPct",
           "meanDelayMs", "p95DelayMs", "sleepFraction",
           "savingsVsCamPct", "savingsVsPsmPct"]


def t975(df):
    if stats is not None:
        return stats.t.ppf(0.975, df)
    keys = sorted(_T975)
    return _T975[min(keys, key=lambda k: abs(k - df))] if df <= 30 else 1.96


def ci95(x):
    x = pd.Series(x).dropna()
    n = len(x)
    return 0.0 if n < 2 else t975(n - 1) * x.std(ddof=1) / np.sqrt(n)


def per_run(df):
    for c in ["wakeIntervalMs", "spDurationMs", "dutyCycle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["delayXrx"] = df["meanDelayMs"] * df["rxPackets"]
    g = df.groupby(CFG + ["run", "simTimeS"], as_index=False).agg(
        energyJ=("energyJ", "sum"),
        txPackets=("txPackets", "sum"),
        rxPackets=("rxPackets", "sum"),
        rxBytes=("rxBytes", "sum"),
        delayXrx=("delayXrx", "sum"),
        p95DelayMs=("p95DelayMs", "max"),         # worst station's 95th percentile
        sleepFraction=("sleepFraction", "mean"),
    )
    g["energyPerStaJ"] = g["energyJ"] / g["nSta"]
    g["throughputMbps"] = g["rxBytes"] * 8 / g["simTimeS"] / 1e6
    g["nJPerBit"] = g["energyJ"] / (g["rxBytes"] * 8).replace(0, np.nan) * 1e9
    g["lossPct"] = 100 * (1 - g["rxPackets"] / g["txPackets"].replace(0, np.nan))
    g["meanDelayMs"] = g["delayXrx"] / g["rxPackets"].replace(0, np.nan)

    pair = ["traffic", "nSta", "run"]
    for base, col in [("cam", "savingsVsCamPct"), ("psm", "savingsVsPsmPct")]:
        ref = g[g["mode"] == base][pair + ["energyPerStaJ"]].rename(
            columns={"energyPerStaJ": "_ref"})
        g = g.merge(ref, on=pair, how="left")
        g[col] = 100 * (1 - g["energyPerStaJ"] / g["_ref"])
        g = g.drop(columns="_ref")
    return g


def summarize(runs):
    agg = {}
    for m in METRICS:
        agg[m] = (m, "mean")
        agg[m + "_ci"] = (m, ci95)
    s = runs.groupby(CFG, as_index=False).agg(**agg, seeds=("run", "nunique"))
    return s.sort_values(CFG).reset_index(drop=True)


# ----------------------------------------------------------------------------- plots
def _baseline(s, traffic, n_sta, mode, col):
    r = s[(s.traffic == traffic) & (s.nSta == n_sta) & (s["mode"] == mode)]
    return None if r.empty else (r[col].iloc[0], r[col + "_ci"].iloc[0])


def plot_vs_interval(s, traffic, n_sta, col, ylabel, fname, out, logy=False):
    t = s[(s.traffic == traffic) & (s.nSta == n_sta) & (s["mode"] == "twt")]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for duty, grp in t.groupby("dutyCycle"):
        grp = grp.sort_values("wakeIntervalMs")
        ax.errorbar(grp.wakeIntervalMs, grp[col], yerr=grp[col + "_ci"],
                    marker="o", capsize=3, label=f"TWT, SP = {duty:.0%} of interval")
    for mode, style in [("psm", "--"), ("cam", ":")]:
        b = _baseline(s, traffic, n_sta, mode, col)
        if b is not None and not np.isnan(b[0]):
            ax.axhline(b[0], ls=style, color="gray", label=f"{mode.upper()}")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("TWT wake interval (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{traffic} traffic, {n_sta} stations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def pareto_front(x, y):
    order = np.argsort(x)
    front, best = [], np.inf
    for i in order:
        if y[i] < best:
            front.append(i)
            best = y[i]
    return front


def plot_pareto(s, traffic, n_sta, out):
    sub = s[(s.traffic == traffic) & (s.nSta == n_sta)]
    t = sub[sub["mode"] == "twt"]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sc = ax.scatter(t.meanDelayMs, t.energyPerStaJ, c=np.log10(t.wakeIntervalMs),
                    s=20 + 120 * t.dutyCycle, cmap="viridis", alpha=0.85,
                    label="TWT (size = duty cycle)")
    cb = fig.colorbar(sc, ax=ax)
    ticks = sorted(t.wakeIntervalMs.unique())
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([f"{v:g}" for v in ticks])
    cb.set_label("wake interval (ms)")
    pts = sub[["meanDelayMs", "energyPerStaJ"]].dropna().to_numpy()
    f = pareto_front(pts[:, 0], pts[:, 1])
    fp = pts[f]
    ax.step(fp[:, 0], fp[:, 1], where="post", color="k", lw=1, alpha=0.6,
            label="Pareto front")
    for mode, mk in [("psm", "s"), ("cam", "^")]:
        r = sub[sub["mode"] == mode]
        if not r.empty:
            ax.scatter(r.meanDelayMs, r.energyPerStaJ, marker=mk, s=90,
                       edgecolor="k", facecolor="white", label=mode.upper(), zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("mean latency (ms)")
    ax.set_ylabel("energy per station (J)")
    ax.set_title(f"Energy-latency trade-off: {traffic}, {n_sta} stations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"pareto_{traffic}_n{n_sta}.png", dpi=150)
    plt.close(fig)


def plot_density(s, traffic, out):
    sub = s[s.traffic == traffic]
    t = sub[sub["mode"] == "twt"]
    if t.empty or sub.nSta.nunique() < 2:
        return
    wis = sorted(t.wakeIntervalMs.unique())
    duties = sorted(t.dutyCycle.unique())
    duty = duties[len(duties) // 2]
    picks = sorted({wis[0], wis[len(wis) // 2], wis[-1]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for wi in picks:
        g = t[(t.wakeIntervalMs == wi) & (t.dutyCycle == duty)].sort_values("nSta")
        lab = f"TWT {wi:g} ms, SP {duty:.0%}"
        axes[0].errorbar(g.nSta, g.savingsVsCamPct, yerr=g.savingsVsCamPct_ci,
                         marker="o", capsize=3, label=lab)
        axes[1].errorbar(g.nSta, g.lossPct, yerr=g.lossPct_ci, marker="o",
                         capsize=3, label=lab)
    p = sub[sub["mode"] == "psm"].sort_values("nSta")
    if not p.empty:
        axes[0].errorbar(p.nSta, p.savingsVsCamPct, yerr=p.savingsVsCamPct_ci,
                         marker="s", ls="--", color="gray", capsize=3, label="PSM")
        axes[1].errorbar(p.nSta, p.lossPct, yerr=p.lossPct_ci, marker="s", ls="--",
                         color="gray", capsize=3, label="PSM")
    axes[0].set_ylabel("energy savings vs CAM (%)")
    axes[1].set_ylabel("packet loss (%)")
    for ax in axes:
        ax.set_xlabel("number of stations")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Effect of station density: {traffic} traffic")
    fig.tight_layout()
    fig.savefig(out / f"density_{traffic}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def headline(s, max_loss, max_delay):
    lines = []
    for traffic in sorted(s.traffic.unique()):
        t = s[(s.traffic == traffic) & (s["mode"] == "twt")]
        if t.empty:
            continue
        ok = t[(t.lossPct <= max_loss) & (t.meanDelayMs <= max_delay)]
        lines.append(f"\n[{traffic}]")
        best_any = t.loc[t.savingsVsCamPct.idxmax()]
        lines.append("  Max savings vs CAM (no constraints): "
                     f"{best_any.savingsVsCamPct:.1f}% ± {best_any.savingsVsCamPct_ci:.1f} "
                     f"(n={best_any.nSta:.0f}, interval {best_any.wakeIntervalMs:g} ms, "
                     f"SP {best_any.spDurationMs:g} ms, loss {best_any.lossPct:.2f}%, "
                     f"latency {best_any.meanDelayMs:.1f} ms)")
        if ok.empty:
            lines.append(f"  No TWT setting meets loss <= {max_loss}% and "
                         f"latency <= {max_delay} ms")
        else:
            b = ok.loc[ok.savingsVsCamPct.idxmax()]
            lines.append(f"  Max savings vs CAM with loss <= {max_loss}% and latency <= "
                         f"{max_delay} ms: {b.savingsVsCamPct:.1f}% ± "
                         f"{b.savingsVsCamPct_ci:.1f} (n={b.nSta:.0f}, interval "
                         f"{b.wakeIntervalMs:g} ms, SP {b.spDurationMs:g} ms); vs PSM: "
                         f"{b.savingsVsPsmPct:.1f}%")
        p = s[(s.traffic == traffic) & (s["mode"] == "psm")]
        if not p.empty:
            lines.append(f"  Legacy PSM savings vs CAM: "
                         f"{p.savingsVsCamPct.min():.1f}% to {p.savingsVsCamPct.max():.1f}% "
                         f"across station counts")
    return "\n".join(lines)


def sanity_checks(runs):
    warn = []
    twt = runs[runs["mode"] == "twt"]
    if not twt.empty and twt.sleepFraction.mean() < 0.01:
        warn.append("TWT runs report almost no sleep time: the PHY is probably never "
                    "put to sleep, so the energy model can't show savings.")
    for base in ["cam", "psm"]:
        if base not in runs["mode"].unique():
            warn.append(f"No {base.upper()} runs found; savings vs {base.upper()} are empty.")
    cam = runs[runs["mode"] == "cam"]
    if not cam.empty and cam.sleepFraction.max() > 0.01:
        warn.append("CAM runs report sleep time; check the CAM baseline isn't using power save.")
    if (runs.txPackets == 0).any():
        warn.append("Some runs sent no packets; check traffic start/stop times.")
    return warn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="all_runs.csv from run_sweep.py")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--nsta-ref", type=int, default=None,
                    help="station count for interval/Pareto plots (default: middle value)")
    ap.add_argument("--max-loss", type=float, default=1.0, help="%% loss for headline")
    ap.add_argument("--max-delay", type=float, default=100.0, help="ms latency for headline")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = per_run(pd.read_csv(a.csv))
    s = summarize(runs)
    runs.to_csv(out / "per_run.csv", index=False)
    s.to_csv(out / "summary.csv", index=False)

    n_vals = sorted(s.nSta.unique())
    n_ref = a.nsta_ref if a.nsta_ref is not None else n_vals[len(n_vals) // 2]
    for traffic in sorted(s.traffic.unique()):
        plot_vs_interval(s, traffic, n_ref, "savingsVsCamPct", "energy savings vs CAM (%)",
                         f"savings_vs_interval_{traffic}_n{n_ref}.png", out)
        plot_vs_interval(s, traffic, n_ref, "meanDelayMs", "mean latency (ms)",
                         f"latency_vs_interval_{traffic}_n{n_ref}.png", out, logy=True)
        plot_vs_interval(s, traffic, n_ref, "lossPct", "packet loss (%)",
                         f"loss_vs_interval_{traffic}_n{n_ref}.png", out)
        plot_vs_interval(s, traffic, n_ref, "throughputMbps", "aggregate throughput (Mb/s)",
                         f"throughput_vs_interval_{traffic}_n{n_ref}.png", out)
        plot_pareto(s, traffic, n_ref, out)
        plot_density(s, traffic, out)

    report = [f"Runs: {len(runs)}   configurations: {len(s)}   "
              f"seeds per config: {int(s.seeds.min())}-{int(s.seeds.max())}",
              headline(s, a.max_loss, a.max_delay)]
    warns = sanity_checks(runs)
    if warns:
        report.append("\nWarnings:\n" + "\n".join(f"  - {w}" for w in warns))
    text = "\n".join(report)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    print(f"\nFigures and tables in {out}/")


if __name__ == "__main__":
    main()
