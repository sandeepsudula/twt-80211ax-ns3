#!/usr/bin/env python3
"""
Stand-in for the ns-3 program, for testing the sweep and analysis pipeline only.

It takes the same arguments and writes the same per-station CSV as the real
scenario, using a rough analytical model with ns-3's default WifiRadioEnergyModel
currents. Its numbers are SYNTHETIC: never report them as results.
"""
import argparse
import csv
import random
import zlib

V = 3.0                                   # BasicEnergySource default voltage
I_TX, I_RX, I_IDLE, I_SLEEP = 0.380, 0.313, 0.273, 0.033   # A (ns-3 defaults)

TRAFFIC = {                               # uplink per station
    "iot":   dict(pkt_bytes=100,  rate_pps=10.0),    # 8 kb/s periodic sensor
    "video": dict(pkt_bytes=1200, rate_pps=208.0),   # ~2 Mb/s
}
CAP_BPS = 100e6                           # effective MAC rate for one station
BEACON_MS = 102.4
QUEUE_PKTS = 500


def efficiency(k):                        # contention efficiency with k active STAs
    return 1.0 / (1.0 + 0.02 * max(k - 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cam", "psm", "twt"])
    ap.add_argument("--traffic", default="iot", choices=list(TRAFFIC))
    ap.add_argument("--nSta", type=int, default=10)
    ap.add_argument("--simTime", type=float, default=30.0)
    ap.add_argument("--RngRun", type=int, default=1)
    ap.add_argument("--wakeIntervalMs", type=float, default=100.0)
    ap.add_argument("--spDurationMs", type=float, default=10.0)
    ap.add_argument("--outFile", required=True)
    a, _ = ap.parse_known_args()

    key = f"{a.mode}|{a.traffic}|{a.nSta}|{a.wakeIntervalMs}|{a.spDurationMs}|{a.RngRun}"
    rng = random.Random(zlib.crc32(key.encode()))
    tr = TRAFFIC[a.traffic]
    load_bps = tr["pkt_bytes"] * 8 * tr["rate_pps"]
    T, n = a.simTime, a.nSta
    tx_pkts = int(tr["rate_pps"] * T)
    base_delay = 1.0                                     # ms, uncontended

    if a.mode == "twt":
        wi, sp = a.wakeIntervalMs, a.spDurationMs
        d = sp / wi
        k = max(1.0, n * d)                              # STAs awake together
        rho = k * (load_bps / d) / (CAP_BPS * efficiency(k))
        loss = max(0.0, 1.0 - 1.0 / rho) if rho > 1 else 0.0
        burst = tr["rate_pps"] * wi / 1000.0
        if burst > QUEUE_PKTS:
            loss = max(loss, 1.0 - QUEUE_PKTS / burst)
        wait = (1 - d) * (wi - sp) / 2.0                 # mean wait for the next SP
        delay = wait + base_delay / (1 - min(rho, 0.95))
        p95_factor = 1.9
        awake = min(1.0, d + 0.5 / wi)                   # + wake-up overhead
    else:
        rho = n * load_bps / (CAP_BPS * efficiency(n))
        loss = max(0.0, 1.0 - 1.0 / rho) if rho > 1 else 0.0
        delay = base_delay / (1 - min(rho, 0.95))
        p95_factor = 2.5
        if a.mode == "cam":
            awake = 1.0
        else:                                            # legacy PS: beacons + tail
            awake = min(1.0, 2.0 / BEACON_MS + tr["rate_pps"] * 0.010
                        + 0.4 * min(rho, 1.0))
            delay += 2.0

    with open(a.outFile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["staId", "energyJ", "txPackets", "rxPackets", "rxBytes",
                    "meanDelayMs", "p95DelayMs", "sleepFraction"])
        for sta in range(n):
            jitter = rng.gauss(1.0, 0.04)
            aw = 1.0 if a.mode == "cam" else min(1.0, max(0.0, awake * jitter))
            f_tx = min(aw, load_bps / (CAP_BPS * efficiency(n)))
            energy = V * T * (aw * I_IDLE + f_tx * (I_TX - I_IDLE) + (1 - aw) * I_SLEEP)
            sl = min(0.999, max(0.0, loss * rng.gauss(1.0, 0.1)))
            rx = int(tx_pkts * (1 - sl))
            dly = max(0.1, delay * rng.gauss(1.0, 0.08))
            w.writerow([sta, f"{energy:.6f}", tx_pkts, rx, rx * tr["pkt_bytes"],
                        f"{dly:.4f}", f"{dly * p95_factor:.4f}", f"{1 - aw:.4f}"])


if __name__ == "__main__":
    main()
