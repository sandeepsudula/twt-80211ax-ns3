"""
TwtEnv: the "environment" for the DQN.

One step = one short run of twt-sim:
  1. the environment describes a home network (the context / state)
  2. the agent picks TWT settings (the action)
  3. twt-sim runs with those settings and writes its per-station CSV
  4. the CSV is turned into a single number (the reward)

Each simulation is independent, so every decision is a one-step episode
(a contextual bandit). The DQN target is therefore just the reward.
"""

import csv
import glob
import math
import os
import random
import subprocess
import tempfile

import numpy as np

# ---- the actions: always-on, or one TWT (wake interval, service period) pair
WAKE_INTERVALS_MS = [10, 20, 50, 100, 200, 500]
SP_DURATIONS_MS = [2, 5, 10, 20]
ACTIONS = [("cam", 0, 0)] + [
    ("twt", wi, sp) for wi in WAKE_INTERVALS_MS for sp in SP_DURATIONS_MS if sp < wi
]

CAM_POWER_W = 0.82  # idle radio power in ns-3's default energy model (0.273 A x 3 V)
STATE_SIZE = 4


def action_name(a):
    mode, wi, sp = ACTIONS[a]
    return "cam" if mode == "cam" else f"twt {wi}/{sp} ms"


class TwtEnv:
    def __init__(self, ns3_dir, sim_time=4.0, delay_target_ms=100.0, w_delay=2.0, w_loss=5.0,
                 max_load_mbps=20.0, seed=1):
        self.ns3_dir = os.path.abspath(os.path.expanduser(ns3_dir))
        found = glob.glob(os.path.join(self.ns3_dir, "build", "scratch", "twt-sim", "*twt-sim*"))
        if not found:
            raise FileNotFoundError("twt-sim is not built: run ./ns3 build twt-sim in " + self.ns3_dir)
        self.binary = found[0]
        self.sim_time = sim_time
        self.delay_target_ms = delay_target_ms
        self.w_delay = w_delay
        self.w_loss = w_loss
        self.max_load_mbps = max_load_mbps
        self.rng = random.Random(seed)
        self.run_counter = 0
        lib = os.path.join(self.ns3_dir, "build", "lib")
        self.proc_env = dict(os.environ)
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            self.proc_env[var] = lib + os.pathsep + self.proc_env.get(var, "")

    # ---- context: a random home network, kept below the channel's capacity
    def sample_context(self):
        while True:
            n_sta = self.rng.randint(2, 30)
            pkt_size = self.rng.choice([100, 500, 1200])
            pkt_interval_ms = math.exp(self.rng.uniform(math.log(10), math.log(500)))
            load_mbps = n_sta * pkt_size * 8 / pkt_interval_ms / 1000
            if load_mbps <= self.max_load_mbps:
                return {"nSta": n_sta, "pktSize": pkt_size,
                        "pktIntervalMs": round(pkt_interval_ms, 2), "loadMbps": load_mbps}

    def state_of(self, ctx):
        """Context -> 4 numbers between about 0 and 1 (what the neural network sees)."""
        return np.array([
            ctx["nSta"] / 30.0,
            math.log(ctx["pktIntervalMs"] / 10.0) / math.log(50.0),
            ctx["pktSize"] / 1200.0,
            ctx["loadMbps"] / self.max_load_mbps,
        ], dtype=np.float32)

    # ---- one simulation
    def run(self, ctx, a, run_number=None):
        mode, wi, sp = ACTIONS[a]
        if run_number is None:
            self.run_counter += 1
            run_number = self.run_counter
        fd, out = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        args = [self.binary, f"--mode={mode}", f"--nSta={ctx['nSta']}",
                f"--pktSize={ctx['pktSize']}", f"--pktIntervalMs={ctx['pktIntervalMs']}",
                f"--simTime={self.sim_time}", f"--RngRun={run_number}", f"--outFile={out}"]
        if mode == "twt":
            args += [f"--wakeIntervalMs={wi}", f"--spDurationMs={sp}"]
        res = subprocess.run(args, cwd=self.ns3_dir, env=self.proc_env,
                             capture_output=True, text=True)
        if res.returncode != 0:
            os.remove(out)
            raise RuntimeError("twt-sim failed: " + " ".join(args) + "\n" + res.stderr[-2000:])
        with open(out) as f:
            rows = list(csv.DictReader(f))
        os.remove(out)
        return self.score(rows)

    # ---- CSV rows -> metrics and reward
    def score(self, rows):
        energy = np.array([float(r["energyJ"]) for r in rows])
        tx = sum(int(r["txPackets"]) for r in rows)
        rx = sum(int(r["rxPackets"]) for r in rows)
        p95 = np.array([float(r["p95DelayMs"]) for r in rows])
        mean_delay = np.array([float(r["meanDelayMs"]) for r in rows])
        power_w = energy.mean() / self.sim_time               # average power per station
        violation = float((p95 > self.delay_target_ms).mean())  # share of stations too slow
        loss = 1.0 - rx / tx if tx else 0.0
        reward = -(power_w / CAM_POWER_W + self.w_delay * violation + self.w_loss * loss)
        return {"reward": reward, "powerW": power_w, "violation": violation, "loss": loss,
                "meanDelayMs": float(mean_delay.mean()), "p95DelayMs": float(p95.max()),
                "sleep": float(np.mean([float(r["sleepFraction"]) for r in rows]))}
