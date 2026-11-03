"""
Train the DQN on twt-sim.

  python train.py --ns3-dir ~/ns-3.48 --steps 2000 --workers 8
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor

from dqn import DQNAgent
from env import ACTIONS, STATE_SIZE, TwtEnv, action_name

p = argparse.ArgumentParser()
p.add_argument("--ns3-dir", default="~/ns-3.48")
p.add_argument("--steps", type=int, default=2000, help="number of simulations to learn from")
p.add_argument("--workers", type=int, default=4, help="simulations run in parallel")
p.add_argument("--sim-time", type=float, default=4.0)
p.add_argument("--delay-target", type=float, default=None,
               help="fix the p95 delay target (ms); default: random per network")
p.add_argument("--updates-per-step", type=int, default=2)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--model", default="dqn_twt.pt")
p.add_argument("--log", default="train_log.csv")
args = p.parse_args()

env = TwtEnv(args.ns3_dir, sim_time=args.sim_time, delay_target_ms=args.delay_target,
             seed=args.seed)
agent = DQNAgent(STATE_SIZE, len(ACTIONS), eps_decay_steps=int(0.75 * args.steps),
                 seed=args.seed)
print(f"{len(ACTIONS)} actions, simulator: {env.binary}")

log = open(args.log, "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "nSta", "pktSize", "pktIntervalMs", "loadMbps", "targetMs", "action", "reward",
                 "powerW", "violation", "loss", "sleep", "epsilon"])

t0 = time.time()
recent = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    while agent.steps < args.steps:
        # one round: `workers` different home networks, decided and simulated in parallel
        ctxs = [env.sample_context() for _ in range(args.workers)]
        states = [env.state_of(c) for c in ctxs]
        acts = [agent.act(s) for s in states]
        # give each simulation its own RngRun here, in the main thread (not inside the workers)
        runs = list(range(env.run_counter + 1, env.run_counter + 1 + len(ctxs)))
        env.run_counter += len(ctxs)
        results = list(pool.map(lambda car: env.run(*car), zip(ctxs, acts, runs)))

        for c, s, a, res in zip(ctxs, states, acts, results):
            agent.remember(s, a, res["reward"], s, True)   # one-step episode
            for _ in range(args.updates_per_step):
                agent.learn()
            recent.append(res["reward"])
            writer.writerow([agent.steps, c["nSta"], c["pktSize"], c["pktIntervalMs"],
                             round(c["loadMbps"], 3), c["targetMs"], action_name(a), round(res["reward"], 4),
                             round(res["powerW"], 4), res["violation"], round(res["loss"], 4),
                             round(res["sleep"], 4), round(agent.epsilon(), 3)])
        if agent.steps % (10 * args.workers) < args.workers:
            avg = sum(recent[-100:]) / len(recent[-100:])
            print(f"step {agent.steps:5d}  eps {agent.epsilon():.2f}  "
                  f"avg reward (last 100) {avg:+.3f}  {time.time() - t0:6.0f} s")
            log.flush()

agent.save(args.model)
log.close()
print(f"saved {args.model} after {agent.steps} simulations in {time.time() - t0:.0f} s")
