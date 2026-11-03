"""
Compare the trained DQN with fixed policies and with the best possible choice (oracle).

  python evaluate.py --ns3-dir ~/ns-3.48 --contexts 30 --workers 8
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dqn import DQNAgent
from env import ACTIONS, STATE_SIZE, TwtEnv, action_name

p = argparse.ArgumentParser()
p.add_argument("--ns3-dir", default="~/ns-3.48")
p.add_argument("--model", default="dqn_twt.pt")
p.add_argument("--contexts", type=int, default=30, help="test networks (each needs 22 runs)")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--sim-time", type=float, default=4.0)
p.add_argument("--delay-target", type=float, default=None)
p.add_argument("--seed", type=int, default=12345, help="different from training")
args = p.parse_args()

env = TwtEnv(args.ns3_dir, sim_time=args.sim_time, delay_target_ms=args.delay_target,
             seed=args.seed)
agent = DQNAgent(STATE_SIZE, len(ACTIONS))
agent.load(args.model)

ctxs = [env.sample_context() for _ in range(args.contexts)]
jobs = [(i, a) for i in range(len(ctxs)) for a in range(len(ACTIONS))]
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    # every action on every test network, same random run number per network (fair comparison)
    out = list(pool.map(lambda job: env.run(ctxs[job[0]], job[1], run_number=100000 + job[0]),
                        jobs))
table = {}  # (context index, action) -> metrics
for (i, a), res in zip(jobs, out):
    table[(i, a)] = res

default_twt = ACTIONS.index(("twt", 100, 10))
avg_reward = [np.mean([table[(i, a)]["reward"] for i in range(len(ctxs))])
              for a in range(len(ACTIONS))]
best_fixed = int(np.argmax(avg_reward))

policies = {
    "always on (cam)": lambda i: 0,
    "fixed twt 100/10": lambda i: default_twt,
    f"best fixed ({action_name(best_fixed)})": lambda i: best_fixed,
    "DQN": lambda i: agent.act(env.state_of(ctxs[i]), greedy=True),
    "oracle (best per network)": lambda i: max(range(len(ACTIONS)),
                                                key=lambda a: table[(i, a)]["reward"]),
}

print(f"{len(ctxs)} test networks (delay target varies per network)\n")
print(f"{'policy':34s} {'reward':>8s} {'power W':>8s} {'viol %':>7s} {'loss %':>7s} {'sleep %':>8s}")
for name, pick in policies.items():
    rows = [table[(i, pick(i))] for i in range(len(ctxs))]
    print(f"{name:34s} {np.mean([r['reward'] for r in rows]):8.3f} "
          f"{np.mean([r['powerW'] for r in rows]):8.3f} "
          f"{100 * np.mean([r['violation'] for r in rows]):7.1f} "
          f"{100 * np.mean([r['loss'] for r in rows]):7.2f} "
          f"{100 * np.mean([r['sleep'] for r in rows]):8.1f}")

match = np.mean([policies["DQN"](i) == policies["oracle (best per network)"](i)
                 for i in range(len(ctxs))])
print(f"\nDQN picked the oracle's action in {100 * match:.0f}% of networks")
print("\nper-network choices (DQN vs oracle):")
for i, c in enumerate(ctxs):
    print(f"  target {c['targetMs']:3d} ms  nSta={c['nSta']:2d} {c['pktSize']:4d} B every {c['pktIntervalMs']:6.1f} ms "
          f"-> DQN {action_name(policies['DQN'](i)):15s} oracle "
          f"{action_name(policies['oracle (best per network)'](i))}")
