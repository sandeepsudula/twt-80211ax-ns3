import sys
from concurrent.futures import ThreadPoolExecutor
from env import ACTIONS, TwtEnv, action_name
env = TwtEnv(sys.argv[1], seed=777)
ctxs = [env.sample_context() for _ in range(int(sys.argv[2]))]
jobs = [(i, a) for i in range(len(ctxs)) for a in range(len(ACTIONS))]
with ThreadPoolExecutor(max_workers=2) as pool:
    res = list(pool.map(lambda j: env.run(ctxs[j[0]], j[1], run_number=5000 + j[0]), jobs))
table = {j: r for j, r in zip(jobs, res)}
for i, c in enumerate(ctxs):
    best = max(range(len(ACTIONS)), key=lambda a: table[(i, a)]["reward"])
    r = table[(i, best)]
    print(f"target {c['targetMs']:3d} ms  nSta={c['nSta']:2d} {c['pktSize']:4d} B / {c['pktIntervalMs']:6.1f} ms "
          f"(load {c['loadMbps']:5.2f} Mb/s) -> best {action_name(best):14s} reward {r['reward']:+.3f} "
          f"power {r['powerW']:.3f} W loss {100*r['loss']:.1f}%")
