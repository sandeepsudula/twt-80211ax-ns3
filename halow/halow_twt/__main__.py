"""
Command line for the HaLow bench.

  python -m halow_twt actions                                   # the DQN's 22 actions, checked
                                                                # against the hardware limits
  python -m halow_twt status --config bench.json                # what the AP has installed
  python -m halow_twt apply  --config bench.json --action 21    # run one DQN action
  python -m halow_twt apply  --config bench.json --schedule cfg.conf
  python -m halow_twt deploy --config bench.json --model ../dqn/runs/v2_delay_target/dqn_twt.pt \\
                             --n-sta 4 --pkt-interval-ms 100 --pkt-size 500 --target-ms 500

Add --dry-run to any of them to use a simulated AP instead of ssh.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

from .actions import DQN_ACTIONS, dqn_action_hardware_notes, dqn_action_name
from .wrapper import HaLowWrapper, load_config, read_schedule_file

REPO_DQN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dqn")


def cmd_actions(_args):
    for i in range(len(DQN_ACTIONS)):
        notes = dqn_action_hardware_notes(i)
        print(f"{i:2d}  {dqn_action_name(i):16s} {'OK on the bench' if not notes else '; '.join(notes)}")


def _wrapper(args):
    return HaLowWrapper(load_config(args.config), dry_run=args.dry_run, verbose=True)


def cmd_status(args):
    env = _wrapper(args)
    agreements, stations = env.ap.agreements(), env.ap.stations()
    for st in env.config.stations:
        a, s = agreements.get(st["mac"]), stations.get(st["mac"], {})
        sched = (f"wake {a.wake_us} us, interval {a.interval_us} us, duration {a.duration_us} us"
                 if a else "no agreement")
        assoc = "associated" if st["mac"] in stations else "NOT associated"
        print(f"{st['name'] or st['mac']:8s} {assoc:15s} {sched}   "
              f"signal {s.get('signal_dbm')} dBm, rx packets {s.get('rx_packets')}")


def _print_obs(obs, as_json):
    if as_json:
        print(json.dumps(obs, indent=2))
        return
    print(f"step {obs['step']}, measured over {obs['window_s']:.1f} s")
    for r in obs["stations"]:
        duty = f"{r['duty_cycle']:.3f}" if r["duty_cycle"] is not None else "-"
        power = f"{r['est_power_w']:.3f} W" if r["est_power_w"] is not None else "-"
        print(f"  {r['name'] or r['mac']:8s} wake {r['wake_us']} interval {r['interval_us']} "
              f"duration {r['duration_us']}  duty {duty}  rx {r['rx_packets']} pkts "
              f"({r['rx_kbps']:.1f} kbit/s)  est. power {power}")
    s = obs["summary"]
    print(f"  total {s['total_rx_kbps']:.1f} kbit/s, {s['stations_with_traffic']} stations with "
          f"traffic, mean duty cycle {s['mean_duty_cycle']}")


def _run_one(env, action, as_json):
    env.reset()
    obs, _ = env.step(action)
    _print_obs(obs, as_json)


def cmd_apply(args):
    env = _wrapper(args)
    if args.schedule:
        action = read_schedule_file(args.schedule, env.config.stations)
    elif args.action is not None:
        action = args.action
    else:
        sys.exit("give --action N or --schedule FILE")
    _run_one(env, action, args.json)


def cmd_deploy(args):
    """Let the trained DQN pick the action for a described network, then run it on the bench."""
    sys.path.insert(0, os.path.abspath(args.dqn_dir))
    try:
        import torch
        from dqn import DQNAgent
        from env import ACTIONS, STATE_SIZE, TwtEnv
    except ImportError as e:
        sys.exit(f"deploy needs the dqn/ folder and torch: {e}")
    if [tuple(a) for a in ACTIONS] != [tuple(a) for a in DQN_ACTIONS]:
        sys.exit("dqn/env.py ACTIONS changed; update DQN_ACTIONS in halow_twt/actions.py")

    load_mbps = args.n_sta * args.pkt_size * 8 / args.pkt_interval_ms / 1000
    ctx = {"nSta": args.n_sta, "pktSize": args.pkt_size, "pktIntervalMs": args.pkt_interval_ms,
           "loadMbps": load_mbps, "targetMs": args.target_ms}
    state = TwtEnv.state_of(SimpleNamespace(max_load_mbps=args.max_load_mbps), ctx)

    agent = DQNAgent(STATE_SIZE, len(ACTIONS))
    agent.load(args.model)
    with torch.no_grad():
        q = agent.q(torch.as_tensor(state).unsqueeze(0))[0].tolist()
    ranked = sorted(range(len(q)), key=lambda a: q[a], reverse=True)
    if args.legal_only:
        ranked = [a for a in ranked if not dqn_action_hardware_notes(a)]
        if not ranked:
            sys.exit("no DQN action is inside the bench's tested limits")
    action = ranked[0]
    print(f"DQN picks action {action}: {dqn_action_name(action)} "
          f"(Q = {q[action]:.3f}; its overall favourite is {dqn_action_name(max(range(len(q)), key=q.__getitem__))})")
    for note in dqn_action_hardware_notes(action):
        print("  warning: " + note)
    _run_one(_wrapper(args), action, args.json)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m halow_twt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("actions", help="list the DQN actions and whether the bench can run them")

    def common(sp):
        sp.add_argument("--config", required=True, help="bench JSON (see bench.example.json)")
        sp.add_argument("--dry-run", action="store_true", help="simulated AP, no ssh")
        sp.add_argument("--json", action="store_true", help="print the full observation")

    common(sub.add_parser("status", help="show installed schedules and station stats"))

    sp = sub.add_parser("apply", help="run one action on the bench and measure it")
    common(sp)
    sp.add_argument("--action", type=int, help="DQN action index (see `actions`)")
    sp.add_argument("--schedule", help="file of 'name wake_us interval_us duration_us' lines")

    sp = sub.add_parser("deploy", help="let the trained DQN choose, then run it on the bench")
    common(sp)
    sp.add_argument("--model", required=True)
    sp.add_argument("--dqn-dir", default=REPO_DQN_DIR)
    sp.add_argument("--n-sta", type=int, required=True)
    sp.add_argument("--pkt-interval-ms", type=float, required=True)
    sp.add_argument("--pkt-size", type=int, required=True)
    sp.add_argument("--target-ms", type=float, required=True, help="delay target")
    sp.add_argument("--max-load-mbps", type=float, default=20.0, help="as used in training")
    sp.add_argument("--legal-only", action="store_true",
                    help="choose the best action among those inside the bench's tested limits")

    args = p.parse_args(argv)
    {"actions": cmd_actions, "status": cmd_status, "apply": cmd_apply,
     "deploy": cmd_deploy}[args.cmd](args)


if __name__ == "__main__":
    main()
