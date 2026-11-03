# DQN for 802.11ax TWT configuration

A Deep Q-Network learns which TWT settings (wake interval, service period) to use for a home
Wi-Fi network, trading station energy against delay. The ns-3 program `twt-sim` is the
environment: every decision is checked by running one short simulation.

## Problem

- **State (context):** the network the AP has to serve, as 5 numbers between about 0 and 1:
  number of stations (2–20), packet interval (2–500 ms), packet size (100/500/1200 B),
  offered load (≤ 20 Mbit/s), and the application's delay target (20/50/100/200/500 ms).
- **Action:** one of 22 choices: always on (`cam`), or TWT with
  wake interval ∈ {10, 20, 50, 100, 200, 500} ms and service period ∈ {2, 5, 10, 20} ms
  (service period shorter than the wake interval).
- **Reward:**

      reward = −( power / 0.82 W  +  2 × (share of stations whose p95 delay > target)  +  5 × loss )

  `0.82 W` is the always-on idle power, so the first term is 1.0 for `cam` and about 0.2 for
  TWT 100/10. The delay target is drawn per network and is part of the state
  (`--delay-target 100` fixes it instead).
- **Why the delay target is in the state:** with one fixed target, a single TWT setting was best
  for every network (see `runs/v1_fixed_target/`), so there was nothing for the DQN to learn. With a
  per-network target, the best wake interval changes from network to network.
- Each simulation is independent, so each decision is a one-step episode (a contextual
  bandit). The DQN target is the reward itself (`gamma` has no effect).

## Files

| File | What it does |
|---|---|
| `env.py` | Samples a network, runs `twt-sim` with the chosen settings, turns its CSV into a reward |
| `dqn.py` | Q-network (2 hidden layers of 128), replay memory, epsilon-greedy, target network |
| `train.py` | Training loop, runs several simulations in parallel |
| `evaluate.py` | Compares the DQN with fixed policies and with the oracle (best action per network) |
| `oracle_check.py` | Checks that the best action really changes between networks (run this before training) |
| `env_v1.py` | First design with one fixed delay target (kept for comparison) |
| `runs/v2_delay_target/` | Trained model, training log and evaluation output of the current design |
| `runs/v1_fixed_target/` | Same for the first design |

## Simulator fixes (already applied in `ns3/scratch/twt-sim/twt-sim.cc`)

1. Battery `1e9` J → `1e6` J. With 1e9 J, ns-3's time-until-empty calculation overflows while
   the radio sleeps and `psm` / `twt` runs abort.
2. `SleepTracker`: after `Close()`, later reports are ignored (a `closed` flag), so the last
   sleep before the window end is not counted twice.

## Setup

    cp -r ns3/scratch/twt-sim ~/ns-3.48/scratch/
    cd ~/ns-3.48 && ./ns3 build twt-sim
    cd <repo>/dqn && pip install -r requirements.txt

## Run

    python oracle_check.py ~/ns-3.48 10      # best action for 10 random networks
    python train.py --ns3-dir ~/ns-3.48 --steps 2000 --workers 8
    python evaluate.py --ns3-dir ~/ns-3.48 --contexts 30 --workers 8

`--workers` should be about the number of CPU cores. Training writes `dqn_twt.pt` (the model)
and `train_log.csv` (every decision and its result).
To evaluate the model trained in the test run without retraining:

    python evaluate.py --ns3-dir ~/ns-3.48 --model runs/v2_delay_target/dqn_twt.pt

## Results from the test run (2 CPU cores, 2000 training simulations, ~28 min)

Evaluated on 30 networks not seen in training (each network: all 22 actions simulated, same
random run, so the comparison is fair):

| Policy | Reward | Power per station | Delay violations | Sleep |
|---|---|---|---|---|
| Always on (cam) | −1.017 | 0.826 W | 0.5% | 0% |
| Fixed TWT 100/10 | −0.877 | 0.172 W | 33.3% | 90% |
| Best single fixed setting (TWT 20/2) | −0.211 | 0.173 W | 0% | 90% |
| **DQN** | **−0.165** | **0.135 W** | **0%** | **95%** |
| Oracle (best action per network) | −0.151 | 0.124 W | 0% | 97% |

- The DQN closes about 77% of the gap between the best fixed setting and the oracle, uses
  22% less power than the best fixed setting and 84% less than always-on, with no delay
  violations.
- It is conservative: it often picks a shorter wake interval than the target allows (e.g.
  100 ms when 200 ms would do). Longer training or a finer reward should close more of the gap.
- `runs/v1_fixed_target/` holds the first design (one fixed 100 ms target), where a single setting
  (TWT 100/2) was optimal for every network, so a DQN added nothing.
