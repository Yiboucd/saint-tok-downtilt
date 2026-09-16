# Training histories for Fig. 3 (seed 20260810)

Compact copies of the per-episode training logs behind Fig. 3 of the paper
(training curves under the spectral-efficiency reward, comparison set).

## Origin

Server run directories `track_a_v12_se_final/seed20260810/<method>/training_history.csv`
(one row per training episode, written by `src/train_track_a_clean_ppo.py`).
Only the columns needed to redraw the figure were kept; the full logs
(30-33 columns, ~7 MB each) additionally carried per-episode diagnostics
(demand family, initial/final/best score, tape hashes, wall time, epsilon,
loss, SAC alpha / entropy) that the figure does not use.

| File | `algo` value | Method (paper label) | Run directory `<method>` |
|---|---|---|---|
| `hist_se_sactok3_s10.csv` | `saint_sac_tok3` | proposed SAINT-Tok, Tok3 tokens + discrete SAC | `saint_sac_tok3` |
| `hist_se_naive_s10.csv` | `bdqn_naive` | generic branching DQN (BDQN) | `bdqn_naive` |
| `hist_se_sac_s10.csv` | `saint_sac` | identity-token discrete SAC | `saint_sac` |
| `hist_se_qmix_s10.csv` | `saint_qmix` | identity-token QMIX | `saint_qmix` |
| `hist_se_id_s10.csv` | `saint_dqn` | identity-token double-Q (SAINT, identity tokens) | `saint_dqn` |
| `hist_se_ppo_s10.csv` | `saint_ppo` | identity-token PPO | `saint_ppo` |

## Columns

| Column | Meaning |
|---|---|
| `episode` | training episode index, 1 ... 10000 (each episode = H = 20 control steps) |
| `algo` | trainer tag as logged by the run (see table above) |
| `episode_return` | kappa-scaled training return of the episode, i.e. kappa * [U(theta_20) - U(theta_0)] with kappa = `SE_REWARD_SCALE` = 6.0 (`src/train_track_a_clean_ppo.py`); divide by 6 to obtain the unscaled terminal gain dU in bps/Hz plotted in Fig. 3 |

There is no `seed` column in the source logs (every file is the seed-20260810 run).

Row counts (data rows, excluding the header line): 10 000 in every file,
episodes 1-10000 contiguous, no missing values. Total size of the six files:
2.16 MB.

## How Fig. 3 is drawn

`figures/scripts/make_se_training_all_fig.py` (run from any directory; it
locates this folder relative to itself) plots, for each file, the trailing
200-episode moving average of `episode_return / 6`, placed at the episode
index that ends the window (x = 200 ... 10000), together with the dotted
best-constant reference (dU = +0.1460 bps/Hz, read from
`results/v12_se_final_eval.json`, key `const.dSE`). No across-seed band is
shown; the paper's Fig. 3 describes the same trailing 200-episode mean.

## When does the proposed curve lead?

Computed from these files (numpy, `np.convolve(r, np.ones(200)/200, "valid")`
on `episode_return / 6`):

* The proposed 200-episode moving average is above the moving average of
  **every** baseline at **every** window, i.e. from the first complete window
  (window ending at episode **200**) through the end of training (episode
  10000). Under the strict definition "first episode after which the proposed
  moving average exceeds every baseline's for the rest of training" the
  answer is therefore episode 200 (the earliest episode at which the
  200-episode average is defined).
* The tightest point is the window ending at episode 203, where the proposed
  average leads the best baseline (PPO) by +0.0068 bps/Hz. The margin over the
  best baseline at selected episodes (best baseline in parentheses):

  | window end | proposed | best baseline | margin |
  |---|---|---|---|
  | 200 | 0.124 | 0.115 (SAC) | +0.009 |
  | 500 | 0.224 | 0.181 (SAC) | +0.043 |
  | 1000 | 0.249 | 0.217 (SAC) | +0.032 |
  | 1700 | 0.205 | 0.174 (SAC) | +0.031 |
  | 2500 | 0.290 | 0.227 (SAC) | +0.063 |
  | 4200 | 0.379 | 0.305 (BDQN) | +0.074 |
  | 7200 | 0.383 | 0.342 (SAC) | +0.041 |
  | 10000 | 0.428 | 0.378 (SAC) | +0.050 |

* Related crossings, for reference: the proposed average exceeds the
  **final** level of the best baseline (identity-token SAC, 0.378 at episode
  10000) for the first time at episode 4158 and for good from episode 7366;
  it stays above the best-constant reference (0.146) from episode 218. The
  baselines clear the constant reference for good at episodes 229 (SAC),
  1798 (identity double-Q), 2033 (BDQN), 2632 (QMIX) and 6050 (PPO).
* Final 200-episode averages (episodes 9801-10000): proposed 0.428, SAC
  0.378, BDQN 0.346, QMIX 0.302, identity double-Q 0.293, PPO 0.198 bps/Hz.

The "~2.5k" figure quoted in the working notes for this crossing is not what
the strict definition yields (that gives episode 200); by episode 2500 the
lead has widened to about +0.06 bps/Hz, roughly its level for the rest of
training. The paper text has not been changed.
