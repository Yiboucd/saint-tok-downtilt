# Training histories for paper Fig. 2

These six compact CSV files contain the single-seed training histories used
in the paper's spectral-efficiency training comparison. Each file has 10,000
episodes from seed `20260810`, with horizon `H = 20`.

The source logs were `track_a_v12_se_final/seed20260810/<method>/training_history.csv`,
written by `src/train_track_a_clean_ppo.py`. Only `episode`, `algo` and
`episode_return` are retained here. The source logs have no seed column.

| File | `algo` / run directory `<method>` | Figure label | Method |
|---|---|---|---|
| `hist_se_sactok3_s10.csv` | `saint_sac_tok3` | Proposed (SAC) | SAINT-Tok with discrete SAC |
| `hist_se_naive_s10.csv` | `bdqn_naive` | BDQN | Branching DQN |
| `hist_se_sac_s10.csv` | `saint_sac` | SAC (identity) | Identity-token discrete SAC |
| `hist_se_qmix_s10.csv` | `saint_qmix` | QMIX | Identity-token QMIX |
| `hist_se_id_s10.csv` | `saint_dqn` | SAINT (identity) | Identity-token double-Q |
| `hist_se_ppo_s10.csv` | `saint_ppo` | PPO | Identity-token PPO |

`episode` is the contiguous index from 1 to 10,000. `episode_return` is the
scaled training return `6 * [U(theta_20) - U(theta_0)]`. The figure plots the
trailing 200-episode arithmetic mean of `episode_return / 6` in bps/Hz, at the
episode ending each full window (200 through 10,000). Drawing uses every
10th smoothed point and includes the final point, as in the paper. These are
exploratory training returns, not frozen greedy evaluation results; no
across-seed uncertainty band is shown.

From the repository root, run:

```sh
python figures/scripts/make_se_training_all_fig.py --output-dir /path/to/output
```

The script locates these CSVs relative to itself and works from any working
directory. It requires NumPy and Matplotlib, validates all six sources, and
fails if a file is missing or invalid. It writes only `training_comparison.png`
to the selected directory (default: `outputs/`) and never modifies the CSVs.
