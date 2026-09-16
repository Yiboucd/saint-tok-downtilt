# SAINT-Tok: Digital-Twin-Assisted Joint Downtilt Control with Sector-Aware RL

Evaluation records, figure sources, and code for the paper

> *Digital-Twin-Assisted Joint Downtilt Control with Sector-Aware
> Reinforcement Learning*
> Yibo Ma, Songyang Zhang, Zhi Ding (submitted to IEEE ICC 2027).

SAINT-Tok is a digital-twin-assisted RL controller that assigns each of nine
co-channel sectors a transformer token carrying its own observations
(commanded tilt, served-report statistics, demand-physics aggregates),
coordinates the tokens with self-attention, and is trained with discrete soft
actor--critic on a parameter-free, coverage-gated truncated-Shannon spectral-
efficiency objective. Deployed policy: one 378k-parameter actor, one forward
pass per control step.

## Reproduce every number in the paper — no GPU, no training

All headline results are frozen, per-episode evaluation records
(90 paired episodes, 3 seeds per learned method) stored as JSON in
`results/`. In every record the per-case entries 0–49 are the 50
seen-family episodes and entries 50–89 the 40 held-out-family episodes.
Table I reports dU (bps/Hz) over all 90 episodes, over the seen and the
held-out episodes separately, and the excellent-band demand share at
episode end. To regenerate Table I, the margin-over-constant retention on
held-out demand, and the paired bootstrap confidence intervals quoted in
the text:

```bash
cd tables
python make_table1.py        # needs numpy only
```

Figure sources are in `figures/scripts/` (matplotlib); the rendered figures
used in the paper are in `figures/`.

## Layout

| Path | Contents |
|---|---|
| `results/` | Frozen evaluation records: main 12-policy comparison, mismatch severity ladders, downstream service-quality metrics, reward-parameter sweep, feedback-error and RSRP-noise ladders, ablation evaluations (per-episode values included, so every CI is recomputable) |
| `tables/make_table1.py` | Regenerates Table I (all / seen / held-out dU, excellent share), the held-out margin retention, and the headline CIs from `results/` |
| `figures/`, `figures/scripts/` | Paper figures, presentation figures, and their generation scripts |
| `eval/` | As-run evaluation scripts (paths refer to our cluster; kept verbatim for provenance) |
| `src/` | Controller, baselines, and training runner (reference; training additionally requires the radio-map cache, below) |
| `data/` | Small derived data (uniform-tilt trade-off locus) |

## Robustness records and figures

| File | What it is |
|---|---|
| `results/v12_se_param_sweep.json` | Reward-parameter sweep: the frozen greedy trajectories re-scored under 11 variants of the objective constants; the ranking of all 12 policies is unchanged (Spearman rho = 1.0 vs. the paper objective on every variant) |
| `results/v12_se_feedback_ladder.json` | Feedback-error ladder (nominal physics, degraded observations only): SINR measurement noise sigma = 1 / 2 dB before quantization, reporting resolution coarsened to 2 dB RSRP / 1 dB SINR, and coarse + sigma = 1 dB; the proposed policy retains 98.4–100.1% of its clean gain |
| `results/v12_se_rss_noise_ladder.json` | RSRP measurement-noise ladder: additive Gaussian noise sigma = 1 / 2 / 3 dB on every reported per-sector RSRP before quantization (deterministic per-case tape, identical for all policies; flows through the estimator, the serving-sector grouping and the report statistics) plus a combined rung (RSRP sigma = 2 dB + SINR sigma = 1 dB). Proposed dU = +0.463 / +0.462 / +0.464 / +0.465 vs. +0.464 clean, i.e. it retains 99.6–100.3% of its clean gain, staying ahead of discrete SAC (identity) by +0.080 to +0.085 and of branching DQN by +0.093 to +0.104 on every rung |
| `eval/se_feedback_ladder.py`, `eval/se_rss_noise_ladder.py` | As-run scripts that produced the two ladders (frozen policies, frozen 90 cases) |
| `figures/fig5_bands_se.png`, `figures/scripts/make_fig5_bands_se.py` | Fig. 5: operator-facing quality at episode end for before control / best baseline (discrete SAC) / proposed: demand-weighted SE 1.84 → 2.22 → 2.30 bps/Hz, excellent-band share 10.5% → 19.5% → 22.2% (sources: `results/v12_downstream_all.csv`, `results/v12_se_final_eval.json`, `results/v12_se_sactok3_full.json`) |
| `figures/fig_param_robustness.png`, `figures/fig_feedback_ladder.png`, `figures/fig_feedback_ladder_rsrp.png`, `figures/fig_feedback_ladder_both.png` | Presentation figures for the reward-parameter sweep and the two feedback-error ladders |
| `figures/scripts/make_robustness_ppt_figs.py` | Regenerates the four figures above from `results/` (run from `figures/scripts/`; needs matplotlib and scipy) |

## Heavy assets (not in this repository)

Training end-to-end requires the factorized radio-map cache and the trained
models. These are distributed via the UC Davis Box folder
**`SAINT-Tok_RL_0-20deg`** (link to be added by the author), together with
a SHA-256 manifest of every file:

- the 0–20° radio-map cache (189 single-sector maps, 61 MB),
- the trained checkpoints (27 `final.pt`),
- the demand estimator checkpoint.

## Method / baselines in the comparison

Random, tuned best-constant, flat joint DQN over the 3^9 action space, PPO,
one-step contextual variant (gamma = 0), SAINT with identity tokens, QMIX-style
monotonic mixing, branching DQN (generic and task-tailored encoders),
discrete SAC on identity tokens, Tok3 + double-Q (trainer ablation), and the
proposed Tok3 + discrete SAC. Details and citations in the paper.

## License

Code: MIT. Evaluation records and figures: CC BY 4.0.
