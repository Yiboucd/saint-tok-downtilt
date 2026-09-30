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
Concretely, the evaluation cases are 9 demand families × 10 repeats = 90
paired cases, generated in a fixed order by every `eval/se_*.py` script:
per-case index 0–49 are the five training families (10 consecutive cases
per family), index 50–89 the four held-out families. Each case fixes a
population of N = 2000 UEs, of which M = 200 reporting UEs are observed per
control step (the same population, reporter subset and initial tilts for
every policy). Note on key names: in `results/v12_se_final_eval.json` the
entries `proposed_s0/1/2` are the Tok3 + double-Q trainer ablation (seeds
20260810/11/12); the proposed Tok3 + discrete-SAC policy is stored in
`results/v12_se_sactok3_final.json` (dU) and
`results/v12_se_sactok3_full.json` (dU on every rung plus band metrics).
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
| `data/` | Small derived data (uniform-tilt trade-off locus) and `data/training_histories/`: compact per-episode training logs (episode, algo, episode_return; 10 000 rows each) behind Fig. 3, with a README giving their origin and the moving-average crossing analysis |

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
| `results/v12_se_ladder_all.json` | Mismatch severity ladder, rungs `bias_only` (per-sector power bias U(−3,3) dB), `tilt1` (bias + tilt offsets clipped to ±1°) and `tilt2` (bias + tilt offsets up to ±2°), for the seven policy families not covered by `v12_se_severity.json`: random, identity-token PPO / double-Q (`id`) / QMIX / SAC and generic BDQN (`naive`), three seeds each, plus the one-step (`myopic`) variant — 17 entries per rung, each with `dSE` (mean over the 90 cases) and the 90 `per_case` values. Together with `v12_se_severity.json` (constant, tailored-encoder BDQN, Tok3 + double-Q) and `v12_se_sactok3_full.json` (proposed) this is the 11-policy ladder of the robustness section. Seed-mean dU on bias_only / tilt1 / tilt2: SAC +0.386 / +0.368 / +0.346, BDQN +0.362 / +0.346 / +0.324, QMIX +0.340 / +0.314 / +0.305, identity double-Q +0.339 / +0.322 / +0.308, one-step +0.296 / +0.273 / +0.255, PPO +0.275 / +0.258 / +0.251, random −0.000 / +0.007 / +0.010 (QMIX and identity double-Q are within 0.01 of each other and swap order between bias_only and tilt1) |
| `results/v12_se_sactok3_full.json` | Proposed policy (Tok3 + discrete SAC, seeds 20260810/11/12) on the clean rung with band metrics (demand shares excellent / good / fair / poor at episode end, outage share, demand-weighted SE) and on the bias_only / tilt1 / tilt2 rungs (`dSE` + `per_case`). Seed-mean dU: clean +0.4636, bias_only +0.4733, tilt1 +0.4336, tilt2 +0.4068 (102.1% / 93.5% / 87.7% of the clean gain); clean-rung episode-end shares: excellent 22.2%, good 17.6%, fair 50.7%, poor 9.5%; demand-weighted SE 2.304 bps/Hz versus 1.840 before control (+25.2%) |
| `results/v12_se_mmtrain_eval.json` | Domain-randomized training matrix: policies `mm_bias` / `mm_tilt1` / `mm_tilt2` (Tok3 tokens trained with the double-Q trainer at seed 20260810 under, respectively, power-bias-only, bias + ±1° tilt and bias + ±2° tilt randomization) evaluated on the clean / bias_only / tilt1 / tilt2 rungs (`dSE`, `seen`, `ood`, `per_case`). dU rows = training condition, columns = clean / bias_only / tilt1 / tilt2: `mm_bias` +0.420 / +0.412 / +0.398 / +0.375; `mm_tilt1` +0.343 / +0.363 / +0.339 / +0.326; `mm_tilt2` +0.357 / +0.376 / +0.340 / +0.340. The same-seed clean-trained Tok3 + double-Q reference row (`v12_se_final_eval.json` `proposed_s0` for clean, `v12_se_severity.json` `tok3_s10` for the other rungs) is +0.461 / +0.454 / +0.422 / +0.398. The paper's "reduces performance by 20% under the matched ±1° test" is `mm_tilt1` on tilt1 (+0.3393) against that reference on tilt1 (+0.4219): −19.6% |
| `results/v12_se_amix3_eval.json` | Reward-swap check: Tok3 + double-Q policies trained on the earlier alpha-mix objective (seeds 20260810/11/12, keys `amix_s10/11/12`) scored under the SE objective on the clean rung (`dSE`, `seen`, `ood`, `per_case`). dU +0.4261 / +0.3751 / +0.4026, mean +0.4012, versus the SE-trained Tok3 + double-Q seeds (`v12_se_final_eval.json` `proposed_s0/1/2`: +0.4614 / +0.4085 / +0.4303, mean +0.4334): paired difference SE-trained − alpha-mix-trained = +0.0322 bps/Hz, 95% paired bootstrap CI ≈ [+0.011, +0.054] (20 000 case resamples; the CI edges move in the fourth decimal with the RNG draw), SE-trained wins on 60% of the 90 cases |
| `eval/se_20k_eval.py` | As-run script for the 20 000-episode Tok3 + double-Q probe (seed 20260810) on the same 90 cases; its record `v12_se_20k_eval.json` was not recovered and is not in this repository |
| `eval/se_final_eval.py`, `eval/se_newbase_eval.py`, `eval/se_naive_eval.py`, `eval/se_sactok3_eval.py`, `eval/se_sactok3_final.py`, `eval/se_sactok3_full.py`, `eval/se_ladder_all.py`, `eval/se_mmtrain_eval.py`, `eval/se_amix3_eval.py` | Recovered as-run evaluation scripts (cluster paths kept verbatim) that produced, in order, `v12_se_final_eval.json` (random, constant, PPO ×3, one-step, identity double-Q ×3, tailored BDQN ×3, Tok3 + double-Q ×3), `v12_se_newbase_eval.json` (identity SAC ×3, QMIX ×3), `v12_se_naive_eval.json` (generic BDQN ×3), the seed-20260810 Tok3 + SAC probe that `se_sactok3_final.py` merges with seeds 11/12 into `v12_se_sactok3_final.json`, `v12_se_sactok3_full.json`, `v12_se_ladder_all.json`, `v12_se_mmtrain_eval.json` and `v12_se_amix3_eval.json`. All share one frozen case tape (`--eval-repeats 10`, same `derived_seed(..., "eval_case", group, family, 0, repeat)` order) |
| `data/training_histories/`, `figures/scripts/make_se_training_all_fig.py`, `figures/fig_se_training_all.png` | Fig. 3 sources: compact seed-20260810 training logs of the six curves (proposed, BDQN, SAC, QMIX, identity double-Q, PPO) and the script that redraws the trailing 200-episode means with the best-constant reference read from `results/v12_se_final_eval.json`; see `data/training_histories/README.md` for the moving-average crossing analysis |

## Heavy assets (not in this repository)

### Radio-map cache on Box

[Open the 0–20° radio-map archive on UC Davis Box](https://ucdavis.app.box.com/file/2495554626645).
Access follows the file's existing Box permissions; sign-in and access approval
may be required. This is a Box file-page link, not an anonymous public-download
link.

- **Archive:** `RFRL_DT_SAC_9e6453c7_downtilt_000-020_9sector_256x256.zip`
- **Scene:** `SAC_9e6453c7`.
- **Contents:** 189 single-sector radio maps: 9 sectors × 21 downtilt angles
  (0° through 20°, inclusive, at 1° intervals), plus metadata and manifests.
  Each map is a `float32` NumPy array with shape `256 × 256`.
- **Download size:** 40,045,727 bytes (approximately 40.05 MB).
- **Archive SHA-256:** `7046bd2a4623105d27e40bc4de33403b4581cd0cda8863eb80e3dcd2fcec7261`.

This archive contains the radio-map cache only. The trained controller
checkpoints and the demand-estimator checkpoint are separate assets and are
**not included** in this download. The frozen result tables above can still be
regenerated from this repository without those assets.

## Method / baselines in the comparison

Random, tuned best-constant, flat joint DQN over the 3^9 action space, PPO,
one-step contextual variant (gamma = 0), SAINT with identity tokens, QMIX-style
monotonic mixing, branching DQN (generic and task-tailored encoders),
discrete SAC on identity tokens, Tok3 + double-Q (trainer ablation), and the
proposed Tok3 + discrete SAC. Details and citations in the paper.

## License

Code: MIT. Evaluation records and figures: CC BY 4.0.
