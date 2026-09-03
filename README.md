# SAINT-Tok: Per-Sector Tokenized RL for Joint Multi-Sector Downtilt Control

Evaluation records, figure sources, and code for the paper

> *Learning Multi-Sector Antenna Downtilt Control from Anonymous Quantized
> Feedback with Per-Sector Tokenized Reinforcement Learning*
> Yibo Ma, Songyang Zhang, Zhi Ding (under review).

A digital-twin-assisted RL controller that assigns each of nine co-channel
sectors a transformer token carrying its own observations (commanded tilt,
served-report statistics, demand-physics aggregates), coordinates the tokens
with self-attention, and is trained with discrete soft actor--critic on a
parameter-free, coverage-gated truncated-Shannon spectral-efficiency
objective. Deployed policy: one 378k-parameter actor, one forward pass per
control step.

## Reproduce every number in the paper — no GPU, no training

All headline results are frozen, per-episode evaluation records
(90 paired episodes, 3 seeds per learned method) stored as JSON in
`results/`. To regenerate Table I and the paired bootstrap confidence
intervals quoted in the text:

```bash
cd tables
python make_table1.py        # needs numpy only
```

Figure sources are in `figures/scripts/` (matplotlib); the rendered figures
used in the paper are in `figures/`.

## Layout

| Path | Contents |
|---|---|
| `results/` | Frozen evaluation records: main 12-policy comparison, mismatch severity ladders, downstream service-quality metrics, reward-parameter sweep, ablation evaluations (per-episode values included, so every CI is recomputable) |
| `tables/make_table1.py` | Regenerates Table I + headline CIs from `results/` |
| `figures/`, `figures/scripts/` | Paper figures and their generation scripts |
| `eval/` | As-run evaluation scripts (paths refer to our cluster; kept verbatim for provenance) |
| `src/` | Controller, baselines, and training runner (reference; training additionally requires the radio-map cache, below) |
| `data/` | Small derived data (uniform-tilt trade-off locus) |

## Heavy assets (not in this repository)

Training end-to-end requires the factorized radio-map cache (ray-traced
per sector--tilt pair once, ~2 GB) and per-run checkpoints/training
histories (~500 MB). Links will be added here; until then they are
available from the authors on request.

## Method / baselines in the comparison

Random, tuned best-constant, flat joint DQN over the 3^9 action space, PPO,
one-step contextual variant (gamma = 0), SAINT with identity tokens, QMIX-style
monotonic mixing, branching DQN (generic and task-tailored encoders),
discrete SAC on identity tokens, Tok3 + double-Q (trainer ablation), and the
proposed Tok3 + discrete SAC. Details and citations in the paper.

## License

Code: MIT. Evaluation records and figures: CC BY 4.0.
