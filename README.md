# SAINT-Tok

Code, frozen evaluation records, and figure sources for
**Digital-Twin-Assisted Reinforcement Learning for Dynamic Coverage Through
Multi-Station Downtilt Control**.

Yibo Ma, Songyang Zhang, Roland Gadbois, Yan Xin,
Jianzhong (Charlie) Zhang, and Zhi Ding.

SAINT-Tok combines digital-twin radio maps and sparse UE reports with
sector-specific tokens and cross-sector attention. The proposed controller
uses factorized discrete soft actor-critic (SAC) and a coverage-gated,
truncated-Shannon spectral-efficiency objective.

## Reproduce the reported results

Frozen-record analysis and training-curve plotting need only NumPy and
Matplotlib, not a GPU, radio-map download, or checkpoints:

```bash
python -m pip install -r requirements.txt
python tables/make_table1.py
python figures/scripts/make_se_training_all_fig.py --output-dir outputs
```

Use `python tables/make_table1.py --bootstrap` for the paired
SAINT-Tok versus identity-token SAC interval (20,000 case resamples).
Run the data checks with `python -m unittest discover -s tests`.

The main table includes Random, Always -1 degree, Always +1 degree, DQN,
BDQN, identity-token Double Q-learning/PPO/QMIX/SAC, and SAINT-Tok (SAC).
Each learned entry averages seeds 20260810, 20260811, and 20260812 on the
same 90 frozen cases: entries 0-49 are seen-demand cases and 50-89 are
held-out-demand cases. Fixed-action entries use the separately verified
endpoint replay in `data/fixed_action_baselines_90.json`.

The training figure retains the six curves in the paper: seed-20260810
training logs, with a trailing 200-episode average of unscaled utility gain.
They are not greedy evaluation curves.

## Contents and result sources

| Path | Contents |
| --- | --- |
| `src/` | SAINT/Tok3 networks, SAC/QMIX modules, and original training runner |
| `eval/` | As-run evaluation scripts supporting retained results, plus endpoint replay |
| `results/` | Frozen main-comparison, physical-KPI, mismatch, and feedback-error records |
| `data/` | Fixed-action replay records and six compact training histories |
| `tables/` | Current main-table calculation |
| `figures/` | Current architecture and training-curve exports; training plotting script |
| `tests/` | Main-table values, seed counts, and paired-case integrity checks |

| Result | Source under `results/` |
| --- | --- |
| Random, identity-token PPO and Double Q-learning | `v12_se_final_eval.json` |
| DQN | `v12_se_flat_eval.json` |
| BDQN | `v12_se_naive_eval.json` |
| Identity-token SAC and QMIX | `v12_se_newbase_eval.json` |
| Proposed SAC, three-seed main evaluation | `v12_se_sactok3_final.json` |
| Proposed seed-20260810 input to the three-seed merge | `v12_se_sactok3_eval.json` |
| RSRP/SINR gains and service shares | `v12_se_decomp.json`, `v12_downstream_all.csv`, and initial/final fields in `v12_se_final_eval.json` |
| Proposed clean and mismatch evaluations | `v12_se_sactok3_full.json` |
| Baseline mismatch evaluations | `v12_se_ladder_all.json`, `v12_se_severity.json` |
| SINR feedback noise and coarser quantization | `v12_se_feedback_ladder.json` |

Original result values are unchanged. Some source files contain additional
historical methods from the same evaluation run; the main-table script
selects only current paper methods. In particular, `proposed_s*` in
`v12_se_final_eval.json` denotes the older Tok3 + Double-Q ablation,
**not** the proposed SAC policy.

## Radio-map cache on Box

[Open the 0-20 degree radio-map archive](https://ucdavis.app.box.com/file/2495554626645).
Box permissions are separate from this public repository. Sign-in or
access approval may be required.

- Archive: `RFRL_DT_SAC_9e6453c7_downtilt_000-020_9sector_256x256.zip`
- Scene: `SAC_9e6453c7`
- Contents: 189 maps, covering 9 sectors and 21 angles (0-20 degrees in
  1-degree steps), with metadata and manifests
- Each map: `float32`, `256 x 256`
- Size: 40,045,727 bytes
- SHA-256: `7046bd2a4623105d27e40bc4de33403b4581cd0cda8863eb80e3dcd2fcec7261`

The archive contains radio maps, not trained controller or demand-estimator
checkpoints. It also excludes the extended 0-89 degree cache used for
tilt-mismatch tests.

## Training and evaluation source

`src/` and `eval/` are reference implementations from the experiment
workspace, **not a standalone retraining package**. They retain original
module names and cluster paths for provenance. Running them additionally
requires PyTorch, the original environment/protocol modules
(`experiments/train_v1_1_controller_gate.py`,
`experiments/train_v1_mismatch_bdqn.py`,
`experiments/v1_2/protocol_v1_2.py`), the cache loader, demand-estimator
implementation and checkpoint, and controller checkpoints for evaluation.
These dependencies are not all included. DQN evaluation also references
the original `flat_joint_dqn` module.

The paper uses the v1.2 three-action protocol (down/keep/up), 10,000
episodes of 20 steps per seed, and reward scale 6. Shared source files
also contain older protocol variants; their default arguments are not
a complete specification of the paper's runs. The lightweight commands
above operate on saved records without executing training/evaluation jobs.

## License

Code: MIT. Evaluation records and figures: CC BY 4.0.
