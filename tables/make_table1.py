# -*- coding: utf-8 -*-
"""Reproduce the current main table from frozen records; no training or GPU.

Run from any working directory:
    python /path/to/repository/tables/make_table1.py
    python /path/to/repository/tables/make_table1.py --bootstrap

The 90 paired episodes are ordered as 50 seen-family cases followed by 40
held-out-family cases. Learned methods average three training seeds within
each case before reporting the overall and split means. Fixed-action rows
come from the verified CPU endpoint replay in data/, not a renamed aggregate.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N_SEEN = 50
N_HELD_OUT = 40
N_CASES = N_SEEN + N_HELD_OUT
SEED_SUFFIXES = (0, 1, 2)  # training seeds 20260810, 20260811, 20260812
RESULT_FILES = (
    "v12_se_final_eval.json",
    "v12_se_newbase_eval.json",
    "v12_se_naive_eval.json",
    "v12_se_flat_eval.json",
    "v12_se_sactok3_final.json",
)
FIXED_ACTION_FILE = "fixed_action_baselines_90.json"


@dataclass
class Row:
    label: str
    per_case: np.ndarray
    training_seeds: int = 0  # random and fixed-action policies are not trained

    @property
    def metrics(self):
        return split_means(self.per_case)


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def checked_cases(values, label):
    values = np.asarray(values, dtype=float)
    assert values.shape == (N_CASES,), (label, values.shape)
    assert np.isfinite(values).all(), (label, "non-finite per-case value")
    return values


def split_means(per_case):
    assert per_case.shape == (N_CASES,)
    return (
        float(per_case.mean()),
        float(per_case[:N_SEEN].mean()),
        float(per_case[N_SEEN:].mean()),
    )


def checked_record(record, label, per_case_key="per_case_dse"):
    per_case = checked_cases(record[per_case_key], label)
    overall, seen, held_out = split_means(per_case)
    for field, expected in (
        ("dSE", overall),
        ("dSE_seen", seen),
        ("seen", seen),
        ("dSE_ood", held_out),
        ("ood", held_out),
    ):
        if field in record:
            assert np.isclose(record[field], expected, rtol=0, atol=1e-6), (
                label, field, record[field], expected
            )
    return per_case


def learned_row(label, source, prefix, per_case_key="per_case_dse"):
    keys = [f"{prefix}{suffix}" for suffix in SEED_SUFFIXES]
    actual = {key for key in source if key.startswith(prefix)}
    assert actual == set(keys), (label, "expected exactly three seeds", actual)
    per_seed = [checked_record(source[key], key, per_case_key) for key in keys]
    return Row(label, np.mean(per_seed, axis=0), len(keys))


def fixed_action_rows(replay):
    assert replay["schema"] == "fixed_action_endpoint_cpu_replay/1"
    protocol = replay["protocol"]
    assert protocol["horizon"] == 20
    assert protocol["commanded_tilt_range"] == [0, 20]
    assert protocol["action_steps"] == [-1, 1]
    assert protocol["fixed_minus1_endpoint"] == 0
    assert protocol["fixed_plus1_endpoint"] == 20
    assert protocol["repeats_per_family"] == 10
    assert protocol["reward_scale_applied"] is False

    cases = replay["cases"]
    assert len(cases) == N_CASES
    assert [case["case_index"] for case in cases] == list(range(N_CASES))
    assert [case["group"] for case in cases] == (
        ["seen"] * N_SEEN + ["held_out"] * N_HELD_OUT
    )
    expected_order = [
        (group, family, repeat)
        for group, families in protocol["groups"]
        for family in families
        for repeat in range(protocol["repeats_per_family"])
    ]
    assert [(c["group"], c["family"], c["repeat"]) for c in cases] == expected_order
    initial_angles = np.asarray([case["initial_angles"] for case in cases])
    assert initial_angles.shape == (N_CASES, 9)
    assert np.all((initial_angles >= 0) & (initial_angles <= 20))

    rows = []
    for label, field, endpoint in (
        ("Always -1deg", "minus1_gain", "uniform0_utility"),
        ("Always +1deg", "plus1_gain", "uniform20_utility"),
    ):
        per_case = checked_cases(replay["per_case"][field], label)
        recorded = np.asarray([case[field] for case in cases])
        recomputed = np.asarray([
            case[endpoint] - case["initial_utility"] for case in cases
        ])
        assert np.allclose(per_case, recorded, rtol=0, atol=1e-12), label
        assert np.allclose(per_case, recomputed, rtol=0, atol=1e-12), label
        for split, mean in zip(("overall", "seen", "held_out"), split_means(per_case)):
            assert np.isclose(replay["summary"][field][split], mean, rtol=0, atol=1e-12), (
                label, split
            )
        rows.append(Row(label, per_case))
    return rows


def build_rows(root=ROOT):
    records = {name: load(root / "results" / name) for name in RESULT_FILES}
    final = records["v12_se_final_eval.json"]
    newbase = records["v12_se_newbase_eval.json"]
    rows = [Row("Random", checked_record(final["random"], "Random"))]
    rows.extend(fixed_action_rows(load(root / "data" / FIXED_ACTION_FILE)))
    for label, source, prefix, per_case_key in (
        ("DQN", records["v12_se_flat_eval.json"], "flat_s", "per_case_dse"),
        ("Identity-token PPO", final, "ppo_s", "per_case_dse"),
        ("Identity-token QMIX", newbase, "saint_qmix_s", "per_case_dse"),
        ("BDQN", records["v12_se_naive_eval.json"], "bdqn_naive_s", "per_case_dse"),
        ("Identity-token double-Q", final, "saint_id_s", "per_case_dse"),
        ("Identity-token SAC", newbase, "saint_sac_s", "per_case_dse"),
        ("SAINT-Tok (SAC)", records["v12_se_sactok3_final.json"], "s", "per_case"),
    ):
        rows.append(learned_row(label, source, prefix, per_case_key))
    assert len(rows) == 10
    assert all(row.training_seeds == 3 for row in rows[3:])
    return rows


def paired_bootstrap(proposed, baseline, nboot=20000, seed=0):
    """Percentile case bootstrap of differences after averaging training seeds.

    Cases are sampled jointly for both methods, without split stratification;
    training seeds are not resampled. This standalone comparison uses a fresh
    RNG, so finite-resample CI edges can differ from older multi-comparison runs.
    """
    difference = checked_cases(proposed - baseline, "paired difference")
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, size=(nboot, N_CASES), replace=True).mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return difference.mean(), low, high, 100 * (difference > 0).mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, choices=(3, 4), default=3,
                        help="decimal places for table metrics (default: paper's 3)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="also report the paired SAINT-Tok vs Identity-token SAC interval")
    args = parser.parse_args(argv)
    rows = build_rows()
    print("Main table: terminal utility gain (bps/Hz); 90 paired frozen episodes")
    print("Seen: first 50 cases; held-out: last 40. All learned methods: three training seeds.")
    print(f"{'Method':<25} {'Overall':>9} {'Seen':>9} {'Held-out':>9}")
    for row in rows:
        metrics = " ".join(f"{value:+9.{args.precision}f}" for value in row.metrics)
        print(f"{row.label:<25} {metrics}")
    if args.bootstrap:
        by_label = {row.label: row.per_case for row in rows}
        mean, low, high, wins = paired_bootstrap(
            by_label["SAINT-Tok (SAC)"], by_label["Identity-token SAC"]
        )
        print("\nPaired case bootstrap: 20,000 resamples, RNG seed 0; training seeds not resampled.")
        print(f"SAINT-Tok (SAC) - Identity-token SAC: {mean:+.4f} bps/Hz; "
              f"95% CI [{low:+.4f}, {high:+.4f}]; wins {wins:.0f}% of cases")


if __name__ == "__main__":
    main()
