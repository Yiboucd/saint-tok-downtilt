"""Redraw the six training curves in paper Fig. 2 from the compact CSV logs.

Seed 20260810, 10,000 episodes, trailing 200-episode mean of
episode_return / 6 in bps/Hz. Requires numpy and matplotlib.
Inputs are read only; the only output is training_comparison.png.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "training_histories"
KAPPA = 6.0
WINDOW = 200
EPISODES = 10000
PLOT_STRIDE = 10
# Labels, colors and line styles match the current paper training figure.
SERIES = [
    ("Proposed (SAC)", "hist_se_sactok3_s10.csv", "saint_sac_tok3", "#00A987", "-"),
    ("BDQN", "hist_se_naive_s10.csv", "bdqn_naive", "#A89B82", "-"),
    ("SAC (identity)", "hist_se_sac_s10.csv", "saint_sac", "#007F91", "--"),
    ("QMIX", "hist_se_qmix_s10.csv", "saint_qmix", "#C17C27", "-"),
    ("SAINT (identity)", "hist_se_id_s10.csv", "saint_dqn", "#C13C30", "-"),
    ("PPO", "hist_se_ppo_s10.csv", "saint_ppo", "#9656B4", "-"),
]


def load_curve(source, expected_algo):
    """Validate a full history and return every complete trailing mean."""
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"episode", "algo", "episode_return"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source}: required columns are {sorted(required)}")
        rows = list(reader)
    if len(rows) != EPISODES:
        raise ValueError(f"{source}: expected {EPISODES} rows, found {len(rows)}")
    try:
        episodes = np.array([float(row["episode"]) for row in rows])
        returns = np.array([float(row["episode_return"]) for row in rows])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid numeric episode or episode_return") from error
    if not np.array_equal(episodes, np.arange(1, EPISODES + 1)):
        raise ValueError(f"{source}: episodes must be contiguous integers 1..{EPISODES}")
    if {row["algo"] for row in rows} != {expected_algo}:
        raise ValueError(f"{source}: every algo value must be {expected_algo!r}")
    if not np.isfinite(returns).all():
        raise ValueError(f"{source}: episode_return must contain only finite values")
    smoothed = np.convolve(returns / KAPPA, np.ones(WINDOW) / WINDOW, mode="valid")
    return episodes[WINDOW - 1:].astype(int), smoothed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs",
        help="PNG output directory (default: the repository outputs directory)",
    )
    args = parser.parse_args()
    # Load all six sources before creating output; a missing source is an error.
    try:
        curves = [load_curve(DATA / filename, algo) for _, filename, algo, _, _ in SERIES]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for index, ((label, _, _, color, linestyle), (episodes, gains)) in enumerate(zip(SERIES, curves)):
        # Match the paper's drawing stride after smoothing, including the last point.
        selected = np.unique(np.r_[np.arange(0, len(gains), PLOT_STRIDE), len(gains) - 1])
        ax.plot(
            episodes[selected], gains[selected], color=color, linestyle=linestyle,
            linewidth=1.5 if index == 0 else 1.0,
            zorder=5 if index == 0 else 3, label=label,
        )
    ax.set(
        xlim=(0, EPISODES), ylim=(-0.02, .51),
        xlabel="Training episode", ylabel="Episode gain (bps/Hz)",
    )
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000], ["0", "2k", "4k", "6k", "8k", "10k"])
    ax.grid(axis="y", color="#E5E7EB", linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right", ncol=2, frameon=True, facecolor="white",
        edgecolor="none", framealpha=.95, labelspacing=.27,
        handlelength=1.7, columnspacing=.75, handletextpad=.4, borderpad=.25,
    )
    fig.tight_layout(pad=.45)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "training_comparison.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output.resolve()} with {len(curves)} curves (seed 20260810).")


if __name__ == "__main__":
    main()
