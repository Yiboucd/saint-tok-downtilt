# -*- coding: utf-8 -*-
"""Fig. 3 of the paper: training curves under the spectral-efficiency reward for
the whole comparison set (seed 20260810, trailing 200-episode mean).

Reads the compact training histories in ../../data/training_histories/
(episode, algo, episode_return) and writes ../fig_se_training_all.png.
The stored episode_return is the kappa-scaled training return (kappa = 6);
dividing by kappa gives the unscaled terminal gain dU in bps/Hz.
The dotted reference line is the best-constant frozen-evaluation dU read from
../../results/v12_se_final_eval.json.  Needs numpy and matplotlib only.
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "training_histories")
RESULTS = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(HERE, "..", "fig_se_training_all.png")
NAVY, TEAL, MINT, GREY, RED, TAN = "#1E2761", "#028090", "#02C39A", "#8A94A6", "#C0392B", "#B0A48E"
PURPLE, OCHRE = "#9B59B6", "#C07A2E"
KAPPA = 6.0      # reward scale used in training; episode_return / KAPPA = dU (bps/Hz)
WINDOW = 200     # trailing moving-average window (episodes)

def curve(name):
    f = os.path.join(DATA, name)
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f, newline="")))
    r = np.array([float(x["episode_return"]) for x in rows]) / KAPPA
    return np.convolve(r, np.ones(WINDOW) / WINDOW, mode="valid")

def const_reference():
    try:
        with open(os.path.join(RESULTS, "v12_se_final_eval.json")) as f:
            return float(json.load(f)["const"]["dSE"])
    except Exception:
        return 0.1460

LINES = [
    ("hist_se_sactok3_s10.csv", "Proposed (Tok3)",  MINT,   2.2, "-"),
    ("hist_se_naive_s10.csv",   "BDQN",             TAN,    1.4, "-"),
    ("hist_se_sac_s10.csv",     "SAC",              TEAL,   1.4, "-"),
    ("hist_se_qmix_s10.csv",    "QMIX",             OCHRE,  1.4, "-"),
    ("hist_se_id_s10.csv",      "SAINT (identity)", RED,    1.2, "-"),
    ("hist_se_ppo_s10.csv",     "PPO",              PURPLE, 1.2, "-"),
]

fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
n_drawn = 0
for fname, nm, c, lw, ls in LINES:
    y = curve(fname)
    if y is None:
        print("  missing", fname); continue
    # x = episode index at the end of each trailing window (WINDOW .. N)
    ax.plot(np.arange(len(y)) + WINDOW, y, color=c, lw=lw, ls=ls, label=nm)
    n_drawn += 1
c0 = const_reference()
ax.axhline(c0, color=NAVY, ls=":", lw=1.0, alpha=0.6)
ax.annotate("best constant policy", (9800, c0 + 0.006), fontsize=7.5, color=NAVY, ha="right")
ax.set_xlabel("training episode", fontsize=9.5)
ax.set_ylabel("episode ΔU (bps/Hz)", fontsize=9.5)
ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 0.50)
ax.legend(fontsize=8.2, frameon=False, loc="lower right", ncol=2)
ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Training under the SE reward — comparison set (seed 20260810, trailing 200-episode mean)",
             fontsize=9, loc="left", color=GREY)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.normpath(OUT), "with", n_drawn, "curves; constant reference =", round(c0, 4))
