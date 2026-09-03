# -*- coding: utf-8 -*-
"""Training curves under the new reward for the whole comparison set
(seed 20260810, 200-episode moving average, colors matched to the bars figure)."""
import os, csv, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Users\26651\OneDrive - University of California, Davis\Documents\RFRL\icc_saint_tok\data"
NAVY, TEAL, MINT, GREY, RED, TAN = "#1E2761", "#028090", "#02C39A", "#8A94A6", "#C0392B", "#B0A48E"
PURPLE, OCHRE = "#9B59B6", "#C07A2E"
KAPPA = 6.0

def curve(name):
    f = os.path.join(DATA, name)
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    r = np.array([float(x["episode_return"]) for x in rows]) / KAPPA
    k = 200
    return np.convolve(r, np.ones(k)/k, mode="valid")

LINES = [
    ("hist_se_sactok3_s10.csv",       "Proposed (Tok3)",        MINT,   2.2, "-"),
    ("hist_se_naive_s10.csv",         "BDQN",                   TAN,    1.4, "-"),
    ("hist_se_sac_s10.csv",           "SAC",                    TEAL,   1.4, "-"),
    ("hist_se_qmix_s10.csv",          "QMIX",                   OCHRE,  1.4, "-"),
    ("hist_se_id_s10.csv",            "SAINT (identity)",       RED,    1.2, "-"),
    ("hist_se_ppo_s10.csv",           "PPO",                    PURPLE, 1.2, "-"),
]

fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
for fname, nm, c, lw, ls in LINES:
    y = curve(fname)
    if y is None:
        print("  missing", fname); continue
    ax.plot(np.arange(len(y)), y, color=c, lw=lw, ls=ls, label=nm)
ax.axhline(0.1460, color=NAVY, ls=":", lw=1.0, alpha=0.6)
ax.annotate("best constant policy", (9800, 0.152), fontsize=7.5, color=NAVY, ha="right")
ax.set_xlabel("training episode", fontsize=9.5)
ax.set_ylabel("episode ΔU (bps/Hz)", fontsize=9.5)
ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 0.50)
ax.legend(fontsize=8.2, frameon=False, loc="lower right", ncol=2)
ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Training under the new reward — comparison set (seed 20260810, 200-episode moving average)",
             fontsize=9, loc="left", color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_se_training_all.png"), bbox_inches="tight")
print("wrote fig_se_training_all.png")
