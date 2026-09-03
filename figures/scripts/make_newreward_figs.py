# -*- coding: utf-8 -*-
"""Figures for the 3-slide new-reward deck.
All values from the FULL 3-seed paired greedy evaluation on the 90 frozen episodes
(v12_se_final_eval.json)."""
import os, csv, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Users\26651\OneDrive - University of California, Davis\Documents\RFRL\icc_saint_tok\data"
NAVY, TEAL, MINT, GREY, RED, TAN = "#1E2761", "#028090", "#02C39A", "#8A94A6", "#C0392B", "#B0A48E"
PURPLE, SLATE, OCHRE = "#9B59B6", "#6B7FA3", "#C07A2E"
KAPPA = 6.0

# ---------------------------------------------------------------- 1  results + OOD
# label, mean dSE, per-seed dSE (None = deterministic policy), colour
ROWS = [("Random",                  -0.0037, None,                            GREY),
        ("Constant",                 0.1460, None,                            GREY),
        ("PPO",                      0.2620, [0.2595, 0.2772, 0.2494],        PURPLE),
        ("One-step\nγ = 0",        0.3123, None,                            SLATE),
        ("SAINT\nidentity",        0.3358, [0.2808, 0.3306, 0.3961],        RED),
        ("QMIX",                     0.3423, [0.3072, 0.3897, 0.3300],        OCHRE),
        ("BDQN",                     0.3625, [0.3601, 0.3549, 0.3724],        TAN),
        ("SAC",                      0.3761, [0.3613, 0.3994, 0.3676],        TEAL),
        ("Proposed",                 0.4636, [0.4863, 0.4714, 0.4333],        MINT)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.5), dpi=200,
                             gridspec_kw={"width_ratios": [2.2, 1.0]})
xs = np.arange(len(ROWS))
a1.bar(xs, [r[1] for r in ROWS], color=[r[3] for r in ROWS], width=0.66,
       edgecolor=["#111111" if i == len(ROWS)-1 else "none" for i in range(len(ROWS))], linewidth=1.1)
for x, r in zip(xs, ROWS):
    top = max([r[1]] + (r[2] or []))
    a1.text(x, top + 0.014, "%+.3f" % r[1], ha="center", fontsize=8, fontweight="bold")
    if r[2]:
        a1.plot([x]*len(r[2]), r[2], "o", ms=3.0, color="#222222", zorder=5, alpha=0.85)
a1.axhline(0.1460, color=NAVY, ls="--", lw=0.9, alpha=0.6)
a1.annotate("best constant policy", (-0.42, 0.112), fontsize=7, color=NAVY, ha="left")
a1.set_xticks(xs); a1.set_xticklabels([r[0] for r in ROWS], fontsize=8.0)
a1.set_ylabel("effective SE gain  ΔU  (bps/Hz)", fontsize=9)
a1.set_ylim(-0.03, 0.56); a1.tick_params(labelsize=7.5)
a1.spines[["top", "right"]].set_visible(False)
a1.set_title("All policies trained and scored on the new reward", fontsize=8.5, loc="left", color=NAVY)
a1.annotate("dots = individual seeds", (-0.42, 0.525), fontsize=7, color=GREY, ha="left")

OOD = [("SAINT\nidentity", 0.3806, 0.2799, RED),
       ("SAC", 0.4159, 0.3262, TEAL),
       ("BDQN", 0.3548, 0.3721, TAN),
       ("Proposed", 0.4670, 0.4595, MINT)]
w = 0.34; xo = np.arange(len(OOD))
a2.bar(xo - w/2, [o[1] for o in OOD], width=w, color=[o[3] for o in OOD], alpha=0.45, label="seen families")
a2.bar(xo + w/2, [o[2] for o in OOD], width=w, color=[o[3] for o in OOD],
       edgecolor="#111111", linewidth=0.8, label="held-out families")
for x, o in zip(xo, OOD):
    a2.text(x, max(o[1], o[2]) + 0.022, "%.0f%%" % (100*o[2]/o[1]), ha="center",
            fontsize=9, fontweight="bold", color=RED if o[2]/o[1] < 0.85 else "#1E8E5A")
a2.set_xticks(xo); a2.set_xticklabels([o[0] for o in OOD], fontsize=6.8)
a2.set_ylabel("ΔU (bps/Hz)", fontsize=9); a2.set_ylim(0, 0.55)
a2.tick_params(labelsize=7.5); a2.spines[["top", "right"]].set_visible(False)
a2.legend(fontsize=7.0, frameon=False, loc="upper left", ncol=1)
a2.set_title("Retained on unseen demand", fontsize=8.5, loc="left", color=NAVY)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_nr_bars.png"), bbox_inches="tight")
print("wrote fig_nr_bars.png")

# ---------------------------------------------------------------- 2  what it buys
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(9.0, 3.1), dpi=200,
                                 gridspec_kw={"width_ratios": [1.0, 1.45, 0.8]})
lbl = ["before\ncontrol", "best\nconstant", "Proposed"]
col = [GREY, TAN, MINT]
se = [1.8399, 1.9859, 2.3036]
b1.bar(np.arange(3), se, color=col, width=0.62, edgecolor=["none", "none", "#111111"], linewidth=1.1)
for i, v in enumerate(se):
    b1.text(i, v + 0.035, "%.2f" % v, ha="center", fontsize=10, fontweight="bold")
b1.text(2, 1.24, "+25.2%", ha="center", fontsize=11, color=RED, fontweight="bold")
b1.text(1, 1.24, "+7.9%", ha="center", fontsize=9.5, color=GREY)
b1.set_xticks(np.arange(3)); b1.set_xticklabels(lbl, fontsize=8.3)
b1.set_ylabel("demand-weighted SE (bps/Hz)", fontsize=9)
b1.set_ylim(0, 2.75); b1.tick_params(labelsize=8)
b1.spines[["top", "right"]].set_visible(False)
b1.set_title("Throughput per Hz", fontsize=9.5, loc="left", color=NAVY)

w = 0.26; xs = np.arange(2)
exc = [10.5, 11.9, 22.2]; poor = [14.7, 9.5, 9.5]
for i in range(3):
    b2.bar(xs + (i-1)*w, [exc[i], poor[i]], width=w, color=col[i], label=lbl[i].replace("\n", " "),
           edgecolor="#111111" if i == 2 else "none", linewidth=1.0)
    for x, v in zip(xs + (i-1)*w, [exc[i], poor[i]]):
        b2.text(x, v + 0.45, "%.1f" % v, ha="center", fontsize=7.6, fontweight="bold" if i == 2 else "normal")
b2.set_xticks(xs)
b2.set_xticklabels(["excellent band\n(RSRP ≥ −80, SINR ≥ 20)", "poor / unserved"], fontsize=8.2)
b2.set_ylabel("share of demand (%)", fontsize=9); b2.set_ylim(0, 26)
b2.legend(fontsize=7.2, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
b2.tick_params(labelsize=8); b2.spines[["top", "right"]].set_visible(False)
b2.set_title("Where the users move", fontsize=9.5, loc="left", color=NAVY)

out = [1.597, 0.960, 1.337]
b3.bar(np.arange(3), out, color=col, width=0.62, edgecolor=["none", "none", "#111111"], linewidth=1.1)
for i, v in enumerate(out):
    b3.text(i, v + 0.035, "%.2f" % v, ha="center", fontsize=9, fontweight="bold")
b3.set_xticks(np.arange(3)); b3.set_xticklabels(lbl, fontsize=8.3)
b3.set_ylabel("coverage outage (%)", fontsize=9); b3.set_ylim(0, 2.0)
b3.tick_params(labelsize=8); b3.spines[["top", "right"]].set_visible(False)
b3.set_title("Outage: constant wins", fontsize=9.5, loc="left", color=RED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_nr_benefit.png"), bbox_inches="tight")
print("wrote fig_nr_benefit.png")

# ---------------------------------------------------------------- 3  training curves
def curve(pat):
    f = sorted(glob.glob(os.path.join(DATA, pat)))
    if not f:
        return None
    rows = list(csv.DictReader(open(f[0])))
    r = np.array([float(x["episode_return"]) for x in rows]) / KAPPA
    k = 200
    return np.convolve(r, np.ones(k)/k, mode="valid")

fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
for pat, nm, c, lw in [("hist_se_sactok3_s10.csv", "Proposed (Tok3 + SAC)", MINT, 1.9),
                       ("hist_se_sac_s10.csv", "SAC", TEAL, 1.3),
                       ("hist_se_naive_s10.csv", "BDQN", TAN, 1.3)]:
    y = curve(pat)
    if y is None:
        print("  missing", pat); continue
    ax.plot(np.arange(len(y)), y, color=c, lw=lw, label=nm)
ax.set_xlabel("training episode", fontsize=9); ax.set_ylabel("episode ΔU (bps/Hz)", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Training under the new reward (seed 20260810, 200-episode moving average)",
             fontsize=8.5, loc="left", color=GREY)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_nr_curve.png"), bbox_inches="tight")
print("wrote fig_nr_curve.png")
