# -*- coding: utf-8 -*-
"""Fig. 6 (SE era): uniform-tilt locus + learned-policy points in the
(demand-weighted dRSRP, dSINR) plane.  Locus: data/tradeoff_curve.json
(policy-independent).  Points: _smoke/v12_se_decomp.json, 3-seed means."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
loc = json.load(open(os.path.join(HERE, "..", "data", "tradeoff_curve.json")))
NAVY, TEAL, MINT, GREY, RED, TAN = "#1E2761", "#028090", "#02C39A", "#8A94A6", "#C0392B", "#B0A48E"
PURPLE = "#9B59B6"

PTS = [  # label, dRSRP, dSINR, color, marker, size
    ("Proposed",        2.90, 3.47, MINT,     "*", 210),
    ("Tok3 + double-Q", 2.84, 3.22, "#0B8F6F", "D", 46),
    ("SAC",             2.82, 2.80, TEAL,     "o", 52),
    ("BDQN",            2.53, 2.70, TAN,      "o", 52),
    ("SAINT identity",  2.31, 2.44, RED,      "o", 52),
    ("PPO",             2.81, 2.13, PURPLE,   "o", 52),
]

fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
xr, ys = loc["d_rsrp_mean"], loc["d_sinr_mean"]
ax.plot(xr, ys, "-", color=NAVY, lw=1.6, marker=".", ms=4, zorder=2,
        label="uniform tilt 0–20°")
for t, x, y in zip(loc["uniform_tilt_deg"], xr, ys):
    if t in (0, 5, 10, 20):
        ax.annotate("%d°" % t, (x, y), textcoords="offset points",
                    xytext=(4, -9), fontsize=7, color=NAVY)
ax.axhline(max(ys), color=NAVY, ls=":", lw=1.0, alpha=0.7)
ax.annotate("uniform-family SINR ceiling (1.28 dB)", (min(xr) + 0.15, max(ys) + 0.09),
            fontsize=7, color=NAVY)
OFF = {"Proposed": (8, 2, "left"), "Tok3 + double-Q": (8, -4, "left"),
       "SAC": (8, -2, "left"), "BDQN": (-7, 2, "right"),
       "SAINT identity": (-7, -4, "right"), "PPO": (8, -4, "left")}
for lbl, x, y, c, mk, sz in PTS:
    ax.scatter([x], [y], s=sz, c=c, marker=mk, zorder=5,
               edgecolors="#111111" if lbl == "Proposed" else "none", linewidths=0.9)
    dx, dy, ha = OFF[lbl]
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=7.5, color=c, fontweight="bold" if lbl == "Proposed" else "normal")
ax.set_xlabel("demand-weighted RSRP gain (dB)", fontsize=9)
ax.set_ylabel("demand-weighted SINR gain (dB)", fontsize=9)
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#E8EAEE", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig6_tradeoff_se.png"), bbox_inches="tight", facecolor="white")
print("wrote fig6_tradeoff_se.png")
