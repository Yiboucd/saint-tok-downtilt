# -*- coding: utf-8 -*-
"""Fig. (bands): operator-facing quality at episode end, true-demand weighted.
Compares BEFORE CONTROL, the STRONGEST BASELINE (discrete SAC on identity tokens)
and the PROPOSED method.  Numbers are three-seed means from the frozen paired
evaluation (sources: _smoke/v12_downstream_all.csv, v12_se_final_eval.json
[initial], v12_se_sactok3_full.json [clean rung]).  Outage deliberately not shown.
Usage: python make_fig5_bands_se.py   -> fig5_bands_se.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["before\ncontrol", "best baseline\n(discrete SAC)", "Proposed"]
se = [1.8399, 2.2160, 2.3036]
exc = [10.49, 19.50, 22.21]
poor = [14.73, 9.98, 9.54]
colors = ["#8b93a7", "#0e7c86", "#0ec28f"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.55]})

# (a) demand-weighted SE
bars = a1.bar(range(3), se, color=colors, width=0.62, edgecolor=["none", "none", "k"], linewidth=[0, 0, 1.4])
for i, v in enumerate(se):
    a1.text(i, v + 0.03, "%.2f" % v, ha="center", va="bottom", fontsize=10, fontweight="bold")
for i in (1, 2):
    a1.text(i, 1.0, "+%.1f%%" % (100 * (se[i] / se[0] - 1)), ha="center", va="center", color="white",
            fontsize=8, rotation=90, fontweight="bold" if i == 2 else "normal")
a1.set_xticks(range(3)); a1.set_xticklabels(labels, fontsize=8.5)
a1.set_ylabel("demand-weighted SE (bps/Hz)", fontsize=9)
a1.set_ylim(0, 2.75)
a1.set_title("Throughput per Hz", fontsize=10, color="#1f2a5a", loc="left")
a1.spines[["top", "right"]].set_visible(False)

# (b) service-band shares
groups = ["excellent band\n(RSRP ≥ −80 dBm, SINR ≥ 20 dB)", "poor / unserved"]
vals = [exc, poor]
w = 0.26
for i in range(3):
    xs = [g + (i - 1) * w for g in range(2)]
    ys = [vals[g][i] for g in range(2)]
    a2.bar(xs, ys, width=w, color=colors[i], edgecolor="k" if i == 2 else "none", linewidth=1.4 if i == 2 else 0,
           label=labels[i].replace("\n", " "))
    for x, y in zip(xs, ys):
        a2.text(x, y + 0.35, "%.1f" % y, ha="center", va="bottom", fontsize=8.5, fontweight="bold" if i == 2 else "normal")
a2.set_xticks(range(2)); a2.set_xticklabels(groups, fontsize=8.5)
a2.set_ylabel("share of demand (%)", fontsize=9)
a2.set_ylim(0, 36)
a2.set_title("Where the users move", fontsize=10, color="#1f2a5a", loc="left")
a2.legend(fontsize=8, frameon=False, loc="upper right", ncol=1, borderaxespad=0.2)
a2.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig5_bands_se.png", dpi=220)
print("saved fig5_bands_se.png")
