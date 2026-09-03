# -*- coding: utf-8 -*-
"""PPT assets: (1) mismatch-sweep slope chart, (2) downstream full table.
All values = 3-seed means on the 90 frozen paired episodes."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, TEAL, MINT, GREY, RED, TAN = "#1E2761", "#028090", "#02C39A", "#8A94A6", "#C0392B", "#B0A48E"
PURPLE, SLATE, OCHRE = "#9B59B6", "#6B7FA3", "#C07A2E"

# ---------------------------------------------------------------- 1  ladder
# label, [clean, bias, tilt1, tilt2], color, lw, ls, label dy nudge
L = [
    ("Random",        [-0.0037, -0.0004, 0.0075, 0.0096], "#B8BEC9", 1.2, "-",  0.000),
    ("Constant",      [0.1460, 0.1563, 0.1562, 0.1533],   GREY,   1.4, "-",  0.000),
    ("PPO",           [0.2620, 0.2747, 0.2577, 0.2507],   PURPLE, 1.4, "-", -0.007),
    ("One-step γ=0",  [0.3123, 0.2963, 0.2734, 0.2550],   SLATE,  1.4, "-",  0.007),
    ("QMIX",          [0.3423, 0.3395, 0.3138, 0.3048],   OCHRE,  1.4, "-", -0.012),
    ("SAINT-identity",[0.3358, 0.3386, 0.3215, 0.3082],   RED,    1.4, "-",  0.000),
    ("BDQN",          [0.3625, 0.3620, 0.3456, 0.3239],   TAN,    1.6, "-",  0.004),
    ("SAC",           [0.3761, 0.3856, 0.3678, 0.3458],   TEAL,   1.6, "-",  0.007),
    ("Tok3 + DDQN",   [0.4334, 0.4384, 0.4166, 0.3918],   "#0B8F6F", 1.5, "--", 0.000),
    ("Proposed",      [0.4636, 0.4733, 0.4336, 0.4068],   MINT,   2.6, "-",  0.004),
]
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
xs = np.arange(4)
for nm, ys, c, lw, ls, dy in L:
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, marker="o", ms=4 if nm == "Proposed" else 3)
    ax.text(3.08, ys[3] + dy, nm, fontsize=8.5, color=c, va="center",
            fontweight="bold" if nm == "Proposed" else "normal")
ax.set_xticks(xs)
ax.set_xticklabels(["clean", "power bias\n±3 dB", "tilt ±1°\n+ bias", "tilt ±2°\n+ bias"], fontsize=9)
ax.set_ylabel("effective SE gain  ΔU  (bps/Hz)", fontsize=10)
ax.set_xlim(-0.15, 3.95); ax.set_ylim(-0.03, 0.50)
ax.tick_params(labelsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Mismatch sweep: ordering preserved on every rung (90 frozen episodes, 3-seed means)",
             fontsize=10, loc="left", color=NAVY)
ax.grid(axis="y", color="#E3E6EB", lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_mismatch_ladder.png"), bbox_inches="tight", facecolor="white")
print("wrote fig_mismatch_ladder.png")

# ---------------------------------------------------------------- 2  table
rows = [
    ("(initial)",        None,    1.8399, None,  10.5, 16.1, 58.7, 14.7, 1.597),
    ("Random",           -0.0037, 1.8362, -0.2,  10.7, 16.0, 58.2, 15.1, 1.626),
    ("Constant",          0.1460, 1.9859,  7.9,  11.9, 17.4, 61.3,  9.5, 0.960),
    ("PPO",               0.2620, 2.1020, 14.2,  17.6, 16.2, 55.3, 10.9, 1.280),
    ("One-step γ=0",      0.3123, 2.1523, 17.0,  18.5, 16.6, 54.4, 10.5, 1.552),
    ("SAINT-identity",    0.3358, 2.1758, 18.3,  19.2, 17.0, 53.3, 10.6, 1.498),
    ("QMIX",              0.3423, 2.1822, 18.6,  19.5, 16.9, 52.7, 10.8, 1.422),
    ("BDQN",              0.3625, 2.2024, 19.7,  19.6, 17.7, 51.8, 10.9, 1.360),
    ("SAC",               0.3761, 2.2160, 20.4,  19.5, 17.7, 52.8, 10.0, 1.349),
    ("Tok3 + DDQN (ablation)", 0.4334, 2.2733, 23.6, 21.4, 17.4, 51.5, 9.7, 1.346),
    ("Proposed (Tok3 + SAC)",  0.4636, 2.3036, 25.2, 22.2, 17.6, 50.7, 9.5, 1.337),
]
cols = ["policy", "ΔSE\n(bps/Hz)", "final SE\n(bps/Hz)", "gain", "excellent\n%", "good\n%", "fair\n%", "poor\n%"]
fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=200)
ax.axis("off")
cell_text = []
for r in rows:
    cell_text.append([r[0],
                      "—" if r[1] is None else "%+.3f" % r[1],
                      "%.3f" % r[2],
                      "—" if r[3] is None else "%+.1f%%" % r[3],
                      "%.1f" % r[4], "%.1f" % r[5], "%.1f" % r[6], "%.1f" % r[7]])
tbl = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center",
               colWidths=[0.26, 0.105, 0.105, 0.095, 0.105, 0.09, 0.09, 0.09])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.6)
tbl.scale(1.0, 1.42)
n = len(rows)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor("#D8DCE2")
    if ri == 0:
        cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(cell.get_height() * 1.25)
    elif ri == 1:
        cell.set_facecolor("#F2F3F5"); cell.set_text_props(style="italic", color="#555")
    elif ri == n:
        cell.set_facecolor("#D8F5EC"); cell.set_text_props(fontweight="bold")
    elif ri == n - 1:
        cell.set_facecolor("#F0F7F4")
    elif ri % 2 == 0:
        cell.set_facecolor("#FAFBFC")
    if ci == 0 and ri > 0:
        cell.set_text_props(ha="left")
ax.set_title("Downstream communications metrics — all policies (90 frozen episodes, 3-seed means; "
             "coverage-gated truncated-Shannon SE)", fontsize=9.5, loc="left", color=NAVY, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "table_downstream_all.png"), bbox_inches="tight", facecolor="white")
print("wrote table_downstream_all.png")
