# -*- coding: utf-8 -*-
"""PPT figures for the two robustness questions the advisor raised.
  fig_param_robustness.png : objective-constant sweep (11 variants x 12 policies, single rollout re-scored)
  fig_feedback_ladder.png  : feedback-error ladder (SINR noise / coarser quantization), frozen policies
Sources: saint-tok-repo/results/v12_se_param_sweep.json, v12_se_feedback_ladder.json;
clean-rung numbers = Table I of the paper (v12_se_naive_eval / newbase / sactok3_final).
Run:  python make_robustness_ppt_figs.py   (from figures/scripts/; writes the four PNGs into figures/)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")   # <repo>/results
OUT = os.path.join(HERE, "..")                     # <repo>/figures
NAVY = "#1f2a5a"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

# ------------------------------------------------------------------ figure 1
d = json.load(open(os.path.join(RES, "v12_se_param_sweep.json")))
variants = d["variants"]
fm = d["family_means"]
labels = {"spec": "spec\n(paper)", "gate-105": "gate\n−105 dBm", "gate-95": "gate\n−95 dBm", "gate-90": "gate\n−90 dBm",
          "atten0.5": "atten.\n0.5", "atten0.7": "atten.\n0.7", "cap4.0": "cap\n4.0", "cap4.8": "cap\n4.8",
          "capNR7.4": "cap 7.4\n(NR)", "floor-6": "floor\n−6 dB", "floor-14": "floor\n−14 dB"}
short = {"spec": "spec", "gate-105": "g −105", "gate-95": "g −95", "gate-90": "g −90", "atten0.5": "a 0.5", "atten0.7": "a 0.7",
         "cap4.0": "c 4.0", "cap4.8": "c 4.8", "capNR7.4": "c 7.4 NR", "floor-6": "f −6", "floor-14": "f −14"}
order = ["spec", "gate-105", "gate-95", "gate-90", "atten0.5", "atten0.7", "cap4.0", "cap4.8", "capNR7.4", "floor-6", "floor-14"]
idx = [variants.index(v) for v in order]
names = {"random": "Random", "const": "Best constant", "flat": "Flat DQN", "ppo": "PPO", "myopic": "One-step",
         "identity": "SAINT identity", "qmix": "QMIX", "naive": "Branching DQN", "sac": "Discrete SAC",
         "bdqn_t": "tailored BDQN", "ddqn_tok3": "Tok3 + double-Q", "proposed": "Proposed (Tok3 + SAC)"}
hi = {"const": ("#8b93a7", "--", 2.0), "naive": ("#b9a98b", "-", 2.0), "sac": ("#0e7c86", "-", 2.0),
      "ddqn_tok3": ("#3a7ca5", "-", 2.0), "proposed": ("#0ec28f", "-", 3.2)}
spec_i = variants.index("spec")
spec_vals = np.array([fm[m][spec_i] for m in fm])
rhos = []
for v in order:
    vi = variants.index(v)
    vals = np.array([fm[m][vi] for m in fm])
    rhos.append(spearmanr(spec_vals, vals).correlation)
print("Spearman vs spec per variant:", dict(zip(order, np.round(rhos, 3))))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1.0]})
x = np.arange(len(order))
for m in fm:
    y = [fm[m][i] for i in idx]
    if m in hi:
        c, ls, lw = hi[m]
        a1.plot(x, y, ls, color=c, lw=lw, marker="o", ms=5, label=names[m], zorder=3,
                markeredgecolor="k" if m == "proposed" else c)
    else:
        a1.plot(x, y, "-", color="#c8ccd6", lw=1.0, marker="o", ms=3, zorder=1)
a1.plot([], [], "-", color="#c8ccd6", lw=1.0, label="other 7 policies")
a1.set_xticks(x); a1.set_xticklabels([labels[v] for v in order], fontsize=9.5)
a1.set_ylabel("ΔU under the perturbed objective (bps/Hz)")
a1.set_title("Ranking of all 12 policies unchanged (Spearman ρ vs. spec = %.3f on all 11 variants)" % min(rhos),
             color=NAVY, loc="left", fontsize=12)
a1.set_ylim(-0.03, 0.66)
a1.grid(alpha=0.3); a1.legend(fontsize=9.5, loc="upper left", frameon=False, ncol=2)
a1.spines[["top", "right"]].set_visible(False)
for xx in (0.5, 3.5, 5.5, 8.5):
    a1.axvline(xx, color="#dddddd", lw=0.8)

# right: margins of the proposed method
m_ddqn = [fm["proposed"][i] - fm["ddqn_tok3"][i] for i in idx]
m_sac = [fm["proposed"][i] - fm["sac"][i] for i in idx]
m_const = [fm["proposed"][i] - fm["const"][i] for i in idx]
w = 0.38
a2.bar(x - w / 2, m_sac, w, color="#0e7c86", label="vs. discrete SAC (identity)")
a2.bar(x + w / 2, m_ddqn, w, color="#3a7ca5", label="vs. Tok3 + double-Q (trainer ablation)")
for i, v in enumerate(m_ddqn):
    a2.text(x[i] + w / 2, v + 0.003, "%.3f" % v, ha="center", va="bottom", fontsize=8, color="#3a7ca5")
for i, v in enumerate(m_sac):
    a2.text(x[i] - w / 2, v + 0.003, "%.3f" % v, ha="center", va="bottom", fontsize=8, color="#0e7c86")
a2.set_xticks(x); a2.set_xticklabels([short[v] for v in order], fontsize=9, rotation=45, ha="right")
a2.set_ylabel("margin of Proposed (bps/Hz)")
a2.set_ylim(0, 0.16)
a2.set_title("Margins never close; the NR cap widens them", color=NAVY, loc="left", fontsize=12)
a2.plot([], [], " ", label="(vs. best constant: 0.28–0.43 on every variant)")
a2.axhline(0, color="k", lw=0.6); a2.grid(alpha=0.3, axis="y")
a2.legend(fontsize=9, frameon=False, loc="upper left")
a2.spines[["top", "right"]].set_visible(False)
fig.suptitle("Objective-constant robustness: greedy trajectories re-scored under 11 variants of Eq. (U); 90 frozen episodes × 3 seeds",
             fontsize=11, color="#555555", y=0.995)
fig.tight_layout()
p1 = os.path.join(OUT, "fig_param_robustness.png"); fig.savefig(p1, dpi=200); print("saved", p1)

# ------------------------------------------------------------------ figure 2
f = json.load(open(os.path.join(RES, "v12_se_feedback_ladder.json")))
rungs = [("clean", "clean\n(1 dB / 0.5 dB)"), ("noise1", "SINR noise\nσ = 1 dB"), ("noise2", "SINR noise\nσ = 2 dB"),
         ("coarse", "coarse quant.\n2 dB / 1 dB"), ("coarse+noise", "coarse +\nσ = 1 dB")]
CLEAN = {"const": 0.1460, "naive": 0.3625, "sac": 0.3761, "proposed": 0.4636}
pols = [("const", "Best constant", "#8b93a7"), ("naive", "Branching DQN", "#b9a98b"),
        ("sac", "Discrete SAC (identity)", "#0e7c86"), ("proposed", "Proposed (Tok3 + SAC)", "#0ec28f")]


def m3(src, pre):
    ks = [k for k in src if k == pre or k.startswith(pre + "_s")]
    return float(np.mean([src[k]["dSE"] for k in ks]))


def draw_ladder(ax, src, rung_list, title):
    vals = {p: [CLEAN[p]] + [m3(src[r], p) for r, _ in rung_list[1:]] for p, _, _ in pols}
    x = np.arange(len(rung_list)); w = 0.2
    for j, (p, lab, c) in enumerate(pols):
        xs = x + (j - 1.5) * w
        ax.bar(xs, vals[p], w, color=c, label=lab, edgecolor="k" if p == "proposed" else "none", linewidth=1.3 if p == "proposed" else 0)
        for xi, v in zip(xs, vals[p]):
            ax.text(xi, v + 0.006, "%.3f" % v, ha="center", va="bottom", fontsize=8, fontweight="bold" if p == "proposed" else "normal")
    for i in range(1, len(rung_list)):
        ret = 100 * vals["proposed"][i] / vals["proposed"][0]
        ax.text(x[i] + 1.5 * w, vals["proposed"][i] + 0.035, "%.0f%% of clean" % ret, ha="center", fontsize=9, color="#0a8a66", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in rung_list], fontsize=10.5)
    ax.set_ylabel("ΔU (bps/Hz), 90 episodes × 3 seeds")
    ax.set_ylim(0, 0.66)
    ax.set_title(title, color=NAVY, loc="left", fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.3, axis="y"); ax.spines[["top", "right"]].set_visible(False)
    return vals


fig, ax = plt.subplots(figsize=(12.5, 4.6))
vals = draw_ladder(ax, f, rungs, "SINR measurement noise and coarser reporting resolution (nominal physics; noise before quantization; full chain incl. estimator)")
fig.tight_layout()
p2 = os.path.join(OUT, "fig_feedback_ladder.png"); fig.savefig(p2, dpi=200); print("saved", p2)
for p, _, _ in pols:
    print("%-9s" % p, " ".join("%.4f" % v for v in vals[p]))

# ---------------------------------------------------------------- figure 2b: RSRP noise (if available)
rss_path = os.path.join(RES, "v12_se_rss_noise_ladder.json")
if os.path.exists(rss_path):
    g = json.load(open(rss_path))
    rungs_r = [("clean", "clean\n(no noise)"), ("rss1", "RSRP noise\nσ = 1 dB"), ("rss2", "RSRP noise\nσ = 2 dB"),
               ("rss3", "RSRP noise\nσ = 3 dB"), ("rss2+sinr1", "RSRP σ = 2 dB\n+ SINR σ = 1 dB")]
    rungs_r = [(r, l) for r, l in rungs_r if r == "clean" or r in g]
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    vals_r = draw_ladder(ax, g, rungs_r, "RSRP measurement noise on every reported sector RSRP (nominal physics and quantization; noise before quantization; full chain incl. estimator)")
    fig.tight_layout()
    p3 = os.path.join(OUT, "fig_feedback_ladder_rsrp.png"); fig.savefig(p3, dpi=200); print("saved", p3)
    for p, _, _ in pols:
        print("%-9s" % p, " ".join("%.4f" % v for v in vals_r[p]))
    # combined two-row figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.8))
    draw_ladder(ax1, f, rungs, "SINR noise and coarser reporting resolution")
    draw_ladder(ax2, g, rungs_r, "RSRP noise (quantization nominal)")
    fig.suptitle("Feedback-error ladders: nominal physics, degraded observations only (noise added before quantization; full chain incl. estimator)",
                 fontsize=12, color="#555555")
    fig.tight_layout()
    p4 = os.path.join(OUT, "fig_feedback_ladder_both.png"); fig.savefig(p4, dpi=200); print("saved", p4)
