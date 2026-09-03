# -*- coding: utf-8 -*-
"""Full model-structure diagram: inputs (DT maps + estimated demand + reports)
-> per-sector Tok3 tokens -> transformer -> per-sector Q -> joint action.
Same visual language as fig_tok3_arch (pastel boxes, math on arrows, dashed frames)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = "#2F2F44"
LAV, LAV2 = "#DCD6EC", "#D8D8E8"
YEL, SAL, BLU, GRY, PNK = "#FBF0CF", "#F6C8C4", "#CFE0F4", "#E4E2EC", "#F7D6D6"
RED = "#D95F5F"

fig, ax = plt.subplots(figsize=(19.8, 9.0), dpi=200)
ax.set_xlim(0, 19.9); ax.set_ylim(0, 9.0)
ax.axis("off")

def rect(x, y, w, h, text, fc, fs=11.5, rounded=False, lw=1.6):
    if rounded:
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                           fc=fc, ec=EDGE, lw=lw)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=EDGE, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
    return p

def arrow(x0, y0, x1, y1, color="black", lw=1.8, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=15))

def line(x0, y0, x1, y1, color="black", lw=1.8, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)

# ================= input side =================
rect(0.30, 7.15, 2.55, 1.10, "Nominal DT maps\nRSRP / SINR at $\\theta_t$", LAV, fs=11)
rect(0.30, 5.70, 2.55, 1.10, "Estimated demand $\\hat{\\rho}$\n(neural estimator)", LAV, fs=11)
ax.text(1.58, 5.50, "from 200 anonymous quantized reports", ha="center", fontsize=7.8, color="#666677")

rect(3.85, 6.30, 1.95, 1.35, "State maps\n$3 \\times 256 \\times 256$", LAV2, fs=11)
arrow(2.85, 7.70, 3.85, 7.28)
arrow(2.85, 6.25, 3.85, 6.68)

rect(6.35, 6.30, 1.55, 1.35, "Global CNN\nencoder", LAV2, fs=11)
arrow(5.80, 6.97, 6.35, 6.97)
line(7.90, 6.97, 9.05, 6.97)
arrow(9.05, 6.97, 9.05, 5.52)
ax.text(8.45, 7.13, r"$\gamma(g_t),\ \beta(g_t)$", ha="center", fontsize=12)

rect(0.30, 3.95, 2.55, 1.00, "Sector identity table\n$E \\in \\mathbb{R}^{9\\times 64}$  (learned)", YEL, fs=10.5)
arrow(2.85, 4.45, 8.30, 4.45)
ax.text(5.55, 4.60, "$E_i$", ha="center", fontsize=12)

rect(0.30, 2.30, 2.55, 1.00, "Served-report set $u_{i,t}$\n(grouped by strongest sector)", LAV, fs=10, rounded=True)
rect(3.85, 2.30, 1.65, 1.00, "Report MLP\n$\\phi_r$", SAL, fs=10.5)
arrow(2.85, 2.80, 3.85, 2.80)
line(5.50, 2.80, 10.55, 2.80, color=RED)
arrow(10.55, 2.80, 10.63, 4.18, color=RED)
ax.text(7.9, 2.95, r"$\phi_r(u_{i,t})$", ha="center", fontsize=11.5, color=RED)

rect(0.30, 0.75, 2.55, 1.00, "Demand-physics\nscalars $d_{i,t}$  (11)", LAV, fs=10.5, rounded=True)
rect(3.85, 0.75, 1.65, 1.00, "Demand MLP\n$\\phi_d$", SAL, fs=10.5)
arrow(2.85, 1.25, 3.85, 1.25)
line(5.50, 1.25, 10.93, 1.25, color=RED)
arrow(10.93, 1.25, 10.85, 4.18, color=RED)
ax.text(7.9, 1.40, r"$\phi_d(d_{i,t})$", ha="center", fontsize=11.5, color=RED)

# dotted provenance: state maps -> demand-physics scalars
line(4.40, 6.30, 3.30, 1.75, color="#999999", lw=1.0, ls=(0, (2, 3)))
ax.text(1.58, 0.55, "computed from state maps × cached tilt sweep", ha="center", fontsize=7.8, color="#666677")

# ================= FiLM + token =================
rect(8.30, 3.35, 1.5, 2.15, "State\nconditioning\n(FiLM)", YEL, fs=11.5)
plus = (10.74, 4.45)
arrow(9.80, 4.45, plus[0]-0.26, 4.45)
ax.text(10.05, 4.62, r"$\tilde{x}_{i,t}$", ha="center", fontsize=11)
ax.add_patch(Circle(plus, 0.26, fc="white", ec=EDGE, lw=1.6, zorder=3))
ax.text(plus[0], plus[1]-0.02, "+", ha="center", va="center", fontsize=16, zorder=4)
ax.text(9.8, 0.28, "token assembly:   $x_{i,t} = \\gamma(g_t)\\odot E_i + \\beta(g_t) + \\phi_r(u_{i,t}) + \\phi_d(d_{i,t})$",
        ha="center", fontsize=12.5)

# ================= transformer + heads =================
ax.add_patch(Rectangle((11.75, 3.10), 1.95, 2.70, fc="none", ec="#999999", lw=1.2, ls=(0, (4, 3))))
ax.text(12.72, 5.92, r"$\times\ L = 2$", ha="center", fontsize=12.5)
rect(12.07, 3.50, 1.32, 1.90, "Transformer\nblock", BLU, fs=11)
arrow(plus[0]+0.26, 4.45, 12.07, 4.45)
ax.text(12.72, 2.85, "4 heads · token dim 64", ha="center", fontsize=8.2, color="#666677")

ax.add_patch(Rectangle((14.05, 3.10), 3.55, 2.70, fc="none", ec="#999999", lw=1.2, ls=(0, (4, 3))))
ax.text(15.82, 5.92, r"$\times\ 9$ sectors", ha="center", fontsize=12.5)
rect(14.30, 3.65, 1.30, 1.60, "Shared\npolicy head", GRY, fs=10.5)
rect(16.45, 3.90, 1.05, 1.10, "arg max", GRY, fs=11)
arrow(13.39, 4.45, 14.30, 4.45)
arrow(15.60, 4.45, 16.45, 4.45)
ax.text(16.02, 5.40, r"$\pi_i(-1,\ 0,\ +1)$", ha="center", fontsize=10)

rect(18.05, 3.75, 1.55, 1.40, "$a_t \\in$\n$\\{-1,0,+1\\}^9$", PNK, fs=11.5)
arrow(17.50, 4.45, 18.05, 4.45)

ax.text(18.82, 2.85, "deployed actor: 378 k", ha="center", fontsize=8.2, color="#666677")
ax.text(15.85, 2.55, "trained with discrete SAC: twin per-sector Q critics + auto-tuned temperature (training only)",
        ha="center", fontsize=8.2, color="#666677")

# closed loop: action feeds back to tilts
line(18.82, 5.15, 18.82, 8.50, lw=1.1, color="#888888")
line(18.82, 8.50, 1.58, 8.50, lw=1.1, color="#888888")
arrow(1.58, 8.50, 1.58, 8.28, lw=1.1, color="#888888")
ax.text(10.2, 8.63, "tilts update:  $\\theta_{t+1} = \\theta_t + a_t$   (20 control steps per episode)",
        ha="center", fontsize=9, color="#666677")

fig.savefig(os.path.join(HERE, "fig_model_arch.png"), bbox_inches="tight", facecolor="white")
print("wrote fig_model_arch.png")
