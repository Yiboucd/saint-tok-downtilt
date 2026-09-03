# -*- coding: utf-8 -*-
"""Tok3 architecture diagram in the user's reference style
(pastel boxes, math on arrows, dashed x L / x 9 frames).
Key structural point: BOTH local paths join AFTER FiLM (additive bypass),
unlike Residual-E where r_sub enters the FiLM multiplication."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Circle
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = "#2F2F44"
LAV, LAV2 = "#DCD6EC", "#D8D8E8"
YEL, SAL, BLU, GRY, PNK = "#FBF0CF", "#F6C8C4", "#CFE0F4", "#E4E2EC", "#F7D6D6"
RED = "#D95F5F"

fig, ax = plt.subplots(figsize=(19.2, 7.8), dpi=200)
ax.set_xlim(0, 19.3); ax.set_ylim(0, 7.8)
ax.axis("off")

def rect(x, y, w, h, text, fc, fs=12.5, rounded=False, lw=1.6):
    if rounded:
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                           fc=fc, ec=EDGE, lw=lw)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=EDGE, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
    return p

def arrow(x0, y0, x1, y1, color="black", lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=16))

def line(x0, y0, x1, y1, color="black", lw=1.8):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=1)

# ---------------- global path ----------------
ax.add_patch(Ellipse((1.35, 6.5), 1.8, 1.5, fc=LAV, ec=EDGE, lw=1.6))
ax.text(1.35, 6.5, "Global state\n$g_t$", ha="center", va="center", fontsize=12.5)
rect(2.95, 5.55, 1.5, 1.9, "Global\nencoder", LAV2)
arrow(2.25, 6.5, 2.95, 6.5)
line(4.45, 6.5, 7.65, 6.5)
arrow(7.65, 6.5, 7.65, 5.32)
ax.text(5.95, 6.66, r"$\gamma(g_t),\ \beta(g_t)$", ha="center", fontsize=13)

# ---------------- identity token ----------------
rect(0.4, 3.68, 2.3, 1.15, "Learnable\nsector embedding\n$E_i$", YEL, fs=11.5)
arrow(2.7, 4.25, 6.9, 4.25)
ax.text(4.8, 4.4, "$E_i$", ha="center", fontsize=12)

# ---------------- FiLM ----------------
rect(6.9, 3.1, 1.5, 2.2, "State\nconditioning\n(FiLM)", YEL, fs=12)

# FiLM -> plus
plus = (10.45, 4.25)
arrow(8.4, 4.25, plus[0]-0.27, 4.25)
ax.text(9.4, 4.47, r"$\tilde{x}_{i,t} = \gamma(g_t) \odot E_i + \beta(g_t)$",
        ha="center", fontsize=10.5)
ax.add_patch(Circle(plus, 0.27, fc="white", ec=EDGE, lw=1.6, zorder=3))
ax.text(plus[0], plus[1]-0.02, "+", ha="center", va="center", fontsize=17, zorder=4)

# ---------------- local bypass 1: served reports ----------------
rect(0.4, 1.78, 2.1, 1.0, "Served-report\nset  $u_{i,t}$", LAV, fs=11.5, rounded=True)
rect(3.05, 1.78, 1.7, 1.0, "Report MLP\n$\\phi_r$", SAL, fs=11.5)
arrow(2.5, 2.28, 3.05, 2.28)
line(4.75, 2.28, 10.25, 2.28, color=RED)
arrow(10.25, 2.28, 10.35, 3.99, color=RED)
ax.text(7.5, 2.44, r"$\phi_r(u_{i,t})$", ha="center", fontsize=12, color=RED)

# ---------------- local bypass 2: demand physics ----------------
rect(0.4, 0.5, 2.1, 1.0, "Demand-physics\nscalars  $d_{i,t}$", LAV, fs=11.5, rounded=True)
rect(3.05, 0.5, 1.7, 1.0, "Demand MLP\n$\\phi_d$", SAL, fs=11.5)
arrow(2.5, 1.0, 3.05, 1.0)
line(4.75, 1.0, 10.62, 1.0, color=RED)
arrow(10.62, 1.0, 10.53, 3.99, color=RED)
ax.text(7.5, 1.16, r"$\phi_d(d_{i,t})$", ha="center", fontsize=12, color=RED)

ax.text(4.95, 0.14, "demand mass in best-server region · demand-weighted own/best RSRP & SINR · "
                    "band-edge masses · deficits   (shape-invariant, 11 scalars)",
        ha="center", fontsize=8.5, color="#666677")

# ---------------- token formula ----------------
ax.text(9.9, 5.32, r"$x_{i,t} = \tilde{x}_{i,t} + \phi_r(u_{i,t}) + \phi_d(d_{i,t})$",
        ha="center", fontsize=12)
ax.text(13.9, 1.35, "red paths: additive bypass — per-sector content joins AFTER FiLM,\n"
                    "outside the $\\gamma\\odot(\\cdot)$ multiplication (unlike Residual-E)",
        ha="center", fontsize=10, color=RED, style="italic")

# ---------------- transformer (x L) ----------------
ax.add_patch(Rectangle((11.45, 2.9), 1.9, 2.7, fc="none", ec="#999999", lw=1.2, ls=(0, (4, 3))))
ax.text(12.4, 5.72, r"$\times\ L$", ha="center", fontsize=13)
rect(11.75, 3.3, 1.3, 1.9, "Transformer\nblock", BLU, fs=11.5)
arrow(plus[0]+0.27, 4.25, 11.75, 4.25)

# ---------------- per-sector head (x 9) ----------------
ax.add_patch(Rectangle((13.65, 2.9), 3.6, 2.7, fc="none", ec="#999999", lw=1.2, ls=(0, (4, 3))))
ax.text(15.45, 5.72, r"$\times\ 9$ sectors", ha="center", fontsize=13)
rect(13.9, 3.45, 1.3, 1.6, "Shared\nQ-head", GRY, fs=11.5)
rect(16.1, 3.7, 1.05, 1.1, "arg max", GRY, fs=11.5)
arrow(13.05, 4.25, 13.9, 4.25)
arrow(15.2, 4.25, 16.1, 4.25)
ax.text(15.65, 5.18, r"$Q_i(-1,\ 0,\ +1)$", ha="center", fontsize=10.5)

# ---------------- output ----------------
rect(17.65, 3.55, 1.5, 1.4, "$a_t =$\n$[a_1, \\ldots, a_9]$", PNK, fs=12)
arrow(17.15, 4.25, 17.65, 4.25)

fig.savefig(os.path.join(HERE, "fig_tok3_arch.png"), bbox_inches="tight", facecolor="white")
print("wrote fig_tok3_arch.png")
