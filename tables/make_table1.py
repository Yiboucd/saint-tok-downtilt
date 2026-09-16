# -*- coding: utf-8 -*-
"""Reproduce Table I and the headline paired confidence intervals of the paper
from the frozen evaluation records in ../results/ -- no training, no GPU.

Table I columns: dU (bps/Hz) over all 90 paired episodes, over the 50
seen-family episodes (per-case entries 0-49) and over the 40 held-out-family
episodes (entries 50-89), plus the excellent-band demand share at episode end
(RSRP >= -80 dBm and SINR >= 20 dB).  The relative SE gain (dU / initial
demand-weighted SE) quoted in the text is printed as an extra column.

Also printed: the margin over the best constant policy on each split and its
held-out retention (held-out margin / seen margin) for every learned row, and
the four headline paired bootstrap CIs.

Usage:  python make_table1.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

N_SEEN = 50   # per-case entries 0..49 = seen-family episodes, 50..89 = held-out families


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


fin = load("v12_se_final_eval.json")        # random/const/ppo/myopic/identity/bdqn(tailored)/tok3+ddqn
nb = load("v12_se_newbase_eval.json")       # sac / qmix (identity tokens)
nv = load("v12_se_naive_eval.json")         # branching dqn (generic encoder)
fl = load("v12_se_flat_eval.json")          # flat joint 3^9 dqn
st = load("v12_se_sactok3_final.json")      # proposed: tok3 + discrete sac (dSE only)
sb = load("v12_se_sactok3_full.json")["clean"]  # proposed: clean-rung band metrics

INITIAL_SE = fin["proposed_s0"]["initial"]["se"]  # 1.8399 bps/Hz


def fam(src, prefix, per_case_key="per_case_dse"):
    ks = sorted(k for k in src if k == prefix or k.startswith(prefix))
    per_case = np.mean([np.asarray(src[k][per_case_key]) for k in ks], axis=0)
    assert per_case.size == 90, (prefix, per_case.size)
    if "dSE_ood" in src[ks[0]]:
        # the held-out mean stored in the record must equal the last-40 per-case mean
        ood_rec = float(np.mean([src[k]["dSE_ood"] for k in ks]))
        assert abs(ood_rec - per_case[N_SEEN:].mean()) < 1e-4, (prefix, ood_rec, per_case[N_SEEN:].mean())
    exc = None
    if "final" in src[ks[0]]:
        exc = float(np.mean([src[k]["final"]["excellent"] for k in ks]))
    return per_case, exc, len(ks)


rows = []
for label, src, prefix, pck in [
    ("Random", fin, "random", "per_case_dse"),
    ("Best constant", fin, "const", "per_case_dse"),
    ("Flat joint DQN (3^9)", fl, "flat_s", "per_case_dse"),
    ("PPO (identity tokens)", fin, "ppo_s", "per_case_dse"),
    ("One-step (gamma=0)", fin, "myopic_s", "per_case_dse"),
    ("SAINT (identity tokens)", fin, "saint_id_s", "per_case_dse"),
    ("QMIX (identity tokens)", nb, "saint_qmix_s", "per_case_dse"),
    ("Branching DQN", nv, "bdqn_naive_s", "per_case_dse"),
    ("Discrete SAC (identity)", nb, "saint_sac_s", "per_case_dse"),
    ("tailored-encoder BDQN", fin, "prev_bdqn_s", "per_case_dse"),
    ("Tok3 + double-Q (abl.)", fin, "proposed_s", "per_case_dse"),
]:
    per_case, exc, n = fam(src, prefix, pck)
    rows.append((label, per_case, exc, n))

# proposed (per-case in sactok3_final; bands in sactok3_full clean rung)
prop_pc = np.mean([np.asarray(st["s%d" % i]["per_case"]) for i in range(3)], axis=0)
prop_ood = float(np.mean([st["s%d" % i]["ood"] for i in range(3)]))
assert abs(prop_ood - prop_pc[N_SEEN:].mean()) < 1e-4
prop_exc = float(np.mean([sb["s%d" % i]["excellent"] for i in range(3)]))
rows.append(("PROPOSED (Tok3 + SAC)", prop_pc, prop_exc, 3))

# ---------------------------------------------------------------- Table I
print("Table I: dU (bps/Hz) over all 90 / 50 seen / 40 held-out episodes; initial SE = %.2f bps/Hz" % INITIAL_SE)
print("(printed at 4 decimals; Table I in the paper shows 3)")
print("%-26s %8s %8s %9s %10s %9s %2s" % ("Method", "dU", "seen", "held-out", "Excellent", "SE gain", "n"))
for label, pc, exc, n in rows:
    m, s, h = pc.mean(), pc[:N_SEEN].mean(), pc[N_SEEN:].mean()
    print("%-26s %+8.4f %+8.4f %+9.4f %9s %+8.1f%% %2d" % (
        label, m, s, h,
        ("%.1f%%" % exc) if exc is not None else "--",
        100.0 * m / INITIAL_SE, n))

# ------------------------------------- margin over best constant, held-out retention
by = {label: pc for label, pc, _, _ in rows}
const_pc = by["Best constant"]
c_seen, c_ood = const_pc[:N_SEEN].mean(), const_pc[N_SEEN:].mean()
print()
print("Margin over best constant (method - const on the same split) and its held-out retention")
print("%-26s %8s %9s %9s" % ("Method", "seen", "held-out", "retained"))
for label, pc, _, _ in rows:
    if label in ("Random", "Best constant"):
        continue
    m_seen = pc[:N_SEEN].mean() - c_seen
    m_ood = pc[N_SEEN:].mean() - c_ood
    print("%-26s %+8.4f %+9.4f %8.0f%%" % (label, m_seen, m_ood, 100.0 * m_ood / m_seen))

# ---------------------------------------------------------- headline paired bootstrap CIs
rng = np.random.default_rng(0)


def ci(diff, nboot=20000):
    bs = np.array([np.mean(rng.choice(diff, diff.size, replace=True)) for _ in range(nboot)])
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), 100.0 * (diff > 0).mean()


print()
for a, b in [("PROPOSED (Tok3 + SAC)", "Best constant"),
             ("PROPOSED (Tok3 + SAC)", "Discrete SAC (identity)"),
             ("PROPOSED (Tok3 + SAC)", "Branching DQN"),
             ("PROPOSED (Tok3 + SAC)", "Tok3 + double-Q (abl.)")]:
    m, lo, hi, w = ci(by[a] - by[b])
    print("%s - %s: %+.4f  CI[%+.4f, %+.4f]  wins %.0f%% of episodes" % (a, b, m, lo, hi, w))
