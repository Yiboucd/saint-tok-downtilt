"""Reward-parameter robustness sweep (eval-only, frozen policies, frozen 90 cases).
Greedy trajectories are reward-independent, so each (case, policy) is rolled once;
the final/initial maps are scored under a grid of objective-constant variants.
Writes _smoke/v12_se_param_sweep.json"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
sys.path.insert(0, "/home/yibo/vcode/data/rl_projects/codex_isolated_spectral_naive_baselines_20260828_code")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T
from experiments import train_v1_mismatch_bdqn as legacy
from flat_joint_dqn import FlatJointDQN, decode_joint_index_numpy

D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
RP, SF = D + "/track_a_v12_rewardprobe", D + "/track_a_v12_se_final"
CX = "/home/yibo/vcode/data/rl_projects/codex_isolated_spectral_naive_baselines_20260828"
EST = D + "/v1_2_estimator_clean_20260818/model/best_v1_oracle_gt_m_estimator.pth"
args = T.build_parser().parse_args(["--mode","eval","--task","v1_2","--alpha-grid","0.25",
    "--demand-source","estimated","--state-map-size","256",
    "--cache-root","results/sac_9e6453c7_tilt_sweep_2e8_90x9",
    "--clean-estimator-checkpoint",EST,"--output-dir","/tmp/x","--policy","x=ppo=x",
    "--device","cuda:0","--eval-repeats","10"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(args, dev)
T.install_task(args, cache)
Z = torch.zeros((1, 18), device=dev)

# spec point first; then one-at-a-time perturbations
VARIANTS = [("spec",       0.6, 4.4, -10.0, -100.0),
            ("gate-105",   0.6, 4.4, -10.0, -105.0),
            ("gate-95",    0.6, 4.4, -10.0,  -95.0),
            ("gate-90",    0.6, 4.4, -10.0,  -90.0),
            ("atten0.5",   0.5, 4.4, -10.0, -100.0),
            ("atten0.7",   0.7, 4.4, -10.0, -100.0),
            ("cap4.0",     0.6, 4.0, -10.0, -100.0),
            ("cap4.8",     0.6, 4.8, -10.0, -100.0),
            ("capNR7.4",   0.6, 7.4, -10.0, -100.0),
            ("floor-6",    0.6, 4.4,  -6.0, -100.0),
            ("floor-14",   0.6, 4.4, -14.0, -100.0)]

def scores(angles, dens):
    """One composition -> U under every variant."""
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d * v.float(); d = d / d.sum().clamp_min(1e-12)
    out = []
    for _, a, cap, fl, gt in VARIANTS:
        se = torch.clamp(a * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=cap)
        se = torch.where(sinr < fl, torch.zeros_like(se), se) * (rss >= gt).float()
        out.append(float((d * se).sum()))
    return np.array(out)

paths = {
    "flat":      [(CX + "/seed2026081%d/flat_joint_dqn_naive/final.pt" % s, "flat") for s in (0, 1, 2)],
    "ppo":       [(SF + "/seed2026081%d/saint_ppo/final.pt" % s, "auto") for s in (0, 1, 2)],
    "myopic":    [(SF + "/seed20260810/myopic_tok3/final.pt", "auto")],
    "identity":  [(SF + "/seed2026081%d/saint_dqn/final.pt" % s, "auto") for s in (0, 1, 2)],
    "qmix":      [(SF + "/seed2026081%d/saint_qmix/final.pt" % s, "auto") for s in (0, 1, 2)],
    "sac":       [(SF + "/seed2026081%d/saint_sac/final.pt" % s, "auto") for s in (0, 1, 2)],
    "naive":     [(SF + "/seed2026081%d/bdqn_naive/final.pt" % s, "auto") for s in (0, 1, 2)],
    "bdqn_t":    [(RP + "/spectral_bdqn_s10/final.pt", "auto"),
                  (SF + "/seed20260811/bdqn/final.pt", "auto"),
                  (SF + "/seed20260812/bdqn/final.pt", "auto")],
    "ddqn_tok3": [(RP + "/spectral_tok3_s10/final.pt", "auto"),
                  (SF + "/seed20260811/saint_dqn_tok3/final.pt", "auto"),
                  (SF + "/seed20260812/saint_dqn_tok3/final.pt", "auto")],
    "proposed":  [(SF + "/seed2026081%d/saint_sac_tok3/final.pt" % s, "auto") for s in (0, 1, 2)],
}
pol = {"const": ("const", None), "random": ("random", None)}
for fam, entries in paths.items():
    for i, (p, mode) in enumerate(entries):
        if mode == "flat":
            m = FlatJointDQN().to(dev)
            pay = torch.load(p, map_location="cpu", weights_only=False)
            m.load_state_dict(pay["model_state"]); m.eval()
            pol[fam + "_s%d" % i] = ("flat", m)
        else:
            m, pay = T.load_track_a_policy(Path(p), dev)
            if hasattr(m, "attach_cache"):
                m.attach_cache(engine.base_dbm, cache.num_angles)
            kind = "ppo" if pay.get("algo") == "saint_ppo" else "bdqn"
            pol[fam + "_s%d" % i] = (kind, m)
print("loaded", len(pol), "policies", flush=True)

env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
rec = {k: [] for k in pol}
ci = 0
for gi, (g, fams) in enumerate(groups):
    for fi, f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
            state0 = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
            u0 = scores(env.angles.copy(), env.density_true)
            dens = env.density_true
            for name, (kind, m) in pol.items():
                state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                rng = np.random.default_rng(gate.derived_seed(cs, "random_policy"))
                done = False
                while not done:
                    if kind == "const":
                        a = np.zeros(9, dtype=np.int64)
                    elif kind == "random":
                        a = rng.integers(0, 3, 9).astype(np.int64)
                    elif kind == "flat":
                        with torch.no_grad():
                            q = m(*legacy.state_tensors(state, dev))[0]
                        a = decode_joint_index_numpy(int(torch.argmax(q).item()))
                    else:
                        a = T.greedy_action(m, kind, state, dev)
                    state, _, done = env.step(a)
                rec[name].append(scores(env.angles.copy(), dens) - u0)
            ci += 1
            if ci % 15 == 0:
                print(ci, "cases", flush=True)

fams_order = ["random", "const", "flat", "ppo", "myopic", "identity", "qmix", "naive", "sac", "bdqn_t", "ddqn_tok3", "proposed"]
def fam_mean(fam):
    ks = [k for k in rec if k == fam or k.startswith(fam + "_s")]
    return np.mean([np.mean(np.stack(rec[k]), axis=0) for k in ks], axis=0)  # [n_variants]
M = {fam: fam_mean(fam) for fam in fams_order}
out = {"variants": [v[0] for v in VARIANTS],
       "family_means": {fam: [float(x) for x in M[fam]] for fam in fams_order}}
json.dump(out, open(D + "/_smoke/v12_se_param_sweep.json", "w"), indent=2)

from scipy.stats import spearmanr
spec = np.array([M[f][0] for f in fams_order])
print()
print("%-10s" % "family" + "".join("%9s" % v[0] for v in VARIANTS))
for fam in fams_order:
    print("%-10s" % fam + "".join("%+9.3f" % x for x in M[fam]))
print()
for j, v in enumerate(VARIANTS):
    col = np.array([M[f][j] for f in fams_order])
    rho = spearmanr(spec, col).correlation
    top = fams_order[int(np.argmax(col))]
    margin = col[fams_order.index("proposed")] - sorted(col)[-2]
    print("%-9s spearman-vs-spec %.3f   top=%s   proposed margin over runner-up %+.4f" % (v[0], rho, top, margin))
print("saved")
