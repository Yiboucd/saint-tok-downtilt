"""Frozen-90 eval of Codex FlatJointDQN (3 seeds, joint 3^9 head).
Writes _smoke/v12_se_flat_eval.json"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
sys.path.insert(0, "/home/yibo/vcode/data/rl_projects/codex_isolated_spectral_naive_baselines_20260828_code")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T
import train_v1_mismatch_bdqn as legacy
from flat_joint_dqn import FlatJointDQN, decode_joint_index_numpy

ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
BANDS = [("excellent", -80.0, 20.0), ("good", -90.0, 13.0), ("fair", -100.0, 0.0)]
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
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

def qual(angles, dens):
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d * v.float(); d = d / d.sum().clamp_min(1e-12)
    o = {}
    cov = torch.zeros_like(rss, dtype=torch.bool)
    for n, r, s in BANDS:
        m = (rss >= r) & (sinr >= s) & ~cov
        o[n] = float((d * m.float()).sum() * 100); cov |= m
    o["poor"] = float((d * (~cov).float()).sum() * 100)
    se = torch.clamp(ATT * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=CAP)
    se = torch.where(sinr < FLOOR, torch.zeros_like(se), se) * (rss >= PMIN).float()
    o["se"] = float((d * se).sum()); o["outage"] = float((d * (rss < PMIN).float()).sum())
    return o

pol = {}
for i, s in enumerate((0, 1, 2)):
    m = FlatJointDQN().to(dev)
    p = torch.load(CX + "/seed2026081%d/flat_joint_dqn_naive/final.pt" % s, map_location="cpu", weights_only=False)
    m.load_state_dict(p["model_state"]); m.eval()
    pol["flat_s%d" % i] = m
print("loaded", sorted(pol), flush=True)
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
rec = {k: {"dse": [], "grp": [], "fin": []} for k in pol}
ci = 0
for gi, (g, fams) in enumerate(groups):
    for fi, f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
            for name, m in pol.items():
                state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                i0 = qual(env.angles.copy(), env.density_true)
                dens = env.density_true
                done = False
                while not done:
                    with torch.no_grad():
                        q = m(*legacy.state_tensors(state, dev))[0]
                    a = decode_joint_index_numpy(int(torch.argmax(q).item()))
                    state, _, done = env.step(a)
                fq = qual(env.angles.copy(), dens)
                rec[name]["dse"].append(fq["se"] - i0["se"])
                rec[name]["grp"].append(g)
                rec[name]["fin"].append(fq)
            ci += 1
            if ci % 30 == 0:
                print(ci, "cases", flush=True)
out = {}
for k, v in rec.items():
    a = np.array(v["dse"]); gr = np.array(v["grp"])
    out[k] = {"dSE": float(a.mean()), "dSE_ood": float(a[gr == "ood"].mean()),
              "final": {kk: float(np.mean([q[kk] for q in v["fin"]])) for kk in v["fin"][0]},
              "per_case_dse": [float(x) for x in a]}
json.dump(out, open(D + "/_smoke/v12_se_flat_eval.json", "w"), indent=2)
A = np.mean([out["flat_s%d" % i]["per_case_dse"] for i in range(3)], axis=0)
print()
print("flat seeds:", "  ".join("%+.4f" % out["flat_s%d" % i]["dSE"] for i in range(3)),
      " mean %+.4f  ood %+.4f" % (A.mean(), float(np.mean([out["flat_s%d" % i]["dSE_ood"] for i in range(3)]))))
print("final SE %.4f  exc %.1f%%  poor %.1f%%" % (
    np.mean([out["flat_s%d" % i]["final"]["se"] for i in range(3)]),
    np.mean([out["flat_s%d" % i]["final"]["excellent"] for i in range(3)]),
    np.mean([out["flat_s%d" % i]["final"]["poor"] for i in range(3)])))
print("saved")
