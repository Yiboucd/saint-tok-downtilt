"""Fig.6 data under the SE-era policies: demand-weighted (dRSRP, dSINR) per family,
plus the uniform-tilt sweep under the SE objective (tuned constant + per-episode oracle).
Writes _smoke/v12_se_decomp.json"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T

ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
RP = D + "/track_a_v12_rewardprobe"
SF = D + "/track_a_v12_se_final"
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

def stats(angles, dens):
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d * v.float(); d = d / d.sum().clamp_min(1e-12)
    se = torch.clamp(ATT * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=CAP)
    se = torch.where(sinr < FLOOR, torch.zeros_like(se), se) * (rss >= PMIN).float()
    return float((d * rss).sum()), float((d * sinr).sum()), float((d * se).sum())

FAMS = {
    "proposed": [(RP + "/spectral_tok3_s10/final.pt", None)],  # placeholder replaced below
}
paths = {
    "proposed":  [SF + "/seed2026081%d/saint_sac_tok3/final.pt" % s for s in (0, 1, 2)],
    "ddqn_tok3": [RP + "/spectral_tok3_s10/final.pt", SF + "/seed20260811/saint_dqn_tok3/final.pt", SF + "/seed20260812/saint_dqn_tok3/final.pt"],
    "sac":       [SF + "/seed2026081%d/saint_sac/final.pt" % s for s in (0, 1, 2)],
    "identity":  [SF + "/seed2026081%d/saint_dqn/final.pt" % s for s in (0, 1, 2)],
    "naive":     [SF + "/seed2026081%d/bdqn_naive/final.pt" % s for s in (0, 1, 2)],
    "ppo":       [SF + "/seed2026081%d/saint_ppo/final.pt" % s for s in (0, 1, 2)],
}
pol = {}
for fam, ps in paths.items():
    for i, p in enumerate(ps):
        m, payload = T.load_track_a_policy(Path(p), dev)
        if hasattr(m, "attach_cache"):
            m.attach_cache(engine.base_dbm, cache.num_angles)
        kind = "ppo" if payload.get("algo") == "saint_ppo" else "bdqn"
        pol[fam + "_s%d" % i] = (kind, m)
print("loaded", len(pol), flush=True)
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
rec = {k: {"dr": [], "ds": []} for k in pol}
uni = {"best_dse": [], "best_tilt": [], "dse_at": {u: [] for u in range(21)}}
ci = 0
for gi, (g, fams) in enumerate(groups):
    for fi, f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
            state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
            dens = env.density_true
            r0, s0, se0 = stats(env.angles.copy(), dens)
            best, bestu = -1e9, 0
            for u in range(21):
                _, _, seu = stats(np.full(9, float(u), dtype=np.float32), dens)
                uni["dse_at"][u].append(seu - se0)
                if seu - se0 > best:
                    best, bestu = seu - se0, u
            uni["best_dse"].append(best); uni["best_tilt"].append(bestu)
            for name, (kind, m) in pol.items():
                state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                done = False
                while not done:
                    a = T.greedy_action(m, kind, state, dev)
                    state, _, done = env.step(a)
                r1, s1, _ = stats(env.angles.copy(), dens)
                rec[name]["dr"].append(r1 - r0)
                rec[name]["ds"].append(s1 - s0)
            ci += 1
            if ci % 15 == 0:
                print(ci, "cases", flush=True)
out = {"policies": {}, "uniform": {}}
for k, v in rec.items():
    out["policies"][k] = {"dRSRP": float(np.mean(v["dr"])), "dSINR": float(np.mean(v["ds"]))}
out["uniform"] = {"mean_dse_by_tilt": {u: float(np.mean(x)) for u, x in uni["dse_at"].items()},
                  "oracle_per_episode_dse": float(np.mean(uni["best_dse"])),
                  "oracle_tilt_hist": {int(u): int((np.array(uni["best_tilt"]) == u).sum()) for u in set(uni["best_tilt"])}}
json.dump(out, open(D + "/_smoke/v12_se_decomp.json", "w"), indent=2)
print()
for fam in paths:
    dr = np.mean([out["policies"][fam + "_s%d" % i]["dRSRP"] for i in range(3)])
    ds = np.mean([out["policies"][fam + "_s%d" % i]["dSINR"] for i in range(3)])
    print("%-10s dRSRP %+6.2f dB   dSINR %+6.2f dB" % (fam, dr, ds))
mb = out["uniform"]["mean_dse_by_tilt"]
bu = max(mb, key=lambda u: mb[u])
print("uniform sweep: best mean tilt = %d deg, dSE %+0.4f;  per-episode oracle %+0.4f" % (
    int(bu), mb[bu], out["uniform"]["oracle_per_episode_dse"]))
print("saved")
