"""Evaluate the 20k-episode tok3 probe on the same 90 frozen cases; pair against the 10k runs."""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T
ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
BANDS = [("excellent", -80.0, 20.0), ("good", -90.0, 13.0), ("fair", -100.0, 0.0)]
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
EST = f"{D}/v1_2_estimator_clean_20260818/model/best_v1_oracle_gt_m_estimator.pth"
args = T.build_parser().parse_args(["--mode","eval","--task","v1_2","--alpha-grid","0.25",
    "--demand-source","estimated","--state-map-size","256",
    "--cache-root","results/sac_9e6453c7_tilt_sweep_2e8_90x9",
    "--clean-estimator-checkpoint",EST,"--output-dir","/tmp/x","--policy","x=ppo=x",
    "--device","cuda:0","--eval-repeats","10"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(args, dev)
T.install_task(args, cache); Z = torch.zeros((1,18), device=dev)
def qual(angles, dens):
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad(): rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d*v.float(); d = d/d.sum().clamp_min(1e-12)
    o = {}; cov = torch.zeros_like(rss, dtype=torch.bool)
    for n,r,s in BANDS:
        m = (rss>=r)&(sinr>=s)&~cov; o[n] = float((d*m.float()).sum()*100); cov |= m
    o["poor"] = float((d*(~cov).float()).sum()*100)
    se = torch.clamp(ATT*torch.log2(1.0+torch.pow(10.0, sinr/10.0)), max=CAP)
    se = torch.where(sinr<FLOOR, torch.zeros_like(se), se)*(rss>=PMIN).float()
    o["se"] = float((d*se).sum()); o["outage"] = float((d*(rss<PMIN).float()).sum())
    return o
p = Path(f"{D}/track_a_v12_se_final/seed20260810/tok3_ep20k/final.pt")
m, payload = T.load_track_a_policy(p, dev)
if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
print("loaded", payload.get("algo"), "episodes", payload.get("episodes"))
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
dse, grp, fins = [], [], []; ci = 0
for gi,(g,fams) in enumerate(groups):
    for fi,f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
            state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
            i0 = qual(env.angles.copy(), env.density_true); dens = env.density_true
            done = False
            while not done:
                a = T.greedy_action(m, "bdqn", state, dev); state,_,done = env.step(a)
            fq = qual(env.angles.copy(), dens)
            dse.append(fq["se"]-i0["se"]); grp.append(g); fins.append(fq); ci += 1
dse = np.array(dse); grp = np.array(grp)
ref = json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
r0 = np.array(ref["proposed_s0"]["per_case_dse"])
r3 = np.mean([ref[f"proposed_s{i}"]["per_case_dse"] for i in range(3)], axis=0)
d1 = dse - r0; d3 = dse - r3
rng = np.random.default_rng(0)
def ci95(x):
    bs = np.array([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(20000)])
    return np.percentile(bs,2.5), np.percentile(bs,97.5)
print()
print("20k probe (seed 20260810, 20000 episodes)")
print("  dSE all  %+.4f    seen %+.4f    ood %+.4f" % (dse.mean(), dse[grp=="seen"].mean(), dse[grp=="ood"].mean()))
print("  excellent %.1f%%   poor %.1f%%   outage %.3f%%" % (np.mean([q["excellent"] for q in fins]), np.mean([q["poor"] for q in fins]), 100*np.mean([q["outage"] for q in fins])))
print()
print("vs SAME-SEED 10k run (proposed_s0 = %+.4f):" % r0.mean())
lo,hi = ci95(d1); print("  diff %+.4f   95%% CI [%+.4f, %+.4f]   win %.0f%%" % (d1.mean(), lo, hi, 100*(d1>0).mean()))
print("vs 3-seed 10k mean (%+.4f):" % r3.mean())
lo,hi = ci95(d3); print("  diff %+.4f   95%% CI [%+.4f, %+.4f]   win %.0f%%" % (d3.mean(), lo, hi, 100*(d3>0).mean()))
json.dump({"dSE":float(dse.mean()),"seen":float(dse[grp=="seen"].mean()),"ood":float(dse[grp=="ood"].mean()),
           "per_case_dse":[float(x) for x in dse]}, open(f"{D}/_smoke/v12_se_20k_eval.json","w"), indent=2)
print("saved")
