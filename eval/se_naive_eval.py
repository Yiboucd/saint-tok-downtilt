"""Frozen-90 greedy eval of bdqn_naive x3 under the spectral reward.
Same case seeds/order as v12_se_final_eval.json. Writes _smoke/v12_se_naive_eval.json"""
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
pol = {}
for i,s in enumerate((0,1,2)):
    p = Path(f"{D}/track_a_v12_se_final/seed2026081{s}/bdqn_naive/final.pt")
    m,_ = T.load_track_a_policy(p, dev)
    if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    pol[f"bdqn_naive_s{i}"] = m
print("loaded", sorted(pol), flush=True)
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
rec = {k: {"dse": [], "grp": [], "fin": []} for k in pol}; ci = 0
for gi,(g,fams) in enumerate(groups):
    for fi,f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
            for name,m in pol.items():
                state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                i0 = qual(env.angles.copy(), env.density_true); dens = env.density_true
                done = False
                while not done:
                    a = T.greedy_action(m, "bdqn", state, dev); state,_,done = env.step(a)
                fq = qual(env.angles.copy(), dens)
                rec[name]["dse"].append(fq["se"]-i0["se"]); rec[name]["grp"].append(g); rec[name]["fin"].append(fq)
            ci += 1
            if ci%30==0: print(ci,"cases",flush=True)
out = {}
for k,v in rec.items():
    a=np.array(v["dse"]); gr=np.array(v["grp"])
    out[k]={"dSE":float(a.mean()),"dSE_ood":float(a[gr=="ood"].mean()),
            "final":{kk: float(np.mean([q[kk] for q in v["fin"]])) for kk in v["fin"][0]},
            "per_case_dse":[float(x) for x in a]}
json.dump(out, open(f"{D}/_smoke/v12_se_naive_eval.json","w"), indent=2)
fin=json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
N=np.mean([out[f"bdqn_naive_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
P=np.mean([fin[f"proposed_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
B=np.mean([fin[f"prev_bdqn_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
rng=np.random.default_rng(0)
def ci95(d):
    bs=np.array([np.mean(rng.choice(d,90,replace=True)) for _ in range(20000)])
    return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d>0).mean()
print()
print("naive seeds:", "  ".join("%+.4f"%out[f"bdqn_naive_s{i}"]["dSE"] for i in range(3)),
      " mean %+.4f  ood %+.4f"%(N.mean(), float(np.mean([out[f"bdqn_naive_s{i}"]["dSE_ood"] for i in range(3)]))))
m,lo,hi,w = ci95(P-N); print("proposed - naive:    %+.4f  CI[%+.4f,%+.4f]  win %.0f%%"%(m,lo,hi,w))
m,lo,hi,w = ci95(B-N); print("tailored - naive:    %+.4f  CI[%+.4f,%+.4f]  win %.0f%%"%(m,lo,hi,w))
print("saved")
