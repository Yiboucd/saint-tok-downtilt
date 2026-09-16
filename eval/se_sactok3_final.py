"""Evaluate sac_tok3 seeds 811/812, merge with s10, print the 3-seed-vs-3-seed verdict."""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T
ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
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
def dse0(angles, dens):
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad(): rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d*v.float(); d = d/d.sum().clamp_min(1e-12)
    se = torch.clamp(ATT*torch.log2(1.0+torch.pow(10.0, sinr/10.0)), max=CAP)
    se = torch.where(sinr<FLOOR, torch.zeros_like(se), se)*(rss>=PMIN).float()
    return float((d*se).sum())
pol = {}
for s in (1, 2):
    p = Path(f"{D}/track_a_v12_se_final/seed2026081{s}/saint_sac_tok3/final.pt")
    m,_ = T.load_track_a_policy(p, dev)
    if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    pol[f"s{s}"] = m
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
rec = {k: {"dse": [], "grp": []} for k in pol}; ci = 0
for gi,(g,fams) in enumerate(groups):
    for fi,f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
            for name,m in pol.items():
                state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                u0 = dse0(env.angles.copy(), env.density_true); dens = env.density_true
                done = False
                while not done:
                    a = T.greedy_action(m, "bdqn", state, dev); state,_,done = env.step(a)
                rec[name]["dse"].append(dse0(env.angles.copy(), dens)-u0); rec[name]["grp"].append(g)
            ci += 1
            if ci%30==0: print(ci,"cases",flush=True)
s10 = json.load(open(f"{D}/_smoke/v12_se_sactok3_eval.json"))
out = {"s0": {"dSE": s10["dSE"], "seen": s10["seen"], "ood": s10["ood"], "per_case": s10["per_case"]}}
for k,v in rec.items():
    a=np.array(v["dse"]); gr=np.array(v["grp"])
    out[k]={"dSE":float(a.mean()),"seen":float(a[gr=="seen"].mean()),"ood":float(a[gr=="ood"].mean()),
            "per_case":[float(x) for x in a]}
json.dump(out, open(f"{D}/_smoke/v12_se_sactok3_final.json","w"), indent=2)
fin=json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
S=np.mean([out[f"s{i}"]["per_case"] for i in range(3)],axis=0)
P=np.mean([fin[f"proposed_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
rng=np.random.default_rng(0)
bs=np.array([np.mean(rng.choice(S-P,90,replace=True)) for _ in range(20000)])
d=S-P
print()
print("sac_tok3 seeds:", "  ".join("%+.4f"%out[f"s{i}"]["dSE"] for i in range(3)), " mean %+.4f"%S.mean())
print("  seen %+.4f  ood %+.4f"%(float(np.mean([out[f"s{i}"]["seen"] for i in range(3)])), float(np.mean([out[f"s{i}"]["ood"] for i in range(3)]))))
print("ddqn_tok3 3-seed mean %+.4f"%P.mean())
print("sac_tok3 - ddqn_tok3: %+.4f  CI[%+.4f,%+.4f]  win %.0f%%"%(d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d>0).mean()))
print("saved")
