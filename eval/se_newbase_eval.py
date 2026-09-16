"""Frozen-90 greedy eval of the 6 new baseline checkpoints (saint_sac x3, saint_qmix x3).
Same case seeds/order as v12_se_final_eval.json; merges and prints the full table.
Writes _smoke/v12_se_newbase_eval.json"""
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
for a in ("saint_sac","saint_qmix"):
    for i,s in enumerate((0,1,2)):
        p = Path(f"{D}/track_a_v12_se_final/seed2026081{s}/{a}/final.pt")
        m,_ = T.load_track_a_policy(p, dev)
        if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
        pol[f"{a}_s{i}"] = m
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
json.dump(out, open(f"{D}/_smoke/v12_se_newbase_eval.json","w"), indent=2)
fin=json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
def fam(src,pre,n):
    ks=[k for k in src if k.startswith(pre)]
    return (float(np.mean([src[k]["dSE"] for k in ks])), float(np.mean([src[k]["dSE_ood"] for k in ks])),
            float(np.mean([src[k]["final"]["excellent"] for k in ks])), float(np.mean([src[k]["final"]["outage"] for k in ks])), len(ks))
print()
print("%-12s %8s %8s %9s %8s %s" % ("policy","dSE","dSE_ood","excellent","outage","n"))
rows = [("random",fin,"random"),("const",fin,"const"),("ppo",fin,"ppo_s"),("sac",out,"saint_sac_s"),
        ("qmix",out,"saint_qmix_s"),("myopic",fin,"myopic_s"),("saint_id",fin,"saint_id_s"),
        ("prev_bdqn",fin,"prev_bdqn_s"),("proposed",fin,"proposed_s")]
for lbl,src,pre in rows:
    a,b,c,e,n = fam(src,pre,None)
    print("%-12s %+8.4f %+8.4f %8.1f%% %7.4f  %d" % (lbl,a,b,c,e,n))
print("saved")
