"""Frozen-90 eval of the tok3+SAC probe; paired vs SE-trained tok3 (same seed + 3-seed)."""
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
m,_ = T.load_track_a_policy(Path(f"{D}/track_a_v12_se_final/seed20260810/saint_sac_tok3/final.pt"), dev)
if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
groups = [("seen", gate.parse_list(args.train_families)), ("ood", gate.parse_list(args.ood_families))]
dse, grp = [], []; ci = 0
for gi,(g,fams) in enumerate(groups):
    for fi,f in enumerate(fams):
        for rep in range(10):
            cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
            state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
            u0 = dse0(env.angles.copy(), env.density_true); dens = env.density_true
            done = False
            while not done:
                a = T.greedy_action(m, "bdqn", state, dev); state,_,done = env.step(a)
            dse.append(dse0(env.angles.copy(), dens)-u0); grp.append(g); ci += 1
            if ci%30==0: print(ci,"cases",flush=True)
dse=np.array(dse); grp=np.array(grp)
json.dump({"dSE":float(dse.mean()),"seen":float(dse[grp=="seen"].mean()),"ood":float(dse[grp=="ood"].mean()),
           "per_case":[float(x) for x in dse]}, open(f"{D}/_smoke/v12_se_sactok3_eval.json","w"), indent=2)
fin=json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
r0=np.array(fin["proposed_s0"]["per_case_dse"]); r3=np.mean([fin[f"proposed_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
rng=np.random.default_rng(0)
def ci95(d):
    bs=np.array([np.mean(rng.choice(d,90,replace=True)) for _ in range(20000)])
    return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d>0).mean()
print()
print("tok3+SAC (seed .810): dSE %+.4f   seen %+.4f   ood %+.4f"%(dse.mean(),dse[grp=="seen"].mean(),dse[grp=="ood"].mean()))
m1,lo,hi,w = ci95(dse-r0); print("vs tok3+DDQN same seed (%+.4f): %+.4f CI[%+.4f,%+.4f] win %.0f%%"%(r0.mean(),m1,lo,hi,w))
m2,lo,hi,w = ci95(dse-r3); print("vs tok3+DDQN 3-seed (%+.4f):   %+.4f CI[%+.4f,%+.4f] win %.0f%%"%(r3.mean(),m2,lo,hi,w))
print("saved")
