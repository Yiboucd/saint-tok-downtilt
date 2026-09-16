"""Score the three alpha-mix-trained tok3 seeds under the SE metric (clean rung, frozen 90).
Makes the reward-swap claim 3-seed vs 3-seed. Writes _smoke/v12_se_amix3_eval.json"""
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
paths = {"amix_s10": f"{D}/track_a_v12_design/tok3_est_amix/final.pt",
         "amix_s11": f"{D}/track_a_v12_est_alpha025/seed20260811/saint_dqn_tok3/final.pt",
         "amix_s12": f"{D}/track_a_v12_est_alpha025/seed20260812/saint_dqn_tok3/final.pt"}
pol = {}
for k, p in paths.items():
    m, _ = T.load_track_a_policy(Path(p), dev)
    if hasattr(m, "attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    pol[k] = m
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
out = {}
for k,v in rec.items():
    a=np.array(v["dse"]); gr=np.array(v["grp"])
    out[k]={"dSE":float(a.mean()),"seen":float(a[gr=="seen"].mean()),"ood":float(a[gr=="ood"].mean()),
            "per_case":[float(x) for x in a]}
json.dump(out, open(f"{D}/_smoke/v12_se_amix3_eval.json","w"), indent=2)
fin=json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
A=np.mean([out[f"amix_s1{i}"]["per_case"] for i in (0,1,2)],axis=0)
P=np.mean([fin[f"proposed_s{i}"]["per_case_dse"] for i in range(3)],axis=0)
rng=np.random.default_rng(0); d=P-A
bs=np.array([np.mean(rng.choice(d,90,replace=True)) for _ in range(20000)])
print()
print("amix-trained tok3 seeds (SE metric):", "  ".join("%+.4f"%out[f"amix_s1{i}"]["dSE"] for i in (0,1,2)), " mean %+.4f"%A.mean())
print("SE-trained  tok3 3-seed mean: %+.4f" % P.mean())
print("reward swap (SE-trained - amix-trained): %+.4f  (%+.1f%%)  CI[%+.4f,%+.4f]  win %.0f%%"
      %(d.mean(),100*d.mean()/A.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d>0).mean()))
print("saved")
