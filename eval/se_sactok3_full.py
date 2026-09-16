"""Decision pack for sac_tok3: (1) clean-rung band/outage metrics, (2) B1 severity ladder.
Policies: sac_tok3 x3. Writes _smoke/v12_se_sactok3_full.json"""
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
    "--device","cuda:0","--eval-repeats","10","--protocol","joint_on_grid"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(args, dev)
T.install_task(args, cache)
def qual(latent, angles, dens, full=False):
    lat = torch.as_tensor(latent, dtype=torch.float32, device=dev)
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad(): rss, sinr = engine.compose(lat, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d*v.float(); d = d/d.sum().clamp_min(1e-12)
    se = torch.clamp(ATT*torch.log2(1.0+torch.pow(10.0, sinr/10.0)), max=CAP)
    se = torch.where(sinr<FLOOR, torch.zeros_like(se), se)*(rss>=PMIN).float()
    o = {"se": float((d*se).sum())}
    if full:
        cov = torch.zeros_like(rss, dtype=torch.bool)
        for n,r,s in BANDS:
            m = (rss>=r)&(sinr>=s)&~cov; o[n] = float((d*m.float()).sum()*100); cov |= m
        o["poor"] = float((d*(~cov).float()).sum()*100)
        o["outage"] = float((d*(rss<PMIN).float()).sum())
    return o
orig = gate.sample_joint_latent
MODE = {"kind": "clean"}
def patched(protocol, rng):
    lat = orig(protocol, rng)
    if MODE["kind"] == "clean": lat[:, :] = 0.0
    elif MODE["kind"] == "bias_only": lat[:, 9:] = 0.0
    elif MODE["kind"] == "tilt1": lat[:, 9:] = np.clip(lat[:, 9:], -1.0, 1.0)
    return lat
gate.sample_joint_latent = patched
pol = {}
for i,s in enumerate((0,1,2)):
    m,_ = T.load_track_a_policy(Path(f"{D}/track_a_v12_se_final/seed2026081{s}/saint_sac_tok3/final.pt"), dev)
    if hasattr(m,"attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    pol[f"s{i}"] = m
families = gate.parse_list(args.train_families)
groups = [("seen", families), ("ood", gate.parse_list(args.ood_families))]
out = {}
for rung in ("clean", "bias_only", "tilt1", "tilt2"):
    MODE["kind"] = "clean" if rung=="clean" else ("bias_only" if rung=="bias_only" else ("tilt1" if rung=="tilt1" else "raw"))
    env = gate.StrictPrivilegedControllerEnv(cache, engine, est, dev, args, "B1", families, int(args.eval_tape_seed))
    res = {p: {"dse": [], "fin": []} for p in pol}
    ci = 0
    for gi,(g,fams) in enumerate(groups):
        for fi,f in enumerate(fams):
            for rep in range(10):
                cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
                for name,m in pol.items():
                    state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                    u0 = qual(env.latent, env.angles.copy(), env.density_true)["se"]
                    dens = env.density_true; done = False
                    while not done:
                        a = T.greedy_action(m, "bdqn", state, dev); state,_,done = env.step(a)
                    fq = qual(env.latent, env.angles.copy(), dens, full=(rung=="clean"))
                    res[name]["dse"].append(fq["se"]-u0); res[name]["fin"].append(fq)
                ci += 1
                if ci%30==0: print(rung, ci, flush=True)
    out[rung] = {}
    for p,v in res.items():
        a = np.array(v["dse"])
        e = {"dSE": float(a.mean()), "per_case": [float(x) for x in a]}
        if rung == "clean":
            for kk in ("excellent","good","fair","poor","outage","se"):
                e[kk] = float(np.mean([q[kk] for q in v["fin"]]))
        out[rung][p] = e
json.dump(out, open(f"{D}/_smoke/v12_se_sactok3_full.json","w"), indent=2)
sev = json.load(open(f"{D}/_smoke/v12_se_severity.json"))
print()
c = out["clean"]
print("clean-rung comms (3-seed mean): SE=%.4f  exc=%.1f%%  poor=%.1f%%  outage=%.3f%%"
      %(np.mean([c[f"s{i}"]["se"] for i in range(3)]), np.mean([c[f"s{i}"]["excellent"] for i in range(3)]),
        np.mean([c[f"s{i}"]["poor"] for i in range(3)]), 100*np.mean([c[f"s{i}"]["outage"] for i in range(3)])))
print("(ddqn_tok3 reference:            SE=2.2733 exc=21.4%  poor=9.7%   outage=1.346%)")
print()
print("%-10s %10s %10s" % ("rung","sac_tok3","ddqn_tok3"))
for rung in ("bias_only","tilt1","tilt2"):
    sN = np.mean([out[rung][f"s{i}"]["dSE"] for i in range(3)])
    dN = np.mean([sev[rung][f"tok3_s1{i}"]["dSE"] for i in range(3)])
    print("%-10s %+10.4f %+10.4f" % (rung, sN, dN))
print("saved")
