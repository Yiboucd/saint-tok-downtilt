"""Severity ladder (bias_only/tilt1/tilt2) for the families not yet covered:
random, ppo x3, myopic, saint_id x3, qmix x3, sac x3, naive x3.
Merges with the existing const/bdqn/tok3 and sac_tok3 ladders into one matrix.
Writes _smoke/v12_se_ladder_all.json"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T
ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
SF = f"{D}/track_a_v12_se_final"
EST = f"{D}/v1_2_estimator_clean_20260818/model/best_v1_oracle_gt_m_estimator.pth"
args = T.build_parser().parse_args(["--mode","eval","--task","v1_2","--alpha-grid","0.25",
    "--demand-source","estimated","--state-map-size","256",
    "--cache-root","results/sac_9e6453c7_tilt_sweep_2e8_90x9",
    "--clean-estimator-checkpoint",EST,"--output-dir","/tmp/x","--policy","x=ppo=x",
    "--device","cuda:0","--eval-repeats","10","--protocol","joint_on_grid"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(args, dev)
T.install_task(args, cache)
def qual_dse(latent, angles, dens):
    lat = torch.as_tensor(latent, dtype=torch.float32, device=dev)
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad(): rss, sinr = engine.compose(lat, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d*v.float(); d = d/d.sum().clamp_min(1e-12)
    se = torch.clamp(ATT*torch.log2(1.0+torch.pow(10.0, sinr/10.0)), max=CAP)
    se = torch.where(sinr<FLOOR, torch.zeros_like(se), se)*(rss>=PMIN).float()
    return float((d*se).sum())
orig = gate.sample_joint_latent
MODE = {"kind": "bias_only"}
def patched(protocol, rng):
    lat = orig(protocol, rng)
    if MODE["kind"] == "bias_only": lat[:, 9:] = 0.0
    elif MODE["kind"] == "tilt1": lat[:, 9:] = np.clip(lat[:, 9:], -1.0, 1.0)
    return lat
gate.sample_joint_latent = patched
paths = {}
for i,s in enumerate((0,1,2)):
    paths[f"ppo_s{i}"] = (f"{SF}/seed2026081{s}/saint_ppo/final.pt")
    paths[f"id_s{i}"] = (f"{SF}/seed2026081{s}/saint_dqn/final.pt")
    paths[f"qmix_s{i}"] = (f"{SF}/seed2026081{s}/saint_qmix/final.pt")
    paths[f"sac_s{i}"] = (f"{SF}/seed2026081{s}/saint_sac/final.pt")
    paths[f"naive_s{i}"] = (f"{SF}/seed2026081{s}/bdqn_naive/final.pt")
paths["myopic_s0"] = f"{SF}/seed20260810/myopic_tok3/final.pt"
policies = {"random": ("random", None)}
for k, p in paths.items():
    m, payload = T.load_track_a_policy(Path(p), dev)
    if hasattr(m, "attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    kind = "ppo" if payload.get("algo") == "saint_ppo" else "bdqn"
    policies[k] = (kind, m)
print("loaded", len(policies), "policies", flush=True)
families = gate.parse_list(args.train_families)
groups = [("seen", families), ("ood", gate.parse_list(args.ood_families))]
out = {}
for rung in ("bias_only", "tilt1", "tilt2"):
    MODE["kind"] = "bias_only" if rung == "bias_only" else ("tilt1" if rung == "tilt1" else "raw")
    env = gate.StrictPrivilegedControllerEnv(cache, engine, est, dev, args, "B1", families, int(args.eval_tape_seed))
    res = {p: [] for p in policies}
    ci = 0
    for gi,(g,fams) in enumerate(groups):
        for fi,f in enumerate(fams):
            for rep in range(10):
                cs = gate.derived_seed(int(args.eval_tape_seed),"eval_case",gi,fi,0,rep)
                for name,(kind,m) in policies.items():
                    state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                    u0 = qual_dse(env.latent, env.angles.copy(), env.density_true)
                    dens = env.density_true
                    rng = np.random.default_rng(gate.derived_seed(cs, "random_policy"))
                    done = False
                    while not done:
                        if kind == "random": a = rng.integers(0, 3, 9).astype(np.int64)
                        else: a = T.greedy_action(m, kind, state, dev)
                        state, _, done = env.step(a)
                    res[name].append(qual_dse(env.latent, env.angles.copy(), dens) - u0)
                ci += 1
                if ci % 30 == 0: print(rung, ci, flush=True)
    out[rung] = {p: {"dSE": float(np.mean(v)), "per_case": [float(x) for x in v]} for p, v in res.items()}
json.dump(out, open(f"{D}/_smoke/v12_se_ladder_all.json", "w"), indent=2)
sev = json.load(open(f"{D}/_smoke/v12_se_severity.json"))
stf = json.load(open(f"{D}/_smoke/v12_se_sactok3_full.json"))
clean = {"random": -0.0037, "const": 0.1460, "ppo": 0.2620, "myopic": 0.3123, "id": 0.3358,
         "qmix": 0.3423, "sac": 0.3761, "naive": 0.3625, "bdqn_t": 0.4004, "tok3_ddqn": 0.4334, "tok3_sac": 0.4636}
def m3(src, pre): return float(np.mean([src[k]["dSE"] for k in src if k.startswith(pre)]))
rows = [("Random", clean["random"], m3(out["bias_only"],"random"), m3(out["tilt1"],"random"), m3(out["tilt2"],"random")),
        ("Constant", clean["const"], sev["bias_only"]["const"]["dSE"], sev["tilt1"]["const"]["dSE"], sev["tilt2"]["const"]["dSE"]),
        ("PPO", clean["ppo"], m3(out["bias_only"],"ppo_"), m3(out["tilt1"],"ppo_"), m3(out["tilt2"],"ppo_")),
        ("One-step", clean["myopic"], m3(out["bias_only"],"myopic"), m3(out["tilt1"],"myopic"), m3(out["tilt2"],"myopic")),
        ("SAINT-id", clean["id"], m3(out["bias_only"],"id_"), m3(out["tilt1"],"id_"), m3(out["tilt2"],"id_")),
        ("QMIX", clean["qmix"], m3(out["bias_only"],"qmix_"), m3(out["tilt1"],"qmix_"), m3(out["tilt2"],"qmix_")),
        ("SAC", clean["sac"], m3(out["bias_only"],"sac_"), m3(out["tilt1"],"sac_"), m3(out["tilt2"],"sac_")),
        ("BDQN-naive", clean["naive"], m3(out["bias_only"],"naive_"), m3(out["tilt1"],"naive_"), m3(out["tilt2"],"naive_")),
        ("BDQN-tailored", clean["bdqn_t"], m3(sev["bias_only"],"bdqn_s"), m3(sev["tilt1"],"bdqn_s"), m3(sev["tilt2"],"bdqn_s")),
        ("Tok3+DDQN", clean["tok3_ddqn"], m3(sev["bias_only"],"tok3_s"), m3(sev["tilt1"],"tok3_s"), m3(sev["tilt2"],"tok3_s")),
        ("Tok3+SAC", clean["tok3_sac"], m3(stf["bias_only"],"s"), m3(stf["tilt1"],"s"), m3(stf["tilt2"],"s"))]
print()
print("%-14s %8s %10s %8s %8s" % ("policy","clean","bias+-3dB","tilt+-1","tilt+-2"))
for lbl,c,b,t1,t2 in rows:
    print("%-14s %+8.4f %+10.4f %+8.4f %+8.4f" % (lbl,c,b,t1,t2))
print("saved")
