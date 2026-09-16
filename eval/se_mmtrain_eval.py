"""4x4 matrix: mismatch-TRAINED tok3 probes evaluated on all four rungs.

Rows (training condition): mm_bias / mm_tilt1 / mm_tilt2  (clean-trained row
already exists: v12_se_final_eval.json proposed_s* + v12_se_severity.json).
Cols (eval rung): clean / bias_only / tilt1 / tilt2 - same 90 frozen paired cases.
Writes _smoke/v12_se_mmtrain_eval.json
"""
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
    "--device","cuda:0","--eval-repeats","10","--protocol","joint_on_grid"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(args, dev)
T.install_task(args, cache)

orig_latent = gate.sample_joint_latent
MODE = {"kind": "clean"}
def patched(protocol, rng):
    lat = orig_latent(protocol, rng)
    if MODE["kind"] == "clean":
        lat[:, :] = 0.0
    elif MODE["kind"] == "bias_only":
        lat[:, 9:] = 0.0
    elif MODE["kind"] == "tilt1":
        lat[:, 9:] = np.clip(lat[:, 9:], -1.0, 1.0)
    return lat
gate.sample_joint_latent = patched

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

policies = {}
for lvl in ("bias", "tilt1", "tilt2"):
    p = Path(f"{D}/track_a_v12_se_mmtrain/mm_{lvl}_s10/final.pt")
    m, _ = T.load_track_a_policy(p, dev)
    if hasattr(m, "attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    policies[f"mm_{lvl}"] = m
print("loaded", list(policies), flush=True)

families_seen = gate.parse_list(args.train_families)
groups = [("seen", families_seen), ("ood", gate.parse_list(args.ood_families))]
out = {}
for rung in ("clean", "bias_only", "tilt1", "tilt2"):
    MODE["kind"] = "clean" if rung == "clean" else ("bias_only" if rung == "bias_only" else ("tilt1" if rung == "tilt1" else "raw"))
    env = gate.StrictPrivilegedControllerEnv(cache, engine, est, dev, args, "B1", families_seen, int(args.eval_tape_seed))
    res = {p: {"dse": [], "group": []} for p in policies}
    ci = 0
    for gi, (g, fams) in enumerate(groups):
        for fi, f in enumerate(fams):
            for rep in range(10):
                cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
                for pname, m in policies.items():
                    state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                    u0 = qual_dse(env.latent, env.angles.copy(), env.density_true)
                    dens = env.density_true; done = False
                    while not done:
                        a = T.greedy_action(m, "bdqn", state, dev)
                        state, _, done = env.step(a)
                    res[pname]["dse"].append(qual_dse(env.latent, env.angles.copy(), dens) - u0)
                    res[pname]["group"].append(g)
                ci += 1
                if ci % 30 == 0: print(rung, ci, "cases", flush=True)
    out[rung] = {}
    for p, v in res.items():
        a = np.array(v["dse"]); grp = np.array(v["group"])
        out[rung][p] = {"dSE": float(a.mean()), "seen": float(a[grp=="seen"].mean()),
                        "ood": float(a[grp=="ood"].mean()), "per_case": [float(x) for x in a]}
json.dump(out, open(f"{D}/_smoke/v12_se_mmtrain_eval.json", "w"), indent=2)

sev = json.load(open(f"{D}/_smoke/v12_se_severity.json"))
fin = json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
clean_trained = {"clean": fin["proposed_s0"]["dSE"],
                 "bias_only": sev["bias_only"]["tok3_s10"]["dSE"],
                 "tilt1": sev["tilt1"]["tok3_s10"]["dSE"],
                 "tilt2": sev["tilt2"]["tok3_s10"]["dSE"]}
print()
print("=== 4x4 matrix, dSE (rows=training condition, cols=eval rung; all seed 20260810) ===")
print("%-14s %8s %10s %8s %8s" % ("trained-on", "clean", "bias_only", "tilt1", "tilt2"))
print("%-14s %+8.4f %+10.4f %+8.4f %+8.4f" % ("clean", clean_trained["clean"], clean_trained["bias_only"], clean_trained["tilt1"], clean_trained["tilt2"]))
for lvl in ("bias", "tilt1", "tilt2"):
    k = f"mm_{lvl}"
    print("%-14s %+8.4f %+10.4f %+8.4f %+8.4f" % (k, out["clean"][k]["dSE"], out["bias_only"][k]["dSE"], out["tilt1"][k]["dSE"], out["tilt2"][k]["dSE"]))
print("saved")
