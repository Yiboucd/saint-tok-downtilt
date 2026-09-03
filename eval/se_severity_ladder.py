"""SE-metric mismatch severity ladder for the SE-trained frozen policies (eval-only).

Rungs (same paired 90 cases, B1 arm = nominal state maps + mismatched physics):
  bias_only : power bias U(-3,3) dB per sector, no tilt offset
  tilt1     : tilt offset clamped to +-1 deg, plus power bias   (calibration-level)
  tilt2     : unclamped tilt offset (+-2) plus power bias       (stress)
Policies: const | SE-trained bdqn x3 | SE-trained tok3 x3.
Scores episodes with the coverage-gated truncated-Shannon dSE computed under the
EPISODE LATENT (the perturbed physics), true density. Clean rung comes from
_smoke/v12_se_final_eval.json (same case order). Writes _smoke/v12_se_severity.json
"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T

ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
RP, SF = f"{D}/track_a_v12_rewardprobe", f"{D}/track_a_v12_se_final"
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
MODE = {"kind": "bias_only"}
def patched(protocol, rng):
    lat = orig_latent(protocol, rng)
    if MODE["kind"] == "bias_only":
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

paths = {
    "bdqn_s10": f"{RP}/spectral_bdqn_s10/final.pt",
    "bdqn_s11": f"{SF}/seed20260811/bdqn/final.pt",
    "bdqn_s12": f"{SF}/seed20260812/bdqn/final.pt",
    "tok3_s10": f"{RP}/spectral_tok3_s10/final.pt",
    "tok3_s11": f"{SF}/seed20260811/saint_dqn_tok3/final.pt",
    "tok3_s12": f"{SF}/seed20260812/saint_dqn_tok3/final.pt",
}
policies = {"const": None}
for k, p in paths.items():
    m, _ = T.load_track_a_policy(Path(p), dev)
    if hasattr(m, "attach_cache"): m.attach_cache(engine.base_dbm, cache.num_angles)
    policies[k] = ("bdqn", m)

families_seen = gate.parse_list(args.train_families)
groups = [("seen", families_seen), ("ood", gate.parse_list(args.ood_families))]
out = {}
for mode in ("bias_only", "tilt1", "tilt2"):
    MODE["kind"] = "bias_only" if mode == "bias_only" else ("tilt1" if mode == "tilt1" else "raw")
    env = gate.StrictPrivilegedControllerEnv(cache, engine, est, dev, args, "B1", families_seen, int(args.eval_tape_seed))
    res = {p: {"dse": [], "group": []} for p in policies}
    ci = 0
    for gi, (g, fams) in enumerate(groups):
        for fi, f in enumerate(fams):
            for rep in range(10):
                cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
                for pname, spec in policies.items():
                    state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                    u0 = qual_dse(env.latent, env.angles.copy(), env.density_true)
                    dens = env.density_true; done = False
                    while not done:
                        if spec is None: a = np.zeros(9, dtype=np.int64)
                        else: a = T.greedy_action(spec[1], spec[0], state, dev)
                        state, _, done = env.step(a)
                    res[pname]["dse"].append(qual_dse(env.latent, env.angles.copy(), dens) - u0)
                    res[pname]["group"].append(g)
                ci += 1
                if ci % 30 == 0: print(mode, ci, "cases", flush=True)
    out[mode] = {}
    for p, v in res.items():
        a = np.array(v["dse"]); grp = np.array(v["group"])
        out[mode][p] = {"dSE": float(a.mean()), "seen": float(a[grp=="seen"].mean()),
                        "ood": float(a[grp=="ood"].mean()), "per_case": [float(x) for x in a]}
json.dump(out, open(f"{D}/_smoke/v12_se_severity.json", "w"), indent=2)

clean = json.load(open(f"{D}/_smoke/v12_se_final_eval.json"))
def fam3(o, pre): return float(np.mean([o[k]["dSE"] for k in o if k.startswith(pre)]))
print()
print("%-10s %8s %8s %8s %8s" % ("rung", "const", "bdqn", "tok3", "margin"))
print("%-10s %+8.4f %+8.4f %+8.4f %+8.4f" % ("clean", clean["const"]["dSE"], fam3(clean, "prev_bdqn_s"), fam3(clean, "proposed_s"), fam3(clean, "proposed_s") - clean["const"]["dSE"]))
for mode in ("bias_only", "tilt1", "tilt2"):
    c = out[mode]["const"]["dSE"]
    b = np.mean([out[mode][f"bdqn_s1{i}"]["dSE"] for i in range(3)])
    t = np.mean([out[mode][f"tok3_s1{i}"]["dSE"] for i in range(3)])
    print("%-10s %+8.4f %+8.4f %+8.4f %+8.4f" % (mode, c, b, t, t - c))
print("saved")
