"""Feedback-error ladder (eval-only, frozen policies, frozen 90 cases).
Physics stays nominal; only the OBSERVATIONS degrade -- additive SINR
measurement noise (pre-quantization) and coarser reporting quantization.
The degradation flows through the whole chain: estimator input and
per-sector report statistics alike.
Writes _smoke/v12_se_feedback_ladder.json"""
import copy
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/track_a_saint")
from pathlib import Path
from experiments import train_v1_1_controller_gate as gate
import train_track_a_clean_ppo as T

ATT, CAP, FLOOR, PMIN = 0.6, 4.4, -10.0, -100.0
D = "/home/yibo/vcode/data/rl_projects/offline_9sector_cache_rl"
SF = D + "/track_a_v12_se_final"
EST = D + "/v1_2_estimator_clean_20260818/model/best_v1_oracle_gt_m_estimator.pth"
base_args = T.build_parser().parse_args(["--mode","eval","--task","v1_2","--alpha-grid","0.25",
    "--demand-source","estimated","--state-map-size","256",
    "--cache-root","results/sac_9e6453c7_tilt_sweep_2e8_90x9",
    "--clean-estimator-checkpoint",EST,"--output-dir","/tmp/x","--policy","x=ppo=x",
    "--device","cuda:0","--eval-repeats","10"])
dev = torch.device("cuda:0")
cache, engine, est, _, _ = T.load_real_components(base_args, dev)
T.install_task(base_args, cache)
Z = torch.zeros((1, 18), device=dev)

def dse0(angles, dens):
    ang = torch.as_tensor(angles[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        rss, sinr = engine.compose(Z, ang)
    rss, sinr = rss[0], sinr[0]
    d = torch.as_tensor(dens, dtype=torch.float32, device=dev)
    v = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    d = d * v.float(); d = d / d.sum().clamp_min(1e-12)
    se = torch.clamp(ATT * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=CAP)
    se = torch.where(sinr < FLOOR, torch.zeros_like(se), se) * (rss >= PMIN).float()
    return float((d * se).sum())

# rung name -> (rss_quant, sinr_quant, sinr_noise)
RUNGS = [
    ("noise1",       1.0, 0.5, 1.0),
    ("noise2",       1.0, 0.5, 2.0),
    ("coarse",       2.0, 1.0, 0.0),
    ("coarse+noise", 2.0, 1.0, 1.0),
]

paths = {
    "naive":    [SF + "/seed2026081%d/bdqn_naive/final.pt" % s for s in (0, 1, 2)],
    "sac":      [SF + "/seed2026081%d/saint_sac/final.pt" % s for s in (0, 1, 2)],
    "proposed": [SF + "/seed2026081%d/saint_sac_tok3/final.pt" % s for s in (0, 1, 2)],
}
pol = {"const": ("const", None)}
for fam, ps in paths.items():
    for i, p in enumerate(ps):
        m, pay = T.load_track_a_policy(Path(p), dev)
        if hasattr(m, "attach_cache"):
            m.attach_cache(engine.base_dbm, cache.num_angles)
        pol[fam + "_s%d" % i] = ("bdqn", m)
print("loaded", len(pol), "policies", flush=True)

groups = [("seen", gate.parse_list(base_args.train_families)), ("ood", gate.parse_list(base_args.ood_families))]
out = {}
for rung, rq, sq, sn in RUNGS:
    args = copy.deepcopy(base_args)
    args.rss_quant_step_db = rq
    args.sinr_quant_step_db = sq
    args.sinr_noise_db = sn
    env = T.make_env(args, cache, engine, est, dev, int(args.eval_tape_seed))
    res = {k: [] for k in pol}
    ci = 0
    for gi, (g, fams) in enumerate(groups):
        for fi, f in enumerate(fams):
            for rep in range(10):
                cs = gate.derived_seed(int(args.eval_tape_seed), "eval_case", gi, fi, 0, rep)
                for name, (kind, m) in pol.items():
                    state = env.reset(ci, kind=f, alpha=0.25, case_seed=cs)
                    u0 = dse0(env.angles.copy(), env.density_true)
                    dens = env.density_true
                    done = False
                    while not done:
                        if kind == "const":
                            a = np.zeros(9, dtype=np.int64)
                        else:
                            a = T.greedy_action(m, kind, state, dev)
                        state, _, done = env.step(a)
                    res[name].append(dse0(env.angles.copy(), dens) - u0)
                ci += 1
                if ci % 30 == 0:
                    print(rung, ci, flush=True)
    out[rung] = {k: {"dSE": float(np.mean(v)), "per_case": [float(x) for x in v]} for k, v in res.items()}
json.dump(out, open(D + "/_smoke/v12_se_feedback_ladder.json", "w"), indent=2)

CLEAN = {"const": 0.1460, "naive": 0.3625, "sac": 0.3761, "proposed": 0.4636}
def m3(src, pre):
    ks = [k for k in src if k == pre or k.startswith(pre + "_s")]
    return float(np.mean([src[k]["dSE"] for k in ks]))
print()
print("%-10s %8s" % ("family", "clean") + "".join("%14s" % r[0] for r in RUNGS))
for fam in ("const", "naive", "sac", "proposed"):
    line = "%-10s %+8.4f" % (fam, CLEAN[fam])
    for rung, _, _, _ in RUNGS:
        line += "%+14.4f" % m3(out[rung], fam)
    print(line)
print()
for rung, _, _, _ in RUNGS:
    p = m3(out[rung], "proposed"); n = m3(out[rung], "naive"); s = m3(out[rung], "sac")
    print("%-13s proposed keeps %5.1f%% of clean gain; still > sac by %+.4f, > naive by %+.4f"
          % (rung, 100.0 * p / CLEAN["proposed"], p - s, p - n))
print("saved")
