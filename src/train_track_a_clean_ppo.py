#!/usr/bin/env python3
"""Track A: clean (no-mismatch) SAINT-PPO vs factorized PPO on the v1.1 C0 env.

Question answered: does explicit sector-token interaction (SAINT) improve
no-mismatch joint 9-sector tilt control over independent/factorized action
heads, at the same state contract, physical-step budget and PPO settings?

Environment: ``StrictPrivilegedControllerEnv`` from
``experiments/train_v1_1_controller_gate.py`` in condition ``C0`` (clean
physics, clean estimator, nominal maps, reset-only estimator, density frozen
for H=20, v1.1 feedback: 1-dB qRSRP + 0.5-dB qSINR, no noise).  Training
episodes use the ``clean_reference`` tape namespace, i.e. exactly the tape the
BDQN C0 gate arm trains on; evaluation cases use the gate's eval case-seed
derivation, so BDQN-C0 / factorized-PPO / SAINT-PPO are all paired.

Modes: ``selftest`` (CPU, synthetic cache), ``train``, ``eval``.
Nothing here touches production BDQN scripts or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from experiments import train_v1_1_controller_gate as gate  # noqa: E402
from experiments import train_v1_mismatch_bdqn as legacy  # noqa: E402
from experiments.v1_2 import protocol_v1_2 as v12  # noqa: E402
from saint_policy import (  # noqa: E402
    ALGORITHMS,
    build_policy,
    joint_log_prob_and_entropy,
    parameter_count,
    sample_actions,
)
from train_offline_9sector_cache_dqn import Offline9SectorCache, set_seed  # noqa: E402
from triple_tx_neural_estimator import load_estimator_checkpoint  # noqa: E402

TRACK_A_SCHEMA = "track_a_clean_ppo_training_complete/1"
TRACK_A_EVAL_SCHEMA = "track_a_clean_ppo_evaluation_complete/1"
CONDITION = "C0"


# ----------------------------------------------------------------- utilities
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def state_batch(states: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, ...]:
    maps = torch.as_tensor(np.stack([s["maps"] for s in states]), dtype=torch.float32, device=device)
    feedback = torch.as_tensor(
        np.stack([s["feedback"] for s in states]), dtype=torch.float32, device=device
    )
    angles = torch.as_tensor(
        np.stack([s["angles"] for s in states]) / 89.0, dtype=torch.float32, device=device
    )
    alpha = torch.as_tensor([[s["alpha"]] for s in states], dtype=torch.float32, device=device)
    progress = torch.as_tensor(
        [[s["progress"]] for s in states], dtype=torch.float32, device=device
    )
    return maps, feedback, angles, alpha, progress


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the policy consumes (drop the always-zero mismatch vector)."""
    return {
        "maps": np.asarray(state["maps"], dtype=np.float32),
        "feedback": np.asarray(state["feedback"], dtype=np.float32),
        "angles": np.asarray(state["angles"], dtype=np.float32),
        "alpha": float(state["alpha"]),
        "progress": float(state["progress"]),
    }


class UniformDemandCleanEnv(gate.StrictPrivilegedControllerEnv):
    """C0 physics/maps, demand channel fixed to the uniform density (estimator
    bypassed): the no-estimator RL baseline -- the policy can only learn demand
    implicitly from the per-step feedback Y_t."""

    def _initialize_frozen_episode_density(self) -> None:  # type: ignore[override]
        if self.step_no != 0 or self.episode_feedback_step_indices != [0]:
            raise RuntimeError("Demand initialization requires exactly reset feedback Y0")
        if self.episode_density_initializations != 0:
            raise RuntimeError("Demand channel was initialized more than once in one episode")
        self.episode_density_initializations = 1
        estimator_input = np.zeros((4, self.cache.h, self.cache.w), dtype=np.float32)
        density_est = np.full((self.cache.h, self.cache.w), 1.0 / (self.cache.h * self.cache.w), dtype=np.float32)
        self.estimator_input = estimator_input
        self.density_est = density_est
        self.estimator_input.setflags(write=False)
        self.density_est.setflags(write=False)
        self._frozen_estimator_input_reference = self.estimator_input
        self._frozen_density_reference = self.density_est

    def _assert_episode_inference_schedule(self) -> None:  # type: ignore[override]
        pass  # estimator is never invoked; the base assertions assume one inference


class TrueDemandCleanEnv(gate.StrictPrivilegedControllerEnv):
    """C0 physics/maps, but the frozen demand channel is the TRUE empirical density
    (estimator bypassed) -- the clean analogue of the O3 arm.  Used to measure the
    policy-reachable gap to the oracle independent of demand estimation."""

    @property
    def true_demand(self) -> bool:  # type: ignore[override]
        return True


def _install_mismatch_level(level: str) -> None:
    """Clamp gate.sample_joint_latent to the --train-mismatch level (declared domain randomization)."""
    base = getattr(gate, "_orig_sample_joint_latent", None)
    if base is None:
        base = gate.sample_joint_latent
        gate._orig_sample_joint_latent = base

    def sampler(protocol, rng, _base=base, _level=level):
        lat = _base(protocol, rng)
        if _level == "bias":
            lat[:, 9:] = 0.0
        elif _level == "tilt1":
            lat[:, 9:] = np.clip(lat[:, 9:], -1.0, 1.0)
        return lat  # tilt2: raw B1 latent (bias U(-3,3) + on-grid tilt {-2..2})

    gate.sample_joint_latent = sampler


def make_env(args: argparse.Namespace, cache, engine, estimator, device, tape_seed: int):
    src = str(getattr(args, "demand_source", "estimated"))
    cls = {"true": TrueDemandCleanEnv, "uniform": UniformDemandCleanEnv}.get(src, gate.StrictPrivilegedControllerEnv)
    arm = CONDITION
    level = str(getattr(args, "train_mismatch", "none"))
    if level != "none" and cls is gate.StrictPrivilegedControllerEnv:
        arm = "B1"  # nominal state maps, perturbed physics + reports; declared in config
        _install_mismatch_level(level)
    return cls(
        cache,
        engine,
        estimator,
        device,
        args,
        arm,
        gate.parse_list(args.train_families),
        int(tape_seed),
    )


def load_real_components(args: argparse.Namespace, device: torch.device):
    cache = Offline9SectorCache(
        Path(args.cache_root), float(args.bandwidth_hz), float(args.noise_figure_db)
    )
    engine = legacy.CoherentTwinEngine(cache, device)
    objective = str(getattr(args, "objective", "alpha_mix"))
    if objective == "coverage":
        install_coverage_objective(engine)
    elif objective == "coverage_graded":
        install_graded_objective(engine)
    elif objective == "risk":
        install_risk_objective(engine)
    elif objective == "spectral":
        install_spectral_objective(engine, proportional_fair=False)
    elif objective == "spectral_pf":
        install_spectral_objective(engine, proportional_fair=True)
    estimator, config, info = load_estimator_checkpoint(
        Path(args.clean_estimator_checkpoint), device
    )
    estimator.eval()
    for parameter in estimator.parameters():
        parameter.requires_grad_(False)
    return cache, engine, estimator, config, info


RSRP_GOOD, RSRP_FAIR = -90.0, -100.0
SINR_GOOD, SINR_FAIR = 10.0, 0.0
COV_SHARP = 2.0
# graded service score: cumulative excellent/good/fair bands (industry drive-test
# convention on RSRP/SINR quality), each band worth 1/3; ceiling = 100 %
GRADED_BANDS = ((-80.0, 20.0), (-90.0, 13.0), (-100.0, 0.0))


def install_coverage_objective(engine) -> None:
    """Replace engine.metrics with the industry good/fair coverage score (percent).
    Estimator, feedback, tapes are untouched; reward keeps its telescoping form."""
    import types

    def _soft_ge(x, thr):
        return torch.sigmoid((x - thr) / COV_SHARP)

    def coverage_metrics(self, latent, angles, density, alpha):
        rss, sinr = self.compose(latent, angles)
        if density.ndim == 2:
            density = density[None].expand(rss.shape[0], -1, -1)
        elif density.shape[0] == 1 and rss.shape[0] > 1:
            density = density.expand(rss.shape[0], -1, -1)
        d = density / torch.clamp(density.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        valid = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
        d = d * valid.float()
        d = d / torch.clamp(d.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        good = _soft_ge(rss, RSRP_GOOD) * _soft_ge(sinr, SINR_GOOD)
        fair = _soft_ge(rss, RSRP_FAIR) * _soft_ge(sinr, SINR_FAIR)
        cov = torch.sum((0.5 * good + 0.5 * fair) * d, dim=(1, 2)) * 100.0
        rss_mean = torch.sum(rss * d, dim=(1, 2)); sinr_mean = torch.sum(sinr * d, dim=(1, 2))
        return rss_mean, sinr_mean, cov

    engine.metrics = types.MethodType(coverage_metrics, engine)


def install_graded_objective(engine) -> None:
    """Replace engine.metrics with the graded 3GPP-style service score (percent).
    A demand pixel earns 1/3 per quality band attained (fair/good/excellent,
    RSRP AND SINR jointly); score = demand-weighted mean attainment, ceiling 100 %."""
    import types

    def _soft_ge(x, thr):
        return torch.sigmoid((x - thr) / COV_SHARP)

    def graded_metrics(self, latent, angles, density, alpha):
        rss, sinr = self.compose(latent, angles)
        if density.ndim == 2:
            density = density[None].expand(rss.shape[0], -1, -1)
        elif density.shape[0] == 1 and rss.shape[0] > 1:
            density = density.expand(rss.shape[0], -1, -1)
        d = density / torch.clamp(density.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        valid = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
        d = d * valid.float()
        d = d / torch.clamp(d.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        grade = torch.zeros_like(rss)
        for rsrp_thr, sinr_thr in GRADED_BANDS:
            grade = grade + _soft_ge(rss, rsrp_thr) * _soft_ge(sinr, sinr_thr) / len(GRADED_BANDS)
        cov = torch.sum(grade * d, dim=(1, 2)) * 100.0
        rss_mean = torch.sum(rss * d, dim=(1, 2)); sinr_mean = torch.sum(sinr * d, dim=(1, 2))
        return rss_mean, sinr_mean, cov

    engine.metrics = types.MethodType(graded_metrics, engine)


SE_ATTEN, SE_CAP, SE_FLOOR_DB = 0.6, 4.4, -10.0   # 3GPP TR 36.942 truncated Shannon
COV_GATE_DBM = -100.0                             # coverage gate: no service below this RSRP
SE_REWARD_SCALE = 6.0                             # matches the per-step TD reward scale the DQN
                                                  # recipe was tuned on (one-step reward sd 0.041
                                                  # bps/Hz); a positive affine rescale leaves the
                                                  # optimal policy unchanged
SE_PF_REWARD_SCALE = 16.0                         # log(1+SE) compresses the range ~2.8x; same target scale


def install_spectral_objective(engine, proportional_fair: bool = False) -> None:
    """Demand-weighted spectral efficiency (TR 36.942 truncated Shannon).

    No tunable constants: the attenuation factor, the cap and the low-SINR floor
    are the values specified in the TR.  ``proportional_fair`` replaces the mean
    by the demand-weighted mean of log(1+SE), the textbook utility that trades
    average against cell-edge rate.  Scaled by 100 for readable returns (a
    positive affine scaling leaves the optimal policy unchanged).

    Coverage enters as a gate rather than as a weighted term: demand whose
    best-server RSRP falls below COV_GATE_DBM contributes no service, mirroring
    the cell-selection minimum receive level.  The gate is a threshold, not a
    tunable weight, and is rarely binding in this scenario.  It is applied hard,
    not softened: DQN never differentiates the reward, and the audit measured a
    soft gate to change the utility by only 0.02%, so the trained objective is
    identical to the one reported.
    """
    import types

    def spectral_metrics(self, latent, angles, density, alpha):
        rss, sinr = self.compose(latent, angles)
        if density.ndim == 2:
            density = density[None].expand(rss.shape[0], -1, -1)
        elif density.shape[0] == 1 and rss.shape[0] > 1:
            density = density.expand(rss.shape[0], -1, -1)
        d = density / torch.clamp(density.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        valid = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
        d = d * valid.float()
        d = d / torch.clamp(d.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        se = torch.clamp(SE_ATTEN * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=SE_CAP)
        se = torch.where(sinr < SE_FLOOR_DB, torch.zeros_like(se), se)
        se = se * (rss >= COV_GATE_DBM).float()                      # coverage gate (hard)
        per_pixel = torch.log1p(se) if proportional_fair else se
        scale = SE_PF_REWARD_SCALE if proportional_fair else SE_REWARD_SCALE
        utility = torch.sum(per_pixel * d, dim=(1, 2)) * scale
        rss_mean = torch.sum(rss * d, dim=(1, 2)); sinr_mean = torch.sum(sinr * d, dim=(1, 2))
        return rss_mean, sinr_mean, utility

    engine.metrics = types.MethodType(spectral_metrics, engine)


RISK_RSRP_MIN, RISK_SINR_MIN = -100.0, 0.0


def install_risk_objective(engine) -> None:
    """Ericsson-style risk objective: demand fractions below service thresholds,
    aggregated nonlinearly.  Both risks are fractions in [0,1] -- no scale
    constants and no weight between them; the squares make the objective attack
    whichever risk is currently larger.

        R_cov  = E_rho 1[RSRP < RISK_RSRP_MIN]      (soft: sigmoid, COV_SHARP dB)
        R_qual = E_rho 1[SINR < RISK_SINR_MIN]
        U      = -log(1 + R_cov^2 + R_qual^2)       (reported x100 for readability)
    """
    import types

    def _soft_below(x, thr):
        return torch.sigmoid((thr - x) / COV_SHARP)

    def risk_metrics(self, latent, angles, density, alpha):
        rss, sinr = self.compose(latent, angles)
        if density.ndim == 2:
            density = density[None].expand(rss.shape[0], -1, -1)
        elif density.shape[0] == 1 and rss.shape[0] > 1:
            density = density.expand(rss.shape[0], -1, -1)
        d = density / torch.clamp(density.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        valid = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
        d = d * valid.float()
        d = d / torch.clamp(d.flatten(1).sum(dim=1), min=1e-12)[:, None, None]
        r_cov = torch.sum(_soft_below(rss, RISK_RSRP_MIN) * d, dim=(1, 2))
        r_qual = torch.sum(_soft_below(sinr, RISK_SINR_MIN) * d, dim=(1, 2))
        utility = -torch.log1p(r_cov ** 2 + r_qual ** 2) * 100.0
        rss_mean = torch.sum(rss * d, dim=(1, 2)); sinr_mean = torch.sum(sinr * d, dim=(1, 2))
        return rss_mean, sinr_mean, utility

    engine.metrics = types.MethodType(risk_metrics, engine)


def install_task(args: argparse.Namespace, cache) -> dict[str, Any]:
    """Apply the task layer to the loaded modules/cache. v1_1 = unchanged."""
    if str(getattr(args, "task", "v1_1")) == "v1_2":
        rec = v12.apply_v12(gate, legacy, cache)
        if str(args.alpha_grid) == "0,0.25,0.5,0.75,1":  # untouched default -> protocol main alpha
            args.alpha_grid = v12.ALPHA_GRID
        rec["alpha_grid"] = str(args.alpha_grid)
        return rec
    return {"protocol_version": "v1_1", "task_contract": "v1_1_tilt0to89_startU_actions_pm2_H20_alphagrid", "num_actions": 5}


def num_actions(args: argparse.Namespace) -> int:
    return v12.NUM_ACTIONS if str(getattr(args, "task", "v1_1")) == "v1_2" else 5


def protocol_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": gate.PROTOCOL_VERSION,
        "feedback_contract": gate.FEEDBACK_CONTRACT,
        "inference_schedule_contract": gate.INFERENCE_SCHEDULE_CONTRACT,
        "condition": ("B1" if str(getattr(args, "train_mismatch", "none")) != "none" else CONDITION),
        "train_mismatch": str(getattr(args, "train_mismatch", "none")),
        "task": str(getattr(args, "task", "v1_1")),
        "objective": str(getattr(args, "objective", "alpha_mix")),
        "sector_map_px": int(getattr(args, "sector_map_px", 64)),
        "coverage_thresholds": {"rsrp_good": RSRP_GOOD, "rsrp_fair": RSRP_FAIR, "sinr_good": SINR_GOOD, "sinr_fair": SINR_FAIR, "soft_db": COV_SHARP} if str(getattr(args, "objective", "alpha_mix")) == "coverage" else ({"graded_bands_rsrp_sinr": [list(b) for b in GRADED_BANDS], "band_weight": 1.0 / len(GRADED_BANDS), "soft_db": COV_SHARP} if str(getattr(args, "objective", "alpha_mix")) == "coverage_graded" else None),
        "task_contract": (v12.TASK_CONTRACT if str(getattr(args, "task", "v1_1")) == "v1_2" else "v1_1"),
        "demand_source": str(getattr(args, "demand_source", "estimated")),
        "myopic_baseline": bool(getattr(args, "myopic_baseline", False)),
        "estimator_checkpoint": str(args.clean_estimator_checkpoint),
        "rss_quant_step_db": float(args.rss_quant_step_db),
        "sinr_quant_step_db": float(args.sinr_quant_step_db),
        "sinr_noise_db": float(args.sinr_noise_db),
        "horizon": int(args.horizon),
        "gamma": float(args.gamma),
        "action_deltas_deg": gate.ACTION_DELTAS.tolist(),
        "num_ues": int(args.num_ues),
        "reporting_ues": int(args.reporting_ues),
        "state_map_size": int(args.state_map_size),
        "alpha_grid": str(args.alpha_grid),
        "train_families": gate.parse_list(args.train_families),
        "ood_families": gate.parse_list(args.ood_families),
    }


def ppo_record(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "algo", "seed", "tape_seed", "episodes", "ppo_lr", "ppo_rollout_episodes",
        "ppo_epochs", "ppo_batch_size", "ppo_clip", "gae_lambda", "entropy_coef",
        "value_coef", "max_grad_norm", "saint_token_dim", "saint_layers",
        "saint_heads", "saint_ffn_dim", "saint_per_sector_heads",
    ]
    return {key: getattr(args, key) for key in keys}


def build_model(args: argparse.Namespace) -> nn.Module:
    na = num_actions(args)
    if args.algo == "bdqn":
        return legacy.V1BranchingDQN(mismatch_input="none", num_actions=na)
    if args.algo in ("saint_dqn", "saint_dqn_tok", "saint_dqn_tok2", "saint_dqn_tok3"):
        from saint_policy import SAINTQNetwork, SAINTQNetworkTok, SAINTQNetworkTok2, SAINTQNetworkTok3
        cls = {"saint_dqn": SAINTQNetwork, "saint_dqn_tok": SAINTQNetworkTok, "saint_dqn_tok2": SAINTQNetworkTok2, "saint_dqn_tok3": SAINTQNetworkTok3}[args.algo]
        kw = dict(num_actions=na, token_dim=int(args.saint_token_dim), num_layers=int(args.saint_layers), num_heads=int(args.saint_heads), ffn_dim=int(args.saint_ffn_dim))
        if args.algo in ("saint_dqn_tok2", "saint_dqn_tok3"):
            kw["sector_map_px"] = int(getattr(args, "sector_map_px", 64))
        return cls(**kw)
    if args.algo in ("saint_sac", "saint_sac_tok3", "saint_qmix"):
        from sac_qmix_baselines import SACDiscrete, SAINTQMix
        kwb = dict(num_actions=na, token_dim=int(args.saint_token_dim), num_layers=int(args.saint_layers), num_heads=int(args.saint_heads), ffn_dim=int(args.saint_ffn_dim))
        if args.algo == "saint_qmix":
            return SAINTQMix(**kwb)
        return SACDiscrete(**kwb, token_variant=("tok3" if args.algo == "saint_sac_tok3" else "identity"), sector_map_px=int(getattr(args, "sector_map_px", 64)))
    if args.algo == "saint_ppo":
        return build_policy(
            "saint_ppo",
            num_actions=na,
            token_dim=int(args.saint_token_dim),
            num_layers=int(args.saint_layers),
            num_heads=int(args.saint_heads),
            ffn_dim=int(args.saint_ffn_dim),
            per_sector_heads=bool(args.saint_per_sector_heads),
        )
    return build_policy(str(args.algo), num_actions=na)


# ---------------------------------------------------------------- PPO pieces
def compute_gae(items: list[dict[str, Any]], gamma: float, lam: float) -> None:
    gae = 0.0
    next_value = 0.0
    for item in reversed(items):
        delta = float(item["reward"]) + gamma * next_value - float(item["value"])
        gae = delta + gamma * lam * gae
        item["advantage"] = float(gae)
        item["return_target"] = float(gae + item["value"])
        next_value = float(item["value"])


def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    rollout: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    adv = np.asarray([x["advantage"] for x in rollout], dtype=np.float32)
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
    for item, value in zip(rollout, adv):
        item["advantage_norm"] = float(value)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    updates = 0
    for _ in range(int(args.ppo_epochs)):
        order = rng.permutation(len(rollout))
        for start in range(0, len(order), int(args.ppo_batch_size)):
            batch = [rollout[int(i)] for i in order[start : start + int(args.ppo_batch_size)]]
            tensors = state_batch(batch, device)
            actions = torch.as_tensor(np.stack([x["action"] for x in batch]), dtype=torch.long, device=device)
            old_logp = torch.as_tensor([x["logp"] for x in batch], dtype=torch.float32, device=device)
            advantage = torch.as_tensor([x["advantage_norm"] for x in batch], dtype=torch.float32, device=device)
            returns = torch.as_tensor([x["return_target"] for x in batch], dtype=torch.float32, device=device)
            logits, values = model(*tensors)
            logp, entropy_sum = joint_log_prob_and_entropy(logits, actions)
            entropy = entropy_sum / float(logits.shape[1])  # mean per-sector entropy
            log_ratio = torch.clamp(logp - old_logp, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - float(args.ppo_clip), 1.0 + float(args.ppo_clip)) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy.mean())
                stats["approx_kl"] += float((torch.exp(log_ratio) - 1.0 - log_ratio).mean())
                stats["clip_frac"] += float(((ratio - 1.0).abs() > float(args.ppo_clip)).float().mean())
            updates += 1
    return {key: value / max(updates, 1) for key, value in stats.items()}


# ---------------------------------------------------------------- train mode
def train(args: argparse.Namespace, *, components=None) -> dict[str, Any]:
    gate.validate_locked_protocol(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fresh-only: refusing non-empty output dir {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    action_rng = np.random.default_rng(gate.derived_seed(int(args.seed), "track_a_ppo"))

    if components is None:
        cache, engine, estimator, est_cfg, est_info = load_real_components(args, device)
    else:
        cache, engine, estimator = components
        est_cfg, est_info = {}, {}
    task_record = install_task(args, cache)
    env = make_env(args, cache, engine, estimator, device, int(args.tape_seed))
    model = build_model(args).to(device)
    if hasattr(model, "attach_cache"):
        model.attach_cache(engine.base_dbm, cache.num_angles)
    if args.algo in ("bdqn", "bdqn_naive", "saint_dqn", "saint_dqn_tok", "saint_dqn_tok2", "saint_dqn_tok3", "saint_qmix"):
        return train_bdqn(args, env, model, device, output_dir, est_cfg, est_info)
    if args.algo in ("saint_sac", "saint_sac_tok3"):
        return train_sac(args, env, model, device, output_dir, est_cfg, est_info)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.ppo_lr), weight_decay=1e-5)
    initial_sha = gate.model_state_sha256(model)

    config = {
        "schema": "track_a_clean_ppo_config/1",
        "algo": str(args.algo),
        "protocol": protocol_record(args),
        "task": task_record,
        "ppo": ppo_record(args),
        "policy_parameter_count": parameter_count(model),
        "initial_policy_sha256": initial_sha,
        "training_tape_namespace": "clean_reference",
        "sources": {
            "runner": file_sha256(HERE),
            "saint_policy": file_sha256(HERE.parent / "saint_policy.py"),
            "gate_runner": file_sha256(Path(gate.__file__)),
        },
        "estimator_config": est_cfg,
        "estimator_train_info": est_info,
        "estimator_checkpoint": str(args.clean_estimator_checkpoint),
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
    }
    atomic_json(output_dir / "config.json", config)

    history_path = output_dir / "training_history.csv"
    rollout: list[dict[str, Any]] = []
    global_step = 0
    start = time.time()
    last_stats: dict[str, float] = {}
    for episode in range(1, int(args.episodes) + 1):
        state = env.reset(episode)
        items: list[dict[str, Any]] = []
        episode_return = 0.0
        done = False
        while not done:
            compact = compact_state(state)
            with torch.no_grad():
                logits, value = model(*state_batch([compact], device))
                action_t = sample_actions(logits[0][None])
                logp, _ = joint_log_prob_and_entropy(logits, action_t)
            action = action_t[0].cpu().numpy().astype(np.int64)
            next_state, reward, done = env.step(action)
            items.append({**compact, "action": action, "reward": float(reward), "value": float(value.item()), "logp": float(logp.item())})
            state = next_state
            episode_return += float(reward)
            global_step += 1
        compute_gae(items, float(args.gamma), float(args.gae_lambda))
        rollout.extend(items)
        if episode % int(args.ppo_rollout_episodes) == 0 or episode == int(args.episodes):
            last_stats = ppo_update(model, optimizer, rollout, args, device, action_rng)
            rollout.clear()
        row = env.episode_row(episode, episode_return, time.time() - start)
        row.update({"algo": str(args.algo), "global_step": int(global_step), **{f"ppo_{k}": v for k, v in last_stats.items()}})
        append_csv(history_path, row)
        if episode == 1 or episode % int(args.log_every) == 0:
            print(f"[track_a/{args.algo}] ep={episode}/{args.episodes} gain={row['composite_final_gain']:.4f} ret={episode_return:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in last_stats.items()), flush=True)
        if episode % int(args.checkpoint_every) == 0 or episode == int(args.episodes):
            payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "episode": int(episode), "global_step": int(global_step), "algo": str(args.algo), "config": config}
            torch.save(payload, output_dir / "latest.pt")
            if episode == int(args.episodes):
                torch.save(payload, output_dir / "final.pt")
    completion = {
        "schema": TRACK_A_SCHEMA,
        "status": "complete",
        "algo": str(args.algo),
        "episodes": int(args.episodes),
        "global_steps": int(global_step),
        "initial_policy_sha256": initial_sha,
        "final_policy_sha256": gate.model_state_sha256(model),
        "artifacts": {"final_checkpoint": gate.artifact(output_dir / "final.pt"), "training_history": gate.artifact(history_path), "config": gate.artifact(output_dir / "config.json")},
    }
    atomic_json(output_dir / "TRAINING_COMPLETE.json", completion)
    print(f"TRACK_A_TRAINING_COMPLETE {output_dir / 'TRAINING_COMPLETE.json'}", flush=True)
    return completion


def choose_action_variant(model, state, device, rng, epsilon, num_act, per_sector):
    if not per_sector:
        return legacy.choose_action(model, state, device, rng, epsilon)
    with torch.no_grad():
        q = model(*legacy.state_tensors(state, device))[0]
    action = torch.argmax(q, dim=1).cpu().numpy().astype(np.int64)
    mask = rng.random(action.shape[0]) < float(epsilon)
    action[mask] = rng.integers(0, num_act, size=int(mask.sum()), dtype=np.int64)
    return action


def _plan_coord_ascent(env, device, sweeps=4):
    """Coordinate ascent on the clean DT with the env's frozen demand channel."""
    engine = env.engine
    dens = torch.as_tensor(np.asarray(env.density_est, dtype=np.float32), device=device)
    zero = torch.zeros((1, 18), device=device)
    na = env.cache.num_angles
    theta = env.angles.copy().astype(np.float32)
    def value(batch):
        ang = torch.as_tensor(batch, dtype=torch.float32, device=device)
        lat = zero.expand(ang.shape[0], -1)
        with torch.no_grad():
            _, _, score = engine.metrics(lat, ang, dens, float(env.alpha))
        return score.cpu().numpy()
    best = float(value(theta[None])[0])
    for _ in range(int(sweeps)):
        improved = False
        for k in range(9):
            cands = np.arange(0, na)
            batch = np.repeat(theta[None], len(cands), axis=0); batch[:, k] = cands
            vals = value(batch); j = int(np.argmax(vals))
            if vals[j] > best + 1e-6:
                best = float(vals[j]); theta[k] = cands[j]; improved = True
        if not improved:
            break
    return theta


def bc_warmstart(args, env, online, device):
    """DT-guided warm start: behaviour-clone the walk-to-planner-target policy.
    Uses only deployable inputs (DT + the env's demand channel)."""
    deltas = legacy.ACTION_DELTAS
    demos = []
    for episode in range(1, int(args.bc_episodes) + 1):
        state = env.reset(episode)
        target = _plan_coord_ascent(env, device)
        done = False
        while not done:
            diff = np.rint(target - env.angles)
            action = np.abs(deltas[None, :] - np.clip(diff, deltas[0], deltas[-1])[:, None]).argmin(axis=1).astype(np.int64)
            demos.append((state, action))
            state, _, done = env.step(action)
        if episode % 200 == 0:
            print(f"[bc] demos from {episode}/{args.bc_episodes} episodes ({len(demos)} pairs)", flush=True)
    optimizer = optim.Adam(online.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    for epoch in range(int(args.bc_epochs)):
        order = rng.permutation(len(demos)); losses = []
        for start in range(0, len(order), 64):
            batch = [demos[i] for i in order[start:start + 64]]
            tensors = [torch.cat([legacy.state_tensors(st, device)[j] for st, _ in batch]) for j in range(6)]
            q = online(*tensors)
            actions = torch.as_tensor(np.stack([a for _, a in batch]), dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(q.reshape(-1, q.shape[-1]), actions.reshape(-1))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss))
        print(f"[bc] epoch {epoch + 1}/{args.bc_epochs} ce={np.mean(losses):.4f}", flush=True)


def train_bdqn(args, env, online, device, output_dir, est_cfg, est_info) -> dict[str, Any]:
    """Double branching DQN, same hyper-parameters as the v1.1 gate runner."""
    import copy
    target = copy.deepcopy(online).to(device)
    target.eval()
    initial_sha = gate.model_state_sha256(online)
    optimizer = optim.Adam(online.parameters(), lr=float(args.bdqn_lr))
    policy_rng = np.random.default_rng(gate.derived_seed(int(args.seed), "policy_replay"))
    replay = legacy.PackedReplay(int(args.replay_capacity), int(args.state_map_size), int(args.reporting_ues), map_channels=3)
    effective_warmup = max(int(args.bdqn_batch_size), min(int(args.warmup_steps), int(args.replay_capacity)))
    config = {
        "schema": "track_a_clean_bdqn_config/1", "algo": str(args.algo), "protocol": protocol_record(args),
        "bdqn": {k: getattr(args, k) for k in ["seed", "tape_seed", "episodes", "bdqn_lr", "bdqn_batch_size", "replay_capacity", "warmup_steps", "target_update_steps", "epsilon_start", "epsilon_end", "epsilon_decay_steps", "nstep", "per_sector_eps", "bc_episodes"]}, "ppo": {k: getattr(args, k) for k in ["saint_token_dim", "saint_layers", "saint_heads", "saint_ffn_dim"]},
        "policy_parameter_count": parameter_count(online), "initial_policy_sha256": initial_sha, "training_tape_namespace": "clean_reference",
        "sources": {"runner": file_sha256(HERE), "gate_runner": file_sha256(Path(gate.__file__)), "legacy": file_sha256(Path(legacy.__file__))},
        "estimator_config": est_cfg, "estimator_train_info": est_info, "estimator_checkpoint": str(args.clean_estimator_checkpoint),
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
    }
    atomic_json(output_dir / "config.json", config)
    history_path = output_dir / "training_history.csv"
    if int(getattr(args, "bc_episodes", 0)) > 0:
        bc_warmstart(args, env, online, device)
        target.load_state_dict(online.state_dict())
    nstep = max(1, int(getattr(args, "nstep", 1)))
    num_act = num_actions(args)
    global_step = 0; start = time.time()
    for episode in range(1, int(args.episodes) + 1):
        state = env.reset(episode); episode_return = 0.0; losses = []; done = False
        pending = []  # (state, action, cumulative reward) for n-step targets
        while not done:
            eps = legacy.epsilon_at(global_step, float(args.epsilon_start), float(args.epsilon_end), int(args.epsilon_decay_steps))
            action = choose_action_variant(online, state, device, policy_rng, eps, num_act, bool(getattr(args, "per_sector_eps", False)))
            next_state, reward, done = env.step(action)
            if nstep == 1:
                replay.push(state, action, reward, next_state, done)
            else:
                pending.append([state, action, 0.0])
                for item in pending:
                    item[2] += reward
                if len(pending) == nstep:
                    s0, a0, r0 = pending.pop(0)
                    replay.push(s0, a0, r0, next_state, done)
                if done:
                    for s0, a0, r0 in pending:
                        replay.push(s0, a0, r0, next_state, True)
                    pending = []
            state = next_state; episode_return += reward; global_step += 1
            if len(replay) >= effective_warmup:
                losses.append(optimize_qmix(online, target, optimizer, replay, int(args.bdqn_batch_size), float(args.gamma), policy_rng, device) if str(args.algo) == "saint_qmix" else legacy.optimize_ddqn(online, target, optimizer, replay, int(args.bdqn_batch_size), float(args.gamma), policy_rng, device))
            if global_step % int(args.target_update_steps) == 0:
                target.load_state_dict(online.state_dict())
        row = env.episode_row(episode, episode_return, time.time() - start)
        row.update({"algo": str(args.algo), "global_step": int(global_step), "epsilon": legacy.epsilon_at(global_step, float(args.epsilon_start), float(args.epsilon_end), int(args.epsilon_decay_steps)), "loss": float(np.mean([l["loss"] for l in losses])) if losses else float("nan")})
        append_csv(history_path, row)
        if episode == 1 or episode % int(args.log_every) == 0:
            print(f"[track_a/{args.algo}/{args.demand_source}] ep={episode}/{args.episodes} gain={row['composite_final_gain']:.4f} eps={row['epsilon']:.3f}", flush=True)
        if episode % int(args.checkpoint_every) == 0 or episode == int(args.episodes):
            payload = {"model_state": online.state_dict(), "target_state": target.state_dict(), "optimizer_state": optimizer.state_dict(), "episode": int(episode), "global_step": int(global_step), "algo": str(args.algo), "config": config}
            torch.save(payload, output_dir / "latest.pt")
            if episode == int(args.episodes):
                torch.save(payload, output_dir / "final.pt")
    completion = {"schema": TRACK_A_SCHEMA, "status": "complete", "algo": str(args.algo), "episodes": int(args.episodes), "global_steps": int(global_step), "initial_policy_sha256": initial_sha, "final_policy_sha256": gate.model_state_sha256(online),
                  "artifacts": {"final_checkpoint": gate.artifact(output_dir / "final.pt"), "training_history": gate.artifact(history_path), "config": gate.artifact(output_dir / "config.json")}}
    atomic_json(output_dir / "TRAINING_COMPLETE.json", completion)
    print(f"TRACK_A_TRAINING_COMPLETE {output_dir / 'TRAINING_COMPLETE.json'}", flush=True)
    return completion


def optimize_qmix(online, target, optimizer, replay, batch_size, gamma, rng, device):
    """QMIX TD step: per-sector double-DQN action selection, monotonic mixing of the
    selected Qs (online and target mixers), Huber loss on the mixed value."""
    import torch.nn.functional as F
    batch = replay.sample(batch_size, rng, device)
    S = (batch["maps"], batch["feedback"], batch["angles"], batch["alpha"], batch["progress"], batch["mismatch"])
    S2 = (batch["next_maps"], batch["next_feedback"], batch["next_angles"], batch["alpha"], batch["next_progress"], batch["next_mismatch"])
    q_sel = online(*S).gather(2, batch["actions"][:, :, None]).squeeze(-1)
    q_tot = online.mix(q_sel, batch["angles"], batch["alpha"], batch["progress"])
    with torch.no_grad():
        next_actions = torch.argmax(online(*S2), dim=2)
        qt_sel = target(*S2).gather(2, next_actions[:, :, None]).squeeze(-1)
        q_tot_next = target.mix(qt_sel, batch["next_angles"], batch["alpha"], batch["next_progress"])
        td_target = batch["reward"] + (1.0 - batch["done"]) * float(gamma) * q_tot_next
    loss = F.smooth_l1_loss(q_tot, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return {"loss": float(loss.item())}


def optimize_sac(model, q1_t, q2_t, opt_actor, opt_critic, opt_alpha, replay, batch_size, gamma, target_entropy, rng, device):
    import torch.nn.functional as F
    batch = replay.sample(batch_size, rng, device)
    S = (batch["maps"], batch["feedback"], batch["angles"], batch["alpha"], batch["progress"], batch["mismatch"])
    S2 = (batch["next_maps"], batch["next_feedback"], batch["next_angles"], batch["alpha"], batch["next_progress"], batch["next_mismatch"])
    alpha = model.log_alpha.exp().detach()
    with torch.no_grad():
        next_logp = F.log_softmax(model.actor(*S2), dim=2)
        next_p = next_logp.exp()
        qt = torch.minimum(q1_t(*S2), q2_t(*S2))
        v_next = (next_p * (qt - alpha * next_logp)).sum(dim=2).mean(dim=1)
        td_target = batch["reward"] + (1.0 - batch["done"]) * float(gamma) * v_next
    a = batch["actions"][:, :, None]
    q1_sel = model.q1(*S).gather(2, a).squeeze(-1).mean(dim=1)
    q2_sel = model.q2(*S).gather(2, a).squeeze(-1).mean(dim=1)
    critic_loss = F.smooth_l1_loss(q1_sel, td_target) + F.smooth_l1_loss(q2_sel, td_target)
    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(list(model.q1.parameters()) + list(model.q2.parameters()), max_norm=10.0)
    opt_critic.step()
    logp = F.log_softmax(model.actor(*S), dim=2)
    p = logp.exp()
    with torch.no_grad():
        qmin = torch.minimum(model.q1(*S), model.q2(*S))
    actor_loss = (p * (alpha * logp - qmin)).sum(dim=2).mean()
    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=10.0)
    opt_actor.step()
    with torch.no_grad():
        entropy = -(p * logp).sum(dim=2).mean()
    alpha_loss = model.log_alpha * (entropy.detach() - float(target_entropy))
    opt_alpha.zero_grad(set_to_none=True)
    alpha_loss.backward()
    opt_alpha.step()
    return {"loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
            "alpha": float(model.log_alpha.exp().item()), "entropy": float(entropy.item())}


def train_sac(args, env, model, device, output_dir, est_cfg, est_info) -> dict[str, Any]:
    """Discrete SAC baseline, standard settings: stochastic exploration (no epsilon),
    polyak target critics (tau=0.005), auto-tuned temperature toward a per-sector
    target entropy of 0.6*log(|A|). Everything else mirrors train_bdqn."""
    import copy
    import math as _math
    q1_t = copy.deepcopy(model.q1).to(device)
    q2_t = copy.deepcopy(model.q2).to(device)
    q1_t.eval(); q2_t.eval()
    initial_sha = gate.model_state_sha256(model)
    opt_actor = optim.Adam(model.actor.parameters(), lr=float(args.bdqn_lr))
    opt_critic = optim.Adam(list(model.q1.parameters()) + list(model.q2.parameters()), lr=float(args.bdqn_lr))
    opt_alpha = optim.Adam([model.log_alpha], lr=3e-4)
    tau = 0.005
    target_entropy = 0.6 * _math.log(float(num_actions(args)))
    policy_rng = np.random.default_rng(gate.derived_seed(int(args.seed), "policy_replay"))
    replay = legacy.PackedReplay(int(args.replay_capacity), int(args.state_map_size), int(args.reporting_ues), map_channels=3)
    effective_warmup = max(int(args.bdqn_batch_size), min(int(args.warmup_steps), int(args.replay_capacity)))
    config = {
        "schema": "track_a_clean_sac_config/1", "algo": str(args.algo), "protocol": protocol_record(args),
        "sac": {**{k: getattr(args, k) for k in ["seed", "tape_seed", "episodes", "bdqn_lr", "bdqn_batch_size", "replay_capacity", "warmup_steps"]},
                "alpha_lr": 3e-4, "tau": tau, "target_entropy": target_entropy, "alpha_init": 0.2,
                "exploration": "on-policy sampling from the softmax actor (no epsilon)"},
        "ppo": {k: getattr(args, k) for k in ["saint_token_dim", "saint_layers", "saint_heads", "saint_ffn_dim"]},
        "policy_parameter_count": parameter_count(model), "initial_policy_sha256": initial_sha, "training_tape_namespace": "clean_reference",
        "sources": {"runner": file_sha256(HERE), "gate_runner": file_sha256(Path(gate.__file__)), "legacy": file_sha256(Path(legacy.__file__))},
        "estimator_config": est_cfg, "estimator_train_info": est_info, "estimator_checkpoint": str(args.clean_estimator_checkpoint),
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
    }
    atomic_json(output_dir / "config.json", config)
    history_path = output_dir / "training_history.csv"
    global_step = 0
    start = time.time()
    for episode in range(1, int(args.episodes) + 1):
        state = env.reset(episode)
        episode_return = 0.0
        losses = []
        done = False
        while not done:
            with torch.no_grad():
                probs = torch.softmax(model.actor(*legacy.state_tensors(state, device))[0], dim=1).cpu().numpy()
            probs = probs / probs.sum(axis=1, keepdims=True)
            action = np.array([policy_rng.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
            next_state, reward, done = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            global_step += 1
            if len(replay) >= effective_warmup:
                losses.append(optimize_sac(model, q1_t, q2_t, opt_actor, opt_critic, opt_alpha, replay,
                                           int(args.bdqn_batch_size), float(args.gamma), target_entropy, policy_rng, device))
                with torch.no_grad():
                    for tp, p_ in zip(q1_t.parameters(), model.q1.parameters()):
                        tp.mul_(1.0 - tau).add_(p_, alpha=tau)
                    for tp, p_ in zip(q2_t.parameters(), model.q2.parameters()):
                        tp.mul_(1.0 - tau).add_(p_, alpha=tau)
        row = env.episode_row(episode, episode_return, time.time() - start)
        row.update({"algo": str(args.algo), "global_step": int(global_step), "epsilon": float("nan"),
                    "loss": float(np.mean([l["loss"] for l in losses])) if losses else float("nan"),
                    "sac_alpha": float(losses[-1]["alpha"]) if losses else float("nan"),
                    "sac_entropy": float(losses[-1]["entropy"]) if losses else float("nan")})
        append_csv(history_path, row)
        if episode == 1 or episode % int(args.log_every) == 0:
            print(f"[track_a/{args.algo}/{args.demand_source}] ep={episode}/{args.episodes} gain={row['composite_final_gain']:.4f} alpha={row['sac_alpha']:.3f} H={row['sac_entropy']:.3f}", flush=True)
        if episode % int(args.checkpoint_every) == 0 or episode == int(args.episodes):
            payload = {"model_state": model.state_dict(), "q1_target_state": q1_t.state_dict(), "q2_target_state": q2_t.state_dict(),
                       "opt_actor_state": opt_actor.state_dict(), "opt_critic_state": opt_critic.state_dict(), "opt_alpha_state": opt_alpha.state_dict(),
                       "episode": int(episode), "global_step": int(global_step), "algo": str(args.algo), "config": config}
            torch.save(payload, output_dir / "latest.pt")
            if episode == int(args.episodes):
                torch.save(payload, output_dir / "final.pt")
    completion = {"schema": TRACK_A_SCHEMA, "status": "complete", "algo": str(args.algo), "episodes": int(args.episodes), "global_steps": int(global_step), "initial_policy_sha256": initial_sha, "final_policy_sha256": gate.model_state_sha256(model),
                  "artifacts": {"final_checkpoint": gate.artifact(output_dir / "final.pt"), "training_history": gate.artifact(history_path), "config": gate.artifact(output_dir / "config.json")}}
    atomic_json(output_dir / "TRAINING_COMPLETE.json", completion)
    print(f"TRACK_A_TRAINING_COMPLETE {output_dir / 'TRAINING_COMPLETE.json'}", flush=True)
    return completion


# ----------------------------------------------------------------- eval mode
def load_track_a_policy(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if payload.get("algo") in ("bdqn", "bdqn_naive", "saint_dqn", "saint_dqn_tok", "saint_dqn_tok2", "saint_dqn_tok3", "saint_sac", "saint_sac_tok3", "saint_qmix"):
        na = v12.NUM_ACTIONS if cfg.get("protocol", {}).get("task", "v1_1") == "v1_2" else 5
        if payload.get("algo") == "bdqn":
            model = legacy.V1BranchingDQN(mismatch_input="none", num_actions=na)
        elif payload.get("algo") == "bdqn_naive":
            from saint_policy import NaiveBranchingDQN
            model = NaiveBranchingDQN(num_actions=na)
        elif payload.get("algo") in ("saint_sac", "saint_sac_tok3", "saint_qmix"):
            from sac_qmix_baselines import SACDiscrete, SAINTQMix
            sp = {k.replace("saint_", ""): v for k, v in cfg.get("ppo", {}).items() if k.startswith("saint_") and k != "saint_per_sector_heads"}
            kwb = dict(num_actions=na, token_dim=int(sp.get("token_dim", 64)), num_layers=int(sp.get("layers", 2)), num_heads=int(sp.get("heads", 4)), ffn_dim=int(sp.get("ffn_dim", 256)))
            if payload.get("algo") == "saint_qmix":
                model = SAINTQMix(**kwb)
            else:
                model = SACDiscrete(**kwb, token_variant=("tok3" if payload.get("algo") == "saint_sac_tok3" else "identity"), sector_map_px=int(cfg.get("protocol", {}).get("sector_map_px", 64) or 64))
        else:
            from saint_policy import SAINTQNetwork, SAINTQNetworkTok, SAINTQNetworkTok2, SAINTQNetworkTok3
            qcls = {"saint_dqn": SAINTQNetwork, "saint_dqn_tok": SAINTQNetworkTok, "saint_dqn_tok2": SAINTQNetworkTok2, "saint_dqn_tok3": SAINTQNetworkTok3}[payload.get("algo")]
            sp = {k.replace("saint_", ""): v for k, v in cfg.get("ppo", {}).items() if k.startswith("saint_") and k != "saint_per_sector_heads"}
            kw = dict(num_actions=na, token_dim=int(sp.get("token_dim", 64)), num_layers=int(sp.get("layers", 2)), num_heads=int(sp.get("heads", 4)), ffn_dim=int(sp.get("ffn_dim", 256)))
            if payload.get("algo") in ("saint_dqn_tok2", "saint_dqn_tok3"):
                kw["sector_map_px"] = int(cfg.get("protocol", {}).get("sector_map_px", 64) or 64)
            model = qcls(**kw)
        model.load_state_dict(payload["model_state"]); model.to(device).eval()
        return model, payload
    ns = argparse.Namespace(algo=payload["algo"], task=cfg.get("protocol", {}).get("task", "v1_1"), **{k: v for k, v in cfg["ppo"].items() if k.startswith("saint_")})
    model = build_model(ns)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def load_bdqn_policy(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = legacy.V1BranchingDQN(mismatch_input="none")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def greedy_action(policy: nn.Module, kind: str, state: dict[str, Any], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if kind == "bdqn":
            q = policy(*legacy.state_tensors(state, device))[0]
            return torch.argmax(q, dim=1).cpu().numpy().astype(np.int64)
        logits, _ = policy(*state_batch([compact_state(state)], device))
        return sample_actions(logits, deterministic=True)[0].cpu().numpy().astype(np.int64)


def evaluate(args: argparse.Namespace, *, components=None, policies=None) -> dict[str, Any]:
    gate.validate_locked_protocol(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if components is None:
        cache, engine, estimator, _cfg, _info = load_real_components(args, device)
    else:
        cache, engine, estimator = components
    install_task(args, cache)
    if policies is None:
        policies = {}
        for spec in args.policy:
            label, kind, path = spec.split("=", 2)
            if kind == "bdqn":
                model, _ = load_bdqn_policy(Path(path), device)
            elif kind in ("ppo", "track"):
                model, payload = load_track_a_policy(Path(path), device)
                if hasattr(model, "attach_cache"):
                    model.attach_cache(engine.base_dbm, cache.num_angles)
                if payload.get("algo") in ("bdqn", "bdqn_naive", "saint_dqn", "saint_dqn_tok", "saint_dqn_tok2", "saint_dqn_tok3", "saint_sac", "saint_sac_tok3", "saint_qmix"):
                    kind = "bdqn"  # greedy = argmax over per-sector Q
            else:
                raise ValueError(f"policy kind must be bdqn|ppo: {spec}")
            policies[label] = (kind, model, gate.artifact(Path(path)))
    labels = list(policies)
    envs = {label: make_env(args, cache, engine, estimator, device, int(args.eval_tape_seed)) for label in labels}
    groups = [("seen", gate.parse_list(args.train_families)), ("diagnostic_selection_exposed", gate.parse_list(args.ood_families))]
    alphas = legacy.parse_float_list(args.alpha_grid)
    rows: list[dict[str, Any]] = []
    case_index = 0
    start = time.time()
    for group_index, (group_name, families) in enumerate(groups):
        for family_index, family in enumerate(families):
            for alpha_index, alpha in enumerate(alphas):
                for repeat in range(int(args.eval_repeats)):
                    case_seed = gate.derived_seed(int(args.eval_tape_seed), "eval_case", group_index, family_index, alpha_index, repeat)
                    case_id = f"{group_name}:{family}:{alpha_index}:{repeat}"
                    tapes = []
                    for label in labels:
                        kind, model, art = policies[label]
                        env = envs[label]
                        state = env.reset(case_index, kind=family, alpha=float(alpha), case_seed=case_seed)
                        tapes.append(env.full_tape_sha256)
                        episode_return = 0.0
                        done = False
                        while not done:
                            state, reward, done = env.step(greedy_action(model, kind, state, device))
                            episode_return += float(reward)
                        row = env.episode_row(case_index + 1, episode_return, time.time() - start)
                        row.update({"policy": label, "policy_kind": kind, "policy_checkpoint_sha256": art["sha256"], "case_id": case_id, "case_seed": int(case_seed), "distribution_group": group_name, "repeat": int(repeat)})
                        rows.append(row)
                    if len(set(tapes)) != 1:
                        raise AssertionError(f"eval tape not paired across policies: {case_id}")
                    case_index += 1
    write_csv(output_dir / "evaluation_rows.csv", rows)
    summary = summarize(rows, labels, args)
    write_csv(output_dir / "evaluation_summary.csv", summary["summary"])
    write_csv(output_dir / "paired_contrasts.csv", summary["contrasts"])
    completion = {"schema": TRACK_A_EVAL_SCHEMA, "status": "complete", "num_cases": case_index, "policies": {label: {"kind": policies[label][0], "checkpoint": policies[label][2]} for label in labels}, "protocol": protocol_record(args), "eval_tape_seed": int(args.eval_tape_seed), "eval_repeats": int(args.eval_repeats), "artifacts": {"rows": gate.artifact(output_dir / "evaluation_rows.csv"), "summary": gate.artifact(output_dir / "evaluation_summary.csv"), "contrasts": gate.artifact(output_dir / "paired_contrasts.csv")}}
    atomic_json(output_dir / "EVALUATION_COMPLETE.json", completion)
    print(f"TRACK_A_EVALUATION_COMPLETE {output_dir / 'EVALUATION_COMPLETE.json'}", flush=True)
    return completion


def summarize(rows: list[dict[str, Any]], labels: list[str], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    metrics = ["initial_score", "final_score", "composite_final_gain", "rss_final_gain_db", "sinr_final_gain_db"]
    rng = np.random.default_rng(int(args.eval_tape_seed))
    by_case: dict[tuple[str, str], dict[str, Any]] = {(r["case_id"], r["policy"]): r for r in rows}
    case_ids = sorted({r["case_id"] for r in rows})
    summary_rows: list[dict[str, Any]] = []
    group_keys = sorted({(r["distribution_group"], None) for r in rows} | {(r["distribution_group"], float(r["alpha"])) for r in rows} | {("all", None)}, key=lambda x: (x[0], -1.0 if x[1] is None else x[1]))
    for label in labels:
        for group, alpha in group_keys:
            sel = [r for r in rows if r["policy"] == label and (group == "all" or r["distribution_group"] == group) and (alpha is None or math.isclose(float(r["alpha"]), alpha))]
            if not sel:
                continue
            out = {"policy": label, "distribution_group": group, "alpha": "" if alpha is None else alpha, "n": len(sel)}
            for m in metrics:
                vals = np.asarray([float(r[m]) for r in sel])
                out[f"{m}_mean"] = float(vals.mean())
                out[f"{m}_ci95"] = float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            summary_rows.append(out)
    contrast_rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for group, alpha in group_keys:
                pairs = []
                for cid in case_ids:
                    ra, rb = by_case.get((cid, a)), by_case.get((cid, b))
                    if ra is None or rb is None:
                        continue
                    if group != "all" and ra["distribution_group"] != group:
                        continue
                    if alpha is not None and not math.isclose(float(ra["alpha"]), alpha):
                        continue
                    pairs.append((ra, rb))
                if not pairs:
                    continue
                out = {"contrast": f"{b}-{a}", "distribution_group": group, "alpha": "" if alpha is None else alpha, "n": len(pairs), "bootstrap_resamples": int(args.bootstrap_resamples)}
                for m in ["final_score", "composite_final_gain"]:
                    d = np.asarray([float(rb[m]) - float(ra[m]) for ra, rb in pairs])
                    boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(int(args.bootstrap_resamples))]) if len(d) > 1 else np.asarray([d.mean()])
                    out[f"delta_{m}_mean"] = float(d.mean())
                    out[f"delta_{m}_win_rate"] = float((d > 1e-9).mean())
                    out[f"delta_{m}_tie_rate"] = float((np.abs(d) <= 1e-9).mean())
                    out[f"delta_{m}_ci95_low"] = float(np.quantile(boot, 0.025))
                    out[f"delta_{m}_ci95_high"] = float(np.quantile(boot, 0.975))
                contrast_rows.append(out)
    return {"summary": summary_rows, "contrasts": contrast_rows}


# --------------------------------------------------------------- selftest
def selftest(cli: argparse.Namespace) -> None:
    args = argparse.Namespace(**vars(cli))
    args.task = "v1_1"  # synthetic cache has 8 angles; the v1_2 task is smoke-tested on the real cache
    args.protocol = "joint_on_grid"
    args.num_ues = 32
    args.reporting_ues = 8
    args.horizon = 2
    args.allow_smoke_protocol = True
    args.alpha_grid = "0,0.5,1"
    args.state_map_size = 8
    args.episodes = 4
    args.ppo_rollout_episodes = 2
    args.ppo_batch_size = 4
    args.eval_repeats = 1
    args.bootstrap_resamples = 20
    args.checkpoint_every = 2
    args.log_every = 1
    args.device = "cpu"
    device = torch.device("cpu")
    cache = gate.SyntheticNineSectorCache()
    engine = legacy.CoherentTwinEngine(cache, device)
    estimator = gate._SelftestEstimator()

    # 1) module tests: shapes, log-prob, critic, gradients, permutation behaviour
    B, M = 3, 8
    maps = torch.rand(B, 3, 8, 8)
    feedback = torch.rand(B, M, 10)
    angles = torch.rand(B, 9)
    alpha = torch.rand(B, 1)
    progress = torch.rand(B, 1)
    counts = {}
    for name in ALGORITHMS:
        ns = argparse.Namespace(algo=name, saint_token_dim=16, saint_layers=1, saint_heads=2, saint_ffn_dim=32, saint_per_sector_heads=False)
        model = build_model(ns)
        logits, value = model(maps, feedback, angles, alpha, progress)
        assert tuple(logits.shape) == (B, 9, 5), (name, logits.shape)
        assert tuple(value.shape) == (B,), (name, value.shape)
        actions = sample_actions(logits)
        assert tuple(actions.shape) == (B, 9)
        logp, ent = joint_log_prob_and_entropy(logits, actions)
        manual = torch.log_softmax(logits, -1).gather(2, actions[..., None]).squeeze(-1).sum(1)
        assert torch.allclose(logp, manual, atol=1e-5), name
        assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
        (logp.mean() + value.mean()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        counts[name] = parameter_count(model)
    # SAINT: permuting the sector-embedding rows permutes the logits (equivariance)
    ns = argparse.Namespace(algo="saint_ppo", saint_token_dim=16, saint_layers=1, saint_heads=2, saint_ffn_dim=32, saint_per_sector_heads=False)
    saint = build_model(ns).eval()
    with torch.no_grad():
        base, _ = saint(maps, feedback, angles, alpha, progress)
        perm = torch.randperm(9)
        saint.sector_embedding.data = saint.sector_embedding.data[perm]
        permuted, _ = saint(maps, feedback, angles, alpha, progress)
    assert torch.allclose(permuted, base[:, perm], atol=1e-5), "SAINT tokens are not permutation-equivariant"

    # 2) two-episode training + eval on the synthetic C0 env for each algo
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policies = {}
        for name in ALGORITHMS:
            a = argparse.Namespace(**vars(args))
            a.algo = name
            a.saint_token_dim, a.saint_layers, a.saint_heads, a.saint_ffn_dim = 16, 1, 2, 32
            a.output_dir = str(root / name)
            done = train(a, components=(cache, engine, estimator))
            assert done["status"] == "complete" and done["global_steps"] == a.episodes * a.horizon
            model, _ = load_track_a_policy(root / name / "final.pt", device)
            policies[name] = ("ppo", model, gate.artifact(root / name / "final.pt"))
        bdqn = legacy.V1BranchingDQN(mismatch_input="none").eval()
        policies["bdqn_untrained"] = ("bdqn", bdqn, {"sha256": "0" * 64})
        a = argparse.Namespace(**vars(args))
        a.output_dir = str(root / "eval")
        result = evaluate(a, components=(cache, engine, estimator), policies=policies)
        expected_cases = (len(gate.TRAIN_FAMILIES) + len(gate.OOD_FAMILIES)) * 3 * a.eval_repeats
        assert result["num_cases"] == expected_cases, (result["num_cases"], expected_cases)
        rows = list(csv.DictReader(open(root / "eval" / "evaluation_rows.csv")))
        assert len(rows) == result["num_cases"] * len(policies)
    print(json.dumps({"selftest": "ok", "parameter_counts": counts, "feedback_contract": gate.FEEDBACK_CONTRACT}, sort_keys=True))


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["selftest", "train", "eval"], required=True)
    p.add_argument("--algo", choices=sorted(set(list(ALGORITHMS) + ["bdqn", "saint_sac", "saint_sac_tok3", "saint_qmix"])), default="saint_ppo")
    p.add_argument("--sector-map-px", type=int, default=64)
    p.add_argument("--objective", choices=["alpha_mix", "coverage", "coverage_graded", "risk", "spectral", "spectral_pf"], default="alpha_mix", help="coverage / coverage_graded = service-score objectives; risk = Ericsson-style -log(1+R_cov^2+R_qual^2) over demand fractions below service thresholds (alpha inert for all three)")
    p.add_argument("--task", choices=["v1_1", "v1_2"], default="v1_2", help="task layer: v1_2 = tilt 0-20, start U{0..20}, actions +-1, alpha 0.5")
    p.add_argument("--demand-source", choices=["estimated", "true", "uniform"], default="estimated", help="frozen demand channel: estimator output (default) or true empirical density (policy-reachable bound)")
    p.add_argument("--bdqn-lr", type=float, default=1e-4)
    p.add_argument("--bdqn-batch-size", type=int, default=32)
    p.add_argument("--replay-capacity", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--target-update-steps", type=int, default=2000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=120_000)
    p.add_argument("--nstep", type=int, default=1, help="n-step return for the DQN target (gamma=1: stored reward = sum of n rewards)")
    p.add_argument("--per-sector-eps", action="store_true", help="epsilon-greedy per sector instead of resampling the whole joint action")
    p.add_argument("--bc-episodes", type=int, default=0, help="behaviour-clone this many planner episodes before RL (DT-guided warm start)")
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--cache-root", default="")
    p.add_argument("--clean-estimator-checkpoint", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--policy", action="append", default=[], help="eval: LABEL=bdqn|ppo|track=PATH (repeatable; bdqn = gate checkpoint, track = any Track-A checkpoint)")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--tape-seed", type=int, default=20261001)
    p.add_argument("--eval-tape-seed", type=int, default=20261101)
    p.add_argument("--protocol", default="joint_on_grid", help="unused for C0; kept for env identity")
    p.add_argument("--train-mismatch", choices=["none", "bias", "tilt1", "tilt2"], default="none",
                   help="train under per-episode B1 mismatch (domain randomization): bias=power bias only, tilt1=bias+tilt clamped to +-1 deg, tilt2=full B1")
    p.add_argument("--bandwidth-hz", type=float, default=20e6)
    p.add_argument("--noise-figure-db", type=float, default=7.0)
    p.add_argument("--num-ues", type=int, default=2000)
    p.add_argument("--reporting-ues", type=int, default=200)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--myopic-baseline", action="store_true",
                   help="declared one-step/bandit baseline (Ericsson-style CMAB analogue): permits gamma != 1; "
                        "the deviation is recorded in the run config, every other protocol term stays locked")
    p.add_argument("--allow-smoke-protocol", action="store_true")
    p.add_argument("--rss-quant-step-db", type=float, default=gate.LOCKED_RSS_QUANT_STEP_DB)
    p.add_argument("--sinr-quant-step-db", type=float, default=gate.LOCKED_SINR_QUANT_STEP_DB)
    p.add_argument("--sinr-noise-db", type=float, default=gate.LOCKED_SINR_NOISE_DB)
    p.add_argument("--estimator-rss-tolerance-db", type=float, default=0.75)
    p.add_argument("--estimator-sinr-tolerance-db", type=float, default=1.0)
    p.add_argument("--state-map-size", type=int, default=64)
    p.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    p.add_argument("--train-families", default=",".join(gate.TRAIN_FAMILIES))
    p.add_argument("--ood-families", default=",".join(gate.OOD_FAMILIES))
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--ppo-lr", type=float, default=1e-4)
    p.add_argument("--ppo-rollout-episodes", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-batch-size", type=int, default=32)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--saint-token-dim", type=int, default=64)
    p.add_argument("--saint-layers", type=int, default=2)
    p.add_argument("--saint-heads", type=int, default=4)
    p.add_argument("--saint-ffn-dim", type=int, default=256)
    p.add_argument("--saint-per-sector-heads", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-repeats", type=int, default=10)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "selftest":
        selftest(args)
    elif args.mode == "train":
        if not (args.cache_root and args.clean_estimator_checkpoint and args.output_dir):
            raise SystemExit("train requires --cache-root --clean-estimator-checkpoint --output-dir")
        train(args)
    else:
        if not (args.cache_root and args.clean_estimator_checkpoint and args.output_dir and args.policy):
            raise SystemExit("eval requires --cache-root --clean-estimator-checkpoint --output-dir and >=1 --policy")
        evaluate(args)


if __name__ == "__main__":
    main()
