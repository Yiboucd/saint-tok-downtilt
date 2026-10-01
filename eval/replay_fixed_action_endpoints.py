#!/usr/bin/env python3
"""Read-only, CPU-only replay of the 90 nominal fixed-action endpoints.

No policy, estimator, or environment is instantiated. Outputs JSON to stdout;
does not write files. Run with PYTHONDONTWRITEBYTECODE=1 (stdin is supported).
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_pure_functions(path: Path, names: tuple[str, ...], namespace: dict) -> str:
    """Compile only named function definitions, not imports or module top level."""
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names) or len(selected) != len(names):
        raise RuntimeError(f"Expected exactly these pure functions in {path}: {names}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return sha256_bytes(data)


def load_maps(cache_root: Path) -> tuple[torch.Tensor, float, dict, list[dict]]:
    """Mirror Offline9SectorCache sorting; load only command angles 0..20."""
    metadata_path = cache_root / "batch_metadata.json"
    metadata_bytes = metadata_path.read_bytes() if metadata_path.exists() else b""
    metadata = json.loads(metadata_bytes.decode("utf-8")) if metadata_bytes else {}
    noise = metadata.get("noise", {})
    if isinstance(noise, dict):
        metadata.setdefault("bandwidth_hz", noise.get("bandwidth_hz", metadata.get("bandwidth_hz", 20e6)))
        metadata.setdefault("noise_figure_db", noise.get("noise_figure_db", metadata.get("noise_figure_db", 7.0)))
    bandwidth = float(metadata.get("bandwidth_hz", 20e6))
    noise_figure = float(metadata.get("noise_figure_db", 7.0))
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth) + noise_figure
    noise_watt = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    angle_dirs = sorted((cache_root / "rss_npys").glob("downtilt_*"))
    expected = [f"downtilt_{angle:03d}" for angle in range(21)]
    if [directory.name for directory in angle_dirs[:21]] != expected:
        raise RuntimeError("The first 21 sorted cache directories are not downtilt_000 through downtilt_020")
    cubes, manifest = [], []
    sector_order = None
    for angle, directory in enumerate(angle_dirs[:21]):
        files = sorted(directory.glob("*_rss_dbm.npy"))
        if len(files) != 9:
            raise RuntimeError(f"Expected 9 sector maps in {directory}")
        names = [path.name for path in files]
        if sector_order is None:
            sector_order = names
        elif names != sector_order:
            raise RuntimeError(f"Sector file order changes in {directory}")
        maps = []
        for sector, path in enumerate(files):
            raw = path.read_bytes()
            maps.append(np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32))
            manifest.append({"angle": angle, "sector": sector, "path": str(path), "sha256": sha256_bytes(raw)})
        cubes.append(np.stack(maps, axis=0))
    array = np.stack(cubes, axis=0).astype(np.float32)
    if array.shape != (21, 9, 256, 256):
        raise RuntimeError(f"Unexpected nominal cache shape: {array.shape}")
    info = {
        "root": str(cache_root), "loaded_shape": list(array.shape),
        "sector_order": sector_order, "metadata_sha256": sha256_bytes(metadata_bytes),
        "bandwidth_hz": bandwidth, "noise_figure_db": noise_figure,
        "noise_dbm": noise_dbm, "noise_watt": noise_watt,
        "manifest_sha256": sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
    }
    return torch.from_numpy(array), noise_watt, info, manifest


def compose(base_dbm: torch.Tensor, angles: np.ndarray, noise_watt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer, zero-mismatch specialization of CoherentTwinEngine.compose."""
    index = torch.as_tensor(angles, dtype=torch.long, device="cpu")
    sector = torch.arange(9, dtype=torch.long, device="cpu")
    power = torch.pow(10.0, (base_dbm[index, sector] - 30.0) / 10.0)
    association = torch.argmax(power, dim=0)
    serving = torch.gather(power, 0, association[None]).squeeze(0)
    interference = torch.clamp(power.sum(dim=0) - serving + noise_watt, min=1e-30)
    rss = 10.0 * torch.log10(torch.clamp(serving, min=1e-30)) + 30.0
    sinr = 10.0 * torch.log10(torch.clamp(serving / interference, min=1e-30))
    return rss, sinr


def utility(fields: tuple[torch.Tensor, torch.Tensor], density: np.ndarray) -> float:
    """Identical coverage-gated, unscaled SE expression to se_decomp_eval.py."""
    rss, sinr = fields
    weight = torch.as_tensor(density, dtype=torch.float32, device="cpu")
    valid = (rss > -250.0) & torch.isfinite(rss) & torch.isfinite(sinr)
    weight = weight * valid.float()
    weight = weight / weight.sum().clamp_min(1e-12)
    spectral = torch.clamp(0.6 * torch.log2(1.0 + torch.pow(10.0, sinr / 10.0)), max=4.4)
    spectral = torch.where(sinr < -10.0, torch.zeros_like(spectral), spectral) * (rss >= -100.0).float()
    return float((weight * spectral).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/yibo/vcode/radiogeneration"))
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    cache_root = args.cache_root or source_root / "results/sac_9e6453c7_tilt_sweep_2e8_90x9"
    namespace = {"np": np, "math": math, "hashlib": hashlib, "Any": Any}
    hashes = {}
    for relative, names in (
        ("train_offline_9sector_cache_dqn.py", ("normalize_density", "make_density", "sample_locations")),
        ("experiments/train_v1_1_controller_gate.py", ("derived_seed",)),
        ("experiments/train_v1_mismatch_bdqn.py", ("empirical_density",)),
    ):
        path = source_root / relative
        hashes[str(path)] = load_pure_functions(path, names, namespace)
    for relative in (
        "train_uncertain_twin_cf_dqn.py",
        "experiments/v1_2/protocol_v1_2.py",
        "experiments/track_a_saint/train_track_a_clean_ppo.py",
    ):
        path = source_root / relative
        hashes[str(path)] = sha256_bytes(path.read_bytes())
    base_dbm, noise_watt, cache_info, map_manifest = load_maps(cache_root.resolve())
    groups = [
        ("seen", ["uniform", "normal", "mixture", "line_x", "edge"]),
        ("held_out", ["ring", "corner", "line_y", "hotspot_two_clusters"]),
    ]
    derive = namespace["derived_seed"]
    make_density = namespace["make_density"]
    sample_locations = namespace["sample_locations"]
    empirical_density = namespace["empirical_density"]
    records = []
    with torch.inference_mode():
        endpoints = {angle: compose(base_dbm, np.full(9, angle), noise_watt) for angle in (0, 20)}
        for group_index, (group, families) in enumerate(groups):
            for family_index, family in enumerate(families):
                for repeat in range(10):
                    seed = derive(20261101, "eval_case", group_index, family_index, 0, repeat)
                    density_rng = np.random.default_rng(derive(seed, "density_population"))
                    angle_rng = np.random.default_rng(derive(seed, "angles"))
                    population = sample_locations(make_density(family, 256, 256, density_rng), 2000, density_rng)
                    density = empirical_density(population, 256, 256)
                    angles = angle_rng.integers(0, 21, size=9).astype(np.float32)
                    assert np.all(np.clip(angles - 20, 0, 20) == 0)
                    assert np.all(np.clip(angles + 20, 0, 20) == 20)
                    initial = utility(compose(base_dbm, angles, noise_watt), density)
                    final0 = utility(endpoints[0], density)
                    final20 = utility(endpoints[20], density)
                    records.append({
                        "case_index": len(records), "group": group, "family": family,
                        "repeat": repeat, "case_seed": seed, "initial_angles": angles.tolist(),
                        "population_sha256": sha256_bytes(population.tobytes(order="C")),
                        "density_sha256": sha256_bytes(density.tobytes(order="C")),
                        "initial_utility": initial, "uniform0_utility": final0, "uniform20_utility": final20,
                        "minus1_gain": final0 - initial, "plus1_gain": final20 - initial,
                    })
    if len(records) != 90:
        raise RuntimeError("Expected exactly 90 paired cases")
    summary = {}
    for key in ("minus1_gain", "plus1_gain"):
        summary[key] = {
            split: float(np.mean([case[key] for case in records if split == "overall" or case["group"] == split]))
            for split in ("overall", "seen", "held_out")
        }
    result = {
        "schema": "fixed_action_endpoint_cpu_replay/1",
        "protocol": {
            "eval_tape_seed": 20261101, "case_seed_labels": ["eval_case", "group_index", "family_index", 0, "repeat"],
            "groups": groups, "repeats_per_family": 10, "num_ues": 2000,
            "horizon": 20, "commanded_tilt_range": [0, 20], "action_steps": [-1, 1],
            "nominal_physics": True, "reward_scale_applied": False,
            "metric": {"attenuation": 0.6, "cap_bps_hz": 4.4, "sinr_cutoff_db": -10.0, "rsrp_cutoff_dbm": -100.0},
            "fixed_minus1_endpoint": 0, "fixed_plus1_endpoint": 20,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "numpy_version": np.__version__, "torch_version": torch.__version__,
            "notes": "No reporters, estimator, policies, environment, training, or GPU are needed. CPU float32 reductions may differ slightly from saved GPU metrics.",
        },
        "source_sha256": hashes, "cache": cache_info, "map_manifest": map_manifest,
        "summary": summary,
        "per_case": {key: [case[key] for case in records] for key in ("minus1_gain", "plus1_gain")},
        "cases": records,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
