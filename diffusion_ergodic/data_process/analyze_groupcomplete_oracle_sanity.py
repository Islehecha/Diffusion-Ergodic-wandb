#!/usr/bin/env python3
"""Small oracle sanity check for gamma-conditioned TOES trajectories."""

import collections
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np


DATA_DIR = Path("/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic/data/ergodic_toes_continuation_v1_groupcomplete_balanced")
OUT_PATH = DATA_DIR / "oracle_gamma_sanity_summary.json"
GAMMAS = [0.01, 0.03, 0.1, 0.2]
HOTSPOT_COV_SCALE = 0.20


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def mean_point_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def path_length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def pack_distribution(dist_data, max_gaussians=20):
    n = int(dist_data["params"]["n_gaussians"])
    centers = np.asarray(dist_data["params"]["centers"], dtype=np.float64)
    weights = np.asarray(dist_data["params"]["weights"], dtype=np.float64)
    covs_raw = dist_data["params"]["covs"]
    packed = np.zeros((max_gaussians, 7), dtype=np.float64)
    valid = np.zeros((max_gaussians,), dtype=bool)
    for i in range(min(n, max_gaussians)):
        packed[i, 0:2] = centers[i]
        c = covs_raw[i]
        if np.isscalar(c):
            cov_mat = np.array([c, 0, 0, c], dtype=np.float64)
        elif len(np.shape(c)) == 1:
            cov_mat = np.array([c[0], 0, 0, c[1]], dtype=np.float64)
        else:
            cov_mat = np.asarray(c, dtype=np.float64).flatten()
        if cov_mat.size == 4:
            packed[i, 2:6] = cov_mat
        packed[i, 6] = weights[i]
        valid[i] = True
    return packed, valid


def hotspot_coverage_score(xy, packed, valid, temperature=12.0):
    centers = packed[:, 0:2]
    cov_flat = packed[:, 2:6]
    weights = packed[:, 6]
    sx2 = np.maximum(np.abs(cov_flat[:, 0]) * HOTSPOT_COV_SCALE, 1e-4)
    sy2 = np.maximum(np.abs(cov_flat[:, 3]) * HOTSPOT_COV_SCALE, 1e-4)
    diff = xy[:, None, :] - centers[None, :, :]
    exponent = -0.5 * ((diff[..., 0] ** 2) / sx2[None, :] + (diff[..., 1] ** 2) / sy2[None, :])
    proximity = np.exp(np.clip(exponent, -50.0, 0.0))
    proximity[:, ~valid] = 0.0
    logits = temperature * proximity
    logits = logits - logits.max(axis=0, keepdims=True)
    soft_select = np.exp(logits)
    soft_select = soft_select / np.maximum(soft_select.sum(axis=0, keepdims=True), 1e-12)
    soft_max = (soft_select * proximity).sum(axis=0)
    norm_weights = weights * valid
    norm_weights = norm_weights / max(norm_weights.sum(), 1e-12)
    return float((soft_max * norm_weights).sum())


def summarize(xs):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "std": float(xs.std()),
        "min": float(xs.min()),
        "max": float(xs.max()),
        "p50": float(np.percentile(xs, 50)),
        "p90": float(np.percentile(xs, 90)),
    }


def corrcoef(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or xs.std() < 1e-12 or ys.std() < 1e-12:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    index = load_json(DATA_DIR / "dataset_index.json")
    by_dist = collections.defaultdict(dict)
    repeat_counts = collections.Counter()
    for pair in index:
        traj = load_json(DATA_DIR / "trajectories" / pair["trajectory_file"])
        gamma = round(float(traj["gamma"]), 6)
        repeat_counts[(pair["distribution_file"], gamma)] += 1
        by_dist[pair["distribution_file"]][gamma] = pair["trajectory_file"]

    group_records = []
    monotonic_coverage = []
    monotonic_path_length = []
    pair_dists = collections.defaultdict(list)
    coverage_gaps = collections.defaultdict(list)
    gamma_gap_values, traj_gap_values = [], []
    same_gamma_repeats = [v for v in repeat_counts.values() if v > 1]

    for dfile, items in sorted(by_dist.items()):
        if not all(round(g, 6) in items for g in GAMMAS):
            continue
        dist = load_json(DATA_DIR / "distributions" / dfile)
        packed, valid = pack_distribution(dist)
        cov_by_g, len_by_g, xy_by_g = {}, {}, {}
        for g in GAMMAS:
            traj = load_json(DATA_DIR / "trajectories" / items[round(g, 6)])
            xy = np.asarray(traj["states"], dtype=np.float64)[:, :2]
            xy_by_g[g] = xy
            cov_by_g[g] = hotspot_coverage_score(xy, packed, valid)
            len_by_g[g] = path_length(xy)
        cov_values_low_to_high_gamma = [cov_by_g[g] for g in GAMMAS]
        len_values_low_to_high_gamma = [len_by_g[g] for g in GAMMAS]
        # Smaller gamma should generally have stronger coverage and longer/more exploratory path.
        monotonic_coverage.append(all(cov_values_low_to_high_gamma[i] >= cov_values_low_to_high_gamma[i + 1] - 1e-9 for i in range(len(GAMMAS) - 1)))
        monotonic_path_length.append(all(len_values_low_to_high_gamma[i] >= len_values_low_to_high_gamma[i + 1] - 1e-9 for i in range(len(GAMMAS) - 1)))
        for ga, gb in itertools.combinations(GAMMAS, 2):
            d = mean_point_dist(xy_by_g[ga], xy_by_g[gb])
            pair_dists[f"{ga:.2f}-{gb:.2f}"].append(d)
            coverage_gaps[f"{ga:.2f}-{gb:.2f}"].append(cov_by_g[ga] - cov_by_g[gb])
            gamma_gap_values.append(abs(math.log(ga) - math.log(gb)))
            traj_gap_values.append(d)
        group_records.append({"distribution_file": dfile, "coverage": cov_by_g, "path_length": len_by_g})

    summary = {
        "data_dir": str(DATA_DIR),
        "n_groups": len(group_records),
        "gammas": GAMMAS,
        "coverage_monotonic_low_gamma_stronger_rate": float(np.mean(monotonic_coverage)) if monotonic_coverage else math.nan,
        "path_length_monotonic_low_gamma_longer_rate": float(np.mean(monotonic_path_length)) if monotonic_path_length else math.nan,
        "same_distribution_gamma_pair_distance_m": {k: summarize(v) for k, v in sorted(pair_dists.items())},
        "same_distribution_coverage_gap_low_minus_high": {k: summarize(v) for k, v in sorted(coverage_gaps.items())},
        "log_gamma_gap_vs_trajectory_distance_corr": corrcoef(gamma_gap_values, traj_gap_values),
        "same_distribution_same_gamma_repeat_keys": len(same_gamma_repeats),
        "same_gamma_random_variation_available": bool(same_gamma_repeats),
        "same_gamma_random_variation_note": "No repeated oracle samples per distribution/gamma in this dataset; random variation cannot be estimated from group-complete continuation data.",
        "sanity_pass": bool(group_records)
        and float(np.mean(monotonic_coverage)) > 0.55
        and pair_dists.get("0.01-0.20", [0.0])
        and float(np.mean(pair_dists["0.01-0.20"])) > 0.3,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
