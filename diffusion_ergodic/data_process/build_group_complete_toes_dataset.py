#!/usr/bin/env python3
"""Build a group-complete TOES continuation dataset index.

The source continuation dataset is gamma-balanced only at the global bucket
level. Pairwise gamma shape learning needs every selected distribution to carry
all target gammas. This script creates a new dataset directory whose
dataset_index.json contains complete distribution groups and whose trajectory /
distribution files are symlinked back to the source dataset.
"""

import argparse
import collections
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_SOURCE = Path("/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic/data/ergodic_toes_continuation_v1")
DEFAULT_OUTPUT = Path("/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic/data/ergodic_toes_continuation_v1_groupcomplete_balanced")
DEFAULT_GAMMAS = [0.01, 0.03, 0.1, 0.2]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def rel_or_abs_symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    dst.symlink_to(src)


def gamma_key(gamma, gammas, tol=1e-6):
    for target in gammas:
        if abs(float(gamma) - float(target)) < tol:
            return float(target)
    return None


def trajectory_state_array(traj_data, trajectory_len):
    states = np.asarray(traj_data["states"], dtype=np.float64)
    pos = states[:, :2]
    if states.shape[1] > 3:
        vel = states[:, 3:4]
    elif states.shape[0] > 1:
        dt = float(traj_data.get("time_step", 1.0))
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        vel = np.vstack([np.zeros((1, 1)), (d / max(dt, 1e-8))[:, None]])
    else:
        vel = np.zeros((states.shape[0], 1), dtype=np.float64)
    heading = states[:, 2:3] if states.shape[1] > 2 else np.zeros((states.shape[0], 1), dtype=np.float64)
    rs = np.hstack([pos, heading, vel])
    if len(rs) > trajectory_len:
        idx = np.linspace(0, len(rs) - 1, trajectory_len).astype(int)
        rs = rs[idx]
    elif len(rs) < trajectory_len:
        rs = np.vstack([rs, np.zeros((trajectory_len - len(rs), rs.shape[1]), dtype=np.float64)])
    return rs.astype(np.float32)


def compute_stats(output_dir, index, trajectory_len):
    robot_states = []
    trajectory_points = []
    for pair in index:
        traj_path = output_dir / "trajectories" / pair["trajectory_file"]
        traj = load_json(traj_path)
        states = trajectory_state_array(traj, trajectory_len)
        robot_states.append(states[0])
        trajectory_points.append(states)
    robot_states = np.asarray(robot_states, dtype=np.float32)
    trajectory_points = np.concatenate(trajectory_points, axis=0).astype(np.float32)
    eps = 1e-6
    return {
        "trajectories": {
            "mean": torch.tensor(trajectory_points.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(np.maximum(trajectory_points.std(axis=0), eps), dtype=torch.float32),
        },
        "robot_state": {
            "mean": torch.tensor(robot_states.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(np.maximum(robot_states.std(axis=0), eps), dtype=torch.float32),
        },
    }


def build_dataset(source_dir, output_dir, gammas, seed, max_distributions, trajectory_len):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    src_index = load_json(source_dir / "dataset_index.json")
    if not isinstance(src_index, list):
        raise RuntimeError(f"Expected list dataset_index.json, got {type(src_index)}")

    grouped = collections.defaultdict(dict)
    counts_all = collections.Counter()
    for pair in src_index:
        traj_path = source_dir / "trajectories" / pair["trajectory_file"]
        traj = load_json(traj_path)
        g = gamma_key(traj["gamma"], gammas)
        if g is None:
            continue
        counts_all[f"{g:.6f}"] += 1
        dfile = pair["distribution_file"]
        if g in grouped[dfile]:
            raise RuntimeError(f"Duplicate gamma {g} for {dfile}; refusing ambiguous group")
        grouped[dfile][g] = {
            "distribution_file": dfile,
            "trajectory_file": pair["trajectory_file"],
        }

    complete = sorted(d for d, items in grouped.items() if all(g in items for g in gammas))
    rng = random.Random(seed)
    rng.shuffle(complete)
    if max_distributions is not None:
        complete = complete[: int(max_distributions)]
    complete = sorted(complete)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "distributions").mkdir(exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)

    index = []
    file_index = []
    for dfile in complete:
        src_dist = source_dir / "distributions" / dfile
        rel_or_abs_symlink(src_dist, output_dir / "distributions" / dfile)
        group_entries = []
        for gamma in gammas:
            pair = grouped[dfile][gamma]
            src_traj = source_dir / "trajectories" / pair["trajectory_file"]
            rel_or_abs_symlink(src_traj, output_dir / "trajectories" / pair["trajectory_file"])
            item = {
                "distribution_file": dfile,
                "trajectory_file": pair["trajectory_file"],
                "gamma": gamma,
            }
            index.append({"distribution_file": dfile, "trajectory_file": pair["trajectory_file"]})
            group_entries.append(item)
        file_index.append({"distribution_file": dfile, "trajectories": group_entries})

    gamma_counts = collections.Counter(f"{x['gamma']:.6f}" for group in file_index for x in group["trajectories"])
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_gammas": gammas,
        "seed": seed,
        "source_gamma_counts": dict(sorted(counts_all.items())),
        "complete_distribution_count_available": len([d for d, items in grouped.items() if all(g in items for g in gammas)]),
        "complete_distribution_count_selected": len(complete),
        "trajectory_count": len(index),
        "gamma_counts": dict(sorted(gamma_counts.items())),
        "trajectory_len": trajectory_len,
        "symlinked_files": True,
        "group_complete": True,
    }

    save_json(output_dir / "dataset_index.json", index)
    save_json(output_dir / "dataset_group_index.json", file_index)
    save_json(output_dir / "dataset_group_complete_summary.json", summary)
    stats = compute_stats(output_dir, index, trajectory_len)
    torch.save(stats, output_dir / "dataset_stats.pt")
    return summary, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_distributions", type=int, default=None)
    parser.add_argument("--trajectory_len", type=int, default=101)
    args = parser.parse_args()

    summary, stats = build_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        gammas=[float(g) for g in args.gammas],
        seed=args.seed,
        max_distributions=args.max_distributions,
        trajectory_len=args.trajectory_len,
    )
    print(json.dumps(summary, indent=2))
    print("robot_state mean:", stats["robot_state"]["mean"].tolist())
    print("robot_state std:", stats["robot_state"]["std"].tolist())
    print("trajectory mean:", stats["trajectories"]["mean"].tolist())
    print("trajectory std:", stats["trajectories"]["std"].tolist())


if __name__ == "__main__":
    main()
