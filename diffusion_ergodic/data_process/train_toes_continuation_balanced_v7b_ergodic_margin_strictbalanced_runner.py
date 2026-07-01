import collections
import itertools
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from timm.models.layers import Mlp


PROJECT_DIR = "/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic"
DATA_DIR = "/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic/data/ergodic_toes_continuation_v1_balanced"
STATS_DATA_DIR = DATA_DIR
OUT_DIR = Path("/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic_results/toes_continuation_balanced_v7b_ergodic_margin_strictbalanced")
INIT_CKPT = "/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic_results/toes_continuation_balanced_v6b_endpoint_shape_groupcomplete/best_mixed_model.pth"
GAMMAS = [0.01, 0.03, 0.1, 0.2]
SEED = 20260630
N_DISTS = 20
N_DPM_SEEDS = 3
INFERENCE_STEPS = 50
T_FIXED = 0.5
SHAPE_LOSS_WEIGHT = 0.03
ERGODIC_LOSS_WEIGHT = 0.02
SMOOTH_LOSS_WEIGHT = 0.005
ENDPOINT_LOSS_WEIGHT = 0.05
ERGODIC_MARGIN = 0.005
HOTSPOT_COV_SCALE = 0.20
MIXED_PAIRWISE_WEIGHT = 0.75
MIXED_ERGODIC_WEIGHT = 0.50
MIXED_SMOOTH_WEIGHT = 0.05
MAX_EPOCHS = 180
PATIENCE = 28
BATCH_DISTS = 12


os.chdir(PROJECT_DIR)
import sys
sys.path.insert(0, PROJECT_DIR)
from models.diffusion_utils.sampling import dpm_sampler


def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_ns(x) for x in d]
    return d


def as_tensor_f32(x):
    return x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


class ErgodicDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_trajectory_len=101, use_index=True, max_gaussians=20):
        self.data_dir = data_dir
        self.transform = transform
        self.max_trajectory_len = max_trajectory_len
        self.max_gaussians = max_gaussians
        self.distributions_dir = os.path.join(data_dir, "distributions")
        self.trajectories_dir = os.path.join(data_dir, "trajectories")
        self.data_pairs = []
        index_path = os.path.join(data_dir, "dataset_index.json")
        if use_index and os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.data_pairs = json.load(f)
        if not self.data_pairs:
            raise RuntimeError(f"No dataset_index entries found in {index_path}")

    def __len__(self):
        return len(self.data_pairs)

    def _generate_distribution_grid(self, dist_data, grid_size=(32, 32)):
        bounds = dist_data.get("bounds", [[0, 3], [0, 3]])
        centers = np.asarray(dist_data["params"]["centers"])
        covs = np.asarray(dist_data["params"]["covs"])
        weights = np.asarray(dist_data["params"]["weights"])
        n = int(dist_data["params"]["n_gaussians"])
        x = np.linspace(bounds[0][0], bounds[0][1], grid_size[0])
        y = np.linspace(bounds[1][0], bounds[1][1], grid_size[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=np.float64)
        for i in range(n):
            cx, cy = centers[i]
            c = covs[i]
            if np.isscalar(c):
                sx, sy = c, c
            elif len(np.shape(c)) == 1:
                sx, sy = c[0], c[1]
            else:
                sx, sy = np.sqrt(np.diag(c))
            Z += weights[i] * np.exp(-(((X - cx) ** 2) / (2 * sx ** 2 + 1e-8) + ((Y - cy) ** 2) / (2 * sy ** 2 + 1e-8)))
        Z /= Z.max() + 1e-8
        return Z

    def _process_trajectory(self, traj_data):
        states = np.asarray(traj_data["states"], dtype=np.float64)
        pos = states[:, :2]
        if states.shape[1] > 3:
            vel = states[:, 3:4]
        elif states.shape[0] > 1:
            dt = float(traj_data.get("time_step", 1.0))
            d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            vel = np.vstack([np.zeros((1, 1)), (d / dt)[:, None]])
        else:
            vel = np.zeros((states.shape[0], 1))
        heading = states[:, 2:3] if states.shape[1] > 2 else np.zeros((states.shape[0], 1))
        rs = np.hstack([pos, heading, vel])
        T = self.max_trajectory_len
        controls = np.zeros((T, 2), dtype=np.float32)
        if len(rs) > T:
            idx = np.linspace(0, len(rs) - 1, T).astype(int)
            rs = rs[idx]
        elif len(rs) < T:
            rs = np.vstack([rs, np.zeros((T - len(rs), rs.shape[1]))])
        return (
            rs,
            controls,
            float(traj_data.get("time_step", 1.0)),
            float(traj_data.get("total_time", 0.0)),
            float(traj_data.get("ergodic_metric", 0.0)),
            float(traj_data["gamma"]),
        )

    def _pack_gaussian_params(self, dist_data):
        n = int(dist_data["params"]["n_gaussians"])
        centers = np.asarray(dist_data["params"]["centers"])
        weights = np.asarray(dist_data["params"]["weights"])
        covs_raw = dist_data["params"]["covs"]
        packed = np.zeros((self.max_gaussians, 7), dtype=np.float32)
        mask = np.zeros((self.max_gaussians,), dtype=bool)
        mask[n:] = True
        for i in range(min(n, self.max_gaussians)):
            packed[i, 0:2] = centers[i]
            c = covs_raw[i]
            if np.isscalar(c):
                cov_mat = np.array([c, 0, 0, c])
            elif len(np.shape(c)) == 1:
                cov_mat = np.array([c[0], 0, 0, c[1]])
            else:
                cov_mat = np.asarray(c).flatten()
            if cov_mat.size == 4:
                packed[i, 2:6] = cov_mat
            packed[i, 6] = weights[i]
        return packed, mask

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        with open(os.path.join(self.distributions_dir, pair["distribution_file"]), "r") as f:
            dist_data = json.load(f)
        with open(os.path.join(self.trajectories_dir, pair["trajectory_file"]), "r") as f:
            traj_data = json.load(f)
        grid = self._generate_distribution_grid(dist_data)
        rs, controls, _, _, _, gamma = self._process_trajectory(traj_data)
        gmm_packed, gmm_padding_mask = self._pack_gaussian_params(dist_data)
        sample = {
            "distribution": as_tensor_f32(grid).unsqueeze(0),
            "robot_state": as_tensor_f32(rs[0]),
            "trajectories": as_tensor_f32(rs),
            "controls": as_tensor_f32(controls),
            "gamma": as_tensor_f32(gamma).view(1),
            "ergodic_metric": as_tensor_f32(float(traj_data.get("ergodic_metric", 0.0))).view(1),
            "gaussian_packed": as_tensor_f32(gmm_packed),
            "gaussian_padding_mask": torch.tensor(gmm_padding_mask, dtype=torch.bool),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class Standardizer:
    def __init__(self, stats, device="cpu"):
        self.traj_mean = stats["trajectories"]["mean"].view(-1).to(device)
        self.traj_std = stats["trajectories"]["std"].view(-1).to(device)
        self.rs_mean = stats["robot_state"]["mean"].view(-1).to(device)
        self.rs_std = stats["robot_state"]["std"].view(-1).to(device)

    def __call__(self, sample):
        safe_t = torch.where(self.traj_std < 1e-6, torch.ones_like(self.traj_std), self.traj_std)
        safe_r = torch.where(self.rs_std < 1e-6, torch.ones_like(self.rs_std), self.rs_std)
        sample = dict(sample)
        sample["trajectories"] = (sample["trajectories"] - self.traj_mean) / safe_t
        sample["robot_state"] = (sample["robot_state"] - self.rs_mean) / safe_r
        return sample


class NormalizedDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
        self.data_pairs = ds.data_pairs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx])


class EvalNormalizer:
    def __init__(self, stats, device):
        self.t_mean = stats["trajectories"]["mean"].view(1, 1, 4).to(device)
        self.t_std = stats["trajectories"]["std"].view(1, 1, 4).to(device)
        self.rs_mean = stats["robot_state"]["mean"].view(1, 4).to(device)
        self.rs_std = stats["robot_state"]["std"].view(1, 4).to(device)

    def denormalize_traj(self, x):
        safe = torch.where(self.t_std < 1e-6, torch.ones_like(self.t_std), self.t_std)
        return x * safe + self.t_mean

    def denormalize_state(self, x):
        safe = torch.where(self.rs_std < 1e-6, torch.ones_like(self.rs_std), self.rs_std)
        return x * safe + self.rs_mean


def normalizer_to_stats(normalizer):
    return {
        "trajectories": {
            "mean": normalizer.t_mean.view(-1),
            "std": normalizer.t_std.view(-1),
        }
    }


class VPSDE_linear:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self._beta_min = float(beta_min)
        self._beta_max = float(beta_max)

    def marginal_prob(self, x, t):
        shape = x.shape
        t = t.view(-1, *([1] * (len(shape) - 1)))
        mlc = -0.25 * t ** 2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t
        mean = torch.exp(mlc) * x
        std = torch.sqrt(torch.clamp(1 - torch.exp(2 * mlc), min=1e-6))
        return mean, std


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.half = hidden_dim // 2
        self.emb = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(0, self.half, device=t.device) / self.half)
        x = torch.cat([torch.cos(t[:, None] * freqs[None, :]), torch.sin(t[:, None] * freqs[None, :])], dim=-1)
        return self.emb(x)


class ErgodicEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = int(getattr(cfg.model, "hidden_dim", 384))
        state_dim = int(cfg.robot_state_dim)
        self.gmm_embedder = nn.Sequential(nn.Linear(7, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.robot_embedder = nn.Sequential(nn.Linear(state_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.gamma_embedder = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.final_ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, inputs):
        rs_emb = self.robot_embedder(inputs["robot_state"]).unsqueeze(1)
        gamma_emb = self.gamma_embedder(inputs["gamma"]).unsqueeze(1)
        gmm_emb = self.gmm_embedder(inputs["gaussian_packed"])
        src = torch.cat([rs_emb, gamma_emb, gmm_emb], dim=1)
        prefix = torch.zeros((src.shape[0], 2), device=src.device, dtype=torch.bool)
        mask = torch.cat([prefix, inputs["gaussian_padding_mask"]], dim=1)
        return {"encoding": self.transformer(src, src_key_padding_mask=mask), "encoding_mask": mask}


class V5TemporalDiTBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.sa = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ca = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4), act_layer=nn.GELU, drop=dropout)
        self.gamma_film = nn.Sequential(nn.Linear(1, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        self.drop = nn.Dropout(dropout)

    def _film(self, x, gamma):
        if gamma is None:
            return x
        film = self.gamma_film(gamma.view(gamma.shape[0], 1).to(x.device))
        scale, shift = film.chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, context=None, context_key_padding_mask=None, gamma=None):
        h = self._film(self.norm1(x), gamma)
        h = self.sa(h, h, h)[0]
        x = x + self.drop(h)
        h = self._film(self.norm2(x), gamma)
        h = self.ca(h, context, context, key_padding_mask=context_key_padding_mask)[0]
        x = x + self.drop(h)
        h = self._film(self.norm3(x), gamma)
        h = self.mlp(h)
        return x + self.drop(h)


class V5TemporalDiT(nn.Module):
    def __init__(self, T, D, hidden_dim, depth, heads, dropout=0.1, model_type="x_start", config=None):
        super().__init__()
        self.T = T
        self.D = D
        self.hidden_dim = hidden_dim
        self.output_dim = T * D
        self.model_type = model_type
        self.config_ref = config
        self.state_proj = nn.Linear(D, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.gamma_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([V5TemporalDiTBlock(hidden_dim, heads, dropout=dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final = nn.Linear(hidden_dim, D)

    def _condition_to_traj_norm(self, cond, device):
        cfg = self.config_ref.normalizer
        rs_mean = cfg.robot_state.mean.to(device)
        rs_std = cfg.robot_state.std.to(device)
        tr_mean = cfg.trajectories.mean.to(device)
        tr_std = cfg.trajectories.std.to(device)
        safe_rs = torch.where(rs_std < 1e-6, torch.ones_like(rs_std), rs_std)
        safe_tr = torch.where(tr_std < 1e-6, torch.ones_like(tr_std), tr_std)
        cond_phys = cond.to(device) * safe_rs + rs_mean
        return (cond_phys - tr_mean) / safe_tr

    def forward(self, x, t, context, conditions=None, gamma=None, context_mask=None):
        B = x.shape[0]
        xv = x.view(B, self.T, self.D)
        if isinstance(conditions, dict) and 0 in conditions and conditions[0] is not None:
            xv = xv.clone()
            xv[:, 0, :] = self._condition_to_traj_norm(conditions[0], xv.device)
        if isinstance(conditions, dict) and (self.T - 1) in conditions and conditions[self.T - 1] is not None:
            if not torch.is_tensor(xv) or xv._base is not None:
                xv = xv.clone()
            xv[:, -1, :] = conditions[self.T - 1].to(xv.device)
        h = self.state_proj(xv) + self.pos_embed + self.t_embedder(t).unsqueeze(1)
        if gamma is not None:
            h = h + self.gamma_embed(gamma.view(gamma.shape[0], 1).to(h.device)).unsqueeze(1)
        for block in self.blocks:
            h = block(h, context, context_key_padding_mask=context_mask, gamma=gamma)
        return self.final(self.final_norm(h)).view(B, self.output_dim)


class V5TemporalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.T = int(config.trajectory_len)
        self.D = int(config.robot_state_dim)
        hidden = int(config.model.hidden_dim)
        depth = int(config.model.decoder_depth)
        heads = int(config.model.num_heads)
        drop = float(config.model.decoder_drop_path_rate)
        model_type = str(getattr(config.diffusion, "model_type", "x_start"))
        self.output_dim = self.T * self.D
        self.dit = V5TemporalDiT(self.T, self.D, hidden, depth, heads, dropout=drop, model_type=model_type, config=config)

    def forward(self, x_t, t, context, conditions=None, gamma=None, context_mask=None):
        out = self.dit(x_t.view(x_t.shape[0], -1), t, context, conditions=conditions, gamma=gamma, context_mask=context_mask)
        return out.view(x_t.shape[0], self.T, self.D)

    @torch.no_grad()
    def inference(self, enc_out, inputs, steps=None, **kwargs):
        enc = enc_out["encoding"]
        mask = enc_out.get("encoding_mask")
        B = enc.shape[0]
        x_T = torch.randn(B, self.output_dim, device=enc.device)
        other = {"context": enc, "context_mask": mask, "gamma": inputs.get("gamma")}
        if inputs.get("robot_state") is not None:
            other["conditions"] = {0: inputs["robot_state"]}
        if inputs.get("end_state") is not None:
            other.setdefault("conditions", {})[self.T - 1] = inputs["end_state"]
        x0 = dpm_sampler(model=self.dit, x_T=x_T, other_model_params=other, diffusion_steps=steps or INFERENCE_STEPS, **kwargs)
        traj = x0.view(B, self.T, self.D)
        if inputs.get("robot_state") is not None:
            traj[:, 0, :] = self.dit._condition_to_traj_norm(inputs["robot_state"], traj.device)
        if inputs.get("end_state") is not None:
            traj[:, -1, :] = inputs["end_state"].to(traj.device)
        return {"prediction": traj}


class ErgodicDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ErgodicEncoder(config)
        self.sde = VPSDE_linear(config.diffusion.beta_min, config.diffusion.beta_max)
        self.decoder = V5TemporalDecoder(config)

    def _build_enc_in(self, inputs):
        return {k: inputs[k] for k in ("robot_state", "distribution", "gaussian_packed", "gaussian_padding_mask", "gamma") if k in inputs}

    def forward(self, inputs, training=True):
        enc = self.encoder(self._build_enc_in(inputs))
        if not training:
            return enc
        x0 = inputs["trajectories"]
        t = inputs.get("diffusion_time")
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        mean, std = self.sde.marginal_prob(x0, t)
        x_t = mean + std * torch.randn_like(x0)
        conditions = {0: inputs["robot_state"]}
        if "end_state" in inputs:
            conditions[self.decoder.T - 1] = inputs["end_state"]
        pred = self.decoder(x_t, t, enc["encoding"], conditions=conditions, gamma=inputs["gamma"], context_mask=enc.get("encoding_mask"))
        return {"prediction": pred, "target": x0}

    @torch.no_grad()
    def inference(self, inputs, **kwargs):
        enc = self.encoder(self._build_enc_in(inputs))
        return self.decoder.inference(enc, inputs, **kwargs)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def corrcoef(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or xs.std() < 1e-12 or ys.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def mean_point_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def path_length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def curvature_stats(xy):
    d = np.diff(xy, axis=0)
    step = np.linalg.norm(d, axis=1)
    angles = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    curv = np.abs(np.diff(angles)) / np.maximum(step[1:], 1e-8) if len(angles) > 1 else np.array([0.0])
    return {
        "first_step": float(step[0]) if len(step) else 0.0,
        "first5_max_step": float(step[:5].max()) if len(step) else 0.0,
        "first5_mean_step": float(step[:5].mean()) if len(step) else 0.0,
        "first5_mean_curvature": float(curv[:4].mean()) if len(curv) else 0.0,
    }


def batch_to_inputs(batch, device, include_traj=False):
    out = {
        "distribution": batch["distribution"].to(device),
        "robot_state": batch["robot_state"].to(device),
        "gaussian_packed": batch["gaussian_packed"].to(device),
        "gaussian_padding_mask": batch["gaussian_padding_mask"].to(device),
        "gamma": batch["gamma"].to(device),
    }
    if "trajectories" in batch:
        out["end_state"] = batch["trajectories"][:, -1, :].to(device)
    if include_traj:
        traj = batch["trajectories"].to(device)
        out["trajectories"] = traj
        out["end_state"] = traj[:, -1, :]
        out["diffusion_time"] = torch.full((traj.shape[0],), T_FIXED, device=device)
    return out


def make_model_input(raw_ds, standardizer, pair_idx, gamma, device):
    sample = standardizer(raw_ds[pair_idx])
    sample["gamma"] = torch.tensor([gamma], dtype=torch.float32)
    return {
        "distribution": sample["distribution"].unsqueeze(0).to(device),
        "robot_state": sample["robot_state"].unsqueeze(0).to(device),
        "end_state": sample["trajectories"][-1:].to(device),
        "gaussian_packed": sample["gaussian_packed"].unsqueeze(0).to(device),
        "gaussian_padding_mask": sample["gaussian_padding_mask"].unsqueeze(0).to(device),
        "gamma": sample["gamma"].view(1, 1).to(device),
    }


def jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def build_val_loader(raw_ds, standardizer, batch_size=64):
    all_dist_files = sorted(set(p["distribution_file"] for p in raw_ds.data_pairs))
    rng = np.random.RandomState(42)
    rng.shuffle(all_dist_files)
    n_test = max(1, int(np.floor(0.05 * len(all_dist_files))))
    n_val = max(1, int(np.floor(0.10 * len(all_dist_files))))
    val_set = set(all_dist_files[n_test:n_test + n_val])
    val_idx = [i for i, p in enumerate(raw_ds.data_pairs) if p["distribution_file"] in val_set]
    ds = NormalizedDataset(raw_ds, standardizer)
    return DataLoader(ds, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=0), val_idx


def build_groupcomplete_val_loader(raw_ds, standardizer, val_set, batch_size=64):
    grouped = collections.defaultdict(dict)
    for idx, pair in enumerate(raw_ds.data_pairs):
        if pair["distribution_file"] not in val_set:
            continue
        traj = json.load(open(os.path.join(raw_ds.trajectories_dir, pair["trajectory_file"])))
        gamma = float(traj["gamma"])
        for target in GAMMAS:
            if abs(gamma - target) < 1e-6:
                grouped[pair["distribution_file"]][target] = idx
    val_idx = []
    for _, items in sorted(grouped.items()):
        if all(g in items for g in GAMMAS):
            val_idx.extend([items[g] for g in GAMMAS])
    ds = NormalizedDataset(raw_ds, standardizer)
    return DataLoader(ds, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=0), val_idx


@torch.no_grad()
def evaluate_validation(model, val_loader, normalizer, device, max_samples=512):
    direct_errs, dpm_errs, direct_fdes, dpm_fdes = [], [], [], []
    start_errors, jump_stats = [], []
    direct_info_deficits, direct_info_ratios, direct_smooth_vals = [], [], []
    n = 0
    for batch in val_loader:
        inputs = batch_to_inputs(batch, device, include_traj=True)
        gt_norm = inputs["trajectories"]
        gt_phys = normalizer.denormalize_traj(gt_norm).cpu().numpy()
        enc = model.encoder(model._build_enc_in(inputs))
        direct = model.decoder(
            gt_norm,
            inputs["diffusion_time"],
            enc["encoding"],
            conditions={0: inputs["robot_state"], model.decoder.T - 1: inputs["end_state"]},
            gamma=inputs["gamma"],
            context_mask=enc.get("encoding_mask"),
        )
        dpm = model.inference(batch_to_inputs(batch, device), steps=INFERENCE_STEPS)["prediction"]
        direct_phys = normalizer.denormalize_traj(direct).cpu().numpy()
        dpm_phys = normalizer.denormalize_traj(dpm).cpu().numpy()
        robot_phys = normalizer.denormalize_state(inputs["robot_state"]).cpu().numpy()
        packed = batch["gaussian_packed"].to(device)
        mask = batch["gaussian_padding_mask"].to(device)
        direct_info = gmm_hotspot_coverage_score(denorm_traj_xy(direct, normalizer_to_stats(normalizer)), packed, mask)
        oracle_info = gmm_hotspot_coverage_score(denorm_traj_xy(gt_norm, normalizer_to_stats(normalizer)), packed, mask)
        direct_smooth = smoothness_loss(direct, normalizer_to_stats(normalizer))
        for i in range(gt_phys.shape[0]):
            gt_xy = gt_phys[i, :, :2]
            direct_xy = direct_phys[i, :, :2]
            dpm_xy = dpm_phys[i, :, :2]
            direct_errs.append(float(np.linalg.norm(direct_xy - gt_xy, axis=1).mean()))
            dpm_errs.append(float(np.linalg.norm(dpm_xy - gt_xy, axis=1).mean()))
            direct_fdes.append(float(np.linalg.norm(direct_xy[-1] - gt_xy[-1])))
            dpm_fdes.append(float(np.linalg.norm(dpm_xy[-1] - gt_xy[-1])))
            start_errors.append(float(np.linalg.norm(dpm_xy[0] - robot_phys[i, :2])))
            jump_stats.append(curvature_stats(dpm_xy))
            oi = float(oracle_info[i].detach().cpu())
            pi = float(direct_info[i].detach().cpu())
            direct_info_deficits.append(max(0.0, oi - pi))
            direct_info_ratios.append(pi / max(oi, 1e-8))
            direct_smooth_vals.append(float(direct_smooth.detach().cpu()))
            n += 1
            if n >= max_samples:
                break
        if n >= max_samples:
            break
    return {
        "n": n,
        "direct_xy_mae_m": summarize(direct_errs),
        "dpm_inference_xy_mae_m": summarize(dpm_errs),
        "direct_fde_m": summarize(direct_fdes),
        "dpm_fde_m": summarize(dpm_fdes),
        "start_anchor_error_m": summarize(start_errors),
        "first_step_m": summarize([x["first_step"] for x in jump_stats]),
        "first5_max_step_m": summarize([x["first5_max_step"] for x in jump_stats]),
        "first5_mean_curvature": summarize([x["first5_mean_curvature"] for x in jump_stats]),
        "direct_info_deficit": summarize(direct_info_deficits),
        "direct_info_ratio": summarize(direct_info_ratios),
        "direct_smoothness": summarize(direct_smooth_vals),
    }


def load_model(ckpt_path, stats, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = to_ns(ckpt["config"])
    cfg.data.data_dir = DATA_DIR
    cfg.data_dir = DATA_DIR
    cfg.data.trajectory_len = 101
    cfg.trajectory_len = 101
    cfg.normalizer.robot_state.mean = stats["robot_state"]["mean"].to(device)
    cfg.normalizer.robot_state.std = stats["robot_state"]["std"].to(device)
    cfg.normalizer.trajectories = SimpleNamespace(mean=stats["trajectories"]["mean"].to(device), std=stats["trajectories"]["std"].to(device))
    model = ErgodicDiffusionModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def select_distributions(pairs, by_dist):
    eligible = []
    for dfile, items in by_dist.items():
        if all(any(abs(x[1] - g) < 1e-6 for x in items) for g in GAMMAS):
            oracle = []
            for g in GAMMAS:
                item = [x for x in items if abs(x[1] - g) < 1e-6][0]
                oracle.append(np.asarray(item[3]["states"], dtype=float)[:, :2])
            score = np.mean([mean_point_dist(oracle[i], oracle[j]) for i in range(len(oracle)) for j in range(i + 1, len(oracle))])
            eligible.append((float(score), dfile))
    return [d for _, d in sorted(eligible, reverse=True)[:N_DISTS]]


@torch.no_grad()
def evaluate_checkpoint(label, ckpt_path, raw_ds, standardizer, normalizer, val_loader, by_dist, selected, stats, device):
    model, ckpt = load_model(ckpt_path, stats, device)
    validation = evaluate_validation(model, val_loader, normalizer, device, max_samples=512)
    out_dir = OUT_DIR / label
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    direct_pairwise, dpm_pairwise, oracle_pairwise = [], [], []
    direct_low_high, dpm_low_high, oracle_low_high = [], [], []
    records, start_artifacts, fig_paths = [], [], []

    colors = {0.01: "#1b9e77", 0.03: "#d95f02", 0.1: "#7570b3", 0.2: "#e7298a"}

    for dist_i, dfile in enumerate(selected):
        items = by_dist[dfile]
        base_idx = [x for x in items if abs(x[1] - 0.1) < 1e-6][0][0]
        base_sample = standardizer(raw_ds[base_idx])
        base_x0 = base_sample["trajectories"].unsqueeze(0).to(device)
        base_t = torch.full((1,), T_FIXED, device=device)
        torch.manual_seed(SEED + dist_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + dist_i)
        mean, std = model.sde.marginal_prob(base_x0, base_t)
        fixed_xt = mean + std * torch.randn_like(base_x0)

        direct_by_gamma, dpm_by_gamma, oracle_by_gamma = {}, {}, {}
        for gamma in GAMMAS:
            item = [x for x in items if abs(x[1] - gamma) < 1e-6][0]
            oracle_by_gamma[gamma] = np.asarray(item[3]["states"], dtype=float)[:, :2]
            inp = make_model_input(raw_ds, standardizer, base_idx, gamma, device)
            enc = model.encoder(model._build_enc_in(inp))
            direct_norm = model.decoder(
                fixed_xt,
                base_t,
                enc["encoding"],
                conditions={0: inp["robot_state"], model.decoder.T - 1: inp["end_state"]},
                gamma=inp["gamma"],
                context_mask=enc.get("encoding_mask"),
            )
            direct_by_gamma[gamma] = normalizer.denormalize_traj(direct_norm).cpu().numpy()[0, :, :2]

            dpm_preds = []
            for seed_offset in range(N_DPM_SEEDS):
                torch.manual_seed(SEED + 10000 * dist_i + seed_offset)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + 10000 * dist_i + seed_offset)
                pred = model.inference(inp, steps=INFERENCE_STEPS)["prediction"]
                pred_xy = normalizer.denormalize_traj(pred).cpu().numpy()[0, :, :2]
                dpm_preds.append(pred_xy)
                start_artifacts.append({
                    "distribution_file": dfile,
                    "gamma": gamma,
                    "seed_offset": seed_offset,
                    "start_anchor_error": float(np.linalg.norm(pred_xy[0] - oracle_by_gamma[gamma][0])),
                    **curvature_stats(pred_xy),
                })
            dpm_by_gamma[gamma] = np.stack(dpm_preds).mean(axis=0)

        for ga, gb in itertools.combinations(GAMMAS, 2):
            od = mean_point_dist(oracle_by_gamma[ga], oracle_by_gamma[gb])
            dd = mean_point_dist(direct_by_gamma[ga], direct_by_gamma[gb])
            sd = mean_point_dist(dpm_by_gamma[ga], dpm_by_gamma[gb])
            oracle_pairwise.append(od)
            direct_pairwise.append(dd)
            dpm_pairwise.append(sd)
            if abs(ga - 0.01) < 1e-9 and abs(gb - 0.2) < 1e-9:
                oracle_low_high.append(od)
                direct_low_high.append(dd)
                dpm_low_high.append(sd)
            records.append({
                "distribution_file": dfile,
                "gamma_pair": [ga, gb],
                "oracle_mean_point_distance_m": od,
                "direct_fixed_xt_mean_point_distance_m": dd,
                "dpm_mean_point_distance_m": sd,
                "oracle_path_length_delta_m": path_length(oracle_by_gamma[gb]) - path_length(oracle_by_gamma[ga]),
                "dpm_path_length_delta_m": path_length(dpm_by_gamma[gb]) - path_length(dpm_by_gamma[ga]),
            })

        if dist_i < 6:
            dist_data = json.load(open(os.path.join(DATA_DIR, "distributions", dfile)))
            centers = np.asarray(dist_data["params"].get("centers", []), dtype=float)
            weights = np.asarray(dist_data["params"].get("weights", []), dtype=float)
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
            for ax, title, data in [
                (axes[0], f"{label}: direct fixed-xt", direct_by_gamma),
                (axes[1], f"{label}: DPM mean", dpm_by_gamma),
                (axes[2], "oracle", oracle_by_gamma),
            ]:
                ax.set_title(title)
                if centers.size:
                    sizes = 50 + 220 * weights / (weights.max() + 1e-8)
                    ax.scatter(centers[:, 0], centers[:, 1], s=sizes, c="gold", edgecolor="black", marker="*", zorder=5)
                for g, xy in data.items():
                    ax.plot(xy[:, 0], xy[:, 1], color=colors[g], lw=2.0, label=f"gamma={g}")
                    ax.scatter(xy[0, 0], xy[0, 1], color=colors[g], marker="o", s=22)
                    ax.scatter(xy[-1, 0], xy[-1, 1], color=colors[g], marker="x", s=45)
                ax.set_xlim(-0.2, 3.8)
                ax.set_ylim(-1.2, 3.8)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
            fig_path = fig_dir / f"{label}_gamma_shape_{dist_i:02d}_{Path(dfile).stem}.png"
            fig.savefig(fig_path, dpi=180)
            plt.close(fig)
            fig_paths.append(str(fig_path))

    summary = {
        "label": label,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "checkpoint_gamma_morphology_metrics": ckpt.get("gamma_morphology_metrics", {}),
        "data_dir": DATA_DIR,
        "n_selected_distributions": len(selected),
        "n_dpm_seeds": N_DPM_SEEDS,
        "inference_steps": INFERENCE_STEPS,
        "gammas": GAMMAS,
        "validation": validation,
        "direct_fixed_xt_pairwise_all_gamma_m": summarize(direct_pairwise),
        "dpm_pairwise_all_gamma_m": summarize(dpm_pairwise),
        "oracle_pairwise_all_gamma_m": summarize(oracle_pairwise),
        "direct_fixed_xt_low_high_0.01_0.2_m": summarize(direct_low_high),
        "dpm_low_high_0.01_0.2_m": summarize(dpm_low_high),
        "oracle_low_high_0.01_0.2_m": summarize(oracle_low_high),
        "direct_oracle_pairwise_correlation": corrcoef(direct_pairwise, oracle_pairwise),
        "dpm_oracle_pairwise_correlation": corrcoef(dpm_pairwise, oracle_pairwise),
        "direct_to_oracle_low_high_ratio": float(np.mean(direct_low_high) / max(np.mean(oracle_low_high), 1e-8)),
        "dpm_to_oracle_low_high_ratio": float(np.mean(dpm_low_high) / max(np.mean(oracle_low_high), 1e-8)),
        "start_artifacts": {
            "start_anchor_error_m": summarize([x["start_anchor_error"] for x in start_artifacts]),
            "first_step_m": summarize([x["first_step"] for x in start_artifacts]),
            "first5_max_step_m": summarize([x["first5_max_step"] for x in start_artifacts]),
            "first5_mean_curvature": summarize([x["first5_mean_curvature"] for x in start_artifacts]),
        },
        "figure_paths": fig_paths,
    }
    with open(out_dir / "direct_vs_dpm_strict_summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2)
    with open(out_dir / "direct_vs_dpm_strict_records.json", "w") as f:
        json.dump(jsonable(records), f, indent=2)
    print(f"[{label}] epoch:", summary["checkpoint_epoch"])
    print(f"[{label}] ckpt metrics:", jsonable(summary["checkpoint_metrics"]))
    print(f"[{label}] val direct/dpm:", summary["validation"]["direct_xy_mae_m"]["mean"], summary["validation"]["dpm_inference_xy_mae_m"]["mean"])
    print(f"[{label}] direct all:", summary["direct_fixed_xt_pairwise_all_gamma_m"])
    print(f"[{label}] dpm all:", summary["dpm_pairwise_all_gamma_m"])
    print(f"[{label}] oracle all:", summary["oracle_pairwise_all_gamma_m"])
    print(f"[{label}] low-high direct/dpm/oracle:", summary["direct_fixed_xt_low_high_0.01_0.2_m"], summary["dpm_low_high_0.01_0.2_m"], summary["oracle_low_high_0.01_0.2_m"])
    print(f"[{label}] corr direct/dpm:", summary["direct_oracle_pairwise_correlation"], summary["dpm_oracle_pairwise_correlation"])
    return summary


def complete_distribution_files(raw_ds):
    grouped = collections.defaultdict(set)
    for pair in raw_ds.data_pairs:
        traj = json.load(open(os.path.join(raw_ds.trajectories_dir, pair["trajectory_file"])))
        gamma = float(traj["gamma"])
        for target in GAMMAS:
            if abs(gamma - target) < 1e-6:
                grouped[pair["distribution_file"]].add(target)
    return sorted(d for d, gs in grouped.items() if all(g in gs for g in GAMMAS))


def split_distribution_sets(raw_ds, seed=42, val_split=0.2, test_split=0.1):
    all_dist_files = complete_distribution_files(raw_ds)
    if len(all_dist_files) < 3:
        all_dist_files = sorted(set(p["distribution_file"] for p in raw_ds.data_pairs))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_dist_files)
    n_test = max(1, int(np.floor(test_split * len(all_dist_files)))) if test_split > 0 else 0
    n_val = max(1, int(np.floor(val_split * len(all_dist_files)))) if val_split > 0 else 0
    test_set = set(all_dist_files[:n_test])
    val_set = set(all_dist_files[n_test:n_test + n_val])
    heldout = test_set | val_set
    train_set = set(p["distribution_file"] for p in raw_ds.data_pairs if p["distribution_file"] not in heldout)
    return train_set, val_set, test_set


class GroupedGammaBatchSampler:
    def __init__(self, raw_ds, allowed_dist_files, batch_dists=BATCH_DISTS, seed=42, drop_last=False):
        self.batch_dists = int(batch_dists)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        grouped = collections.defaultdict(dict)
        for idx, pair in enumerate(raw_ds.data_pairs):
            if pair["distribution_file"] not in allowed_dist_files:
                continue
            traj = json.load(open(os.path.join(raw_ds.trajectories_dir, pair["trajectory_file"])))
            gamma = float(traj["gamma"])
            for target in GAMMAS:
                if abs(gamma - target) < 1e-6:
                    grouped[pair["distribution_file"]][target] = idx
        self.groups = [
            [items[g] for g in GAMMAS]
            for _, items in sorted(grouped.items())
            if all(g in items for g in GAMMAS)
        ]
        if not self.groups:
            raise RuntimeError("No complete gamma groups for grouped training")

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = list(self.groups)
        rng.shuffle(groups)
        for start in range(0, len(groups), self.batch_dists):
            chunk = groups[start:start + self.batch_dists]
            if self.drop_last and len(chunk) < self.batch_dists:
                continue
            batch = [idx for group in chunk for idx in group]
            yield batch
        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.groups) // self.batch_dists
        return int(np.ceil(len(self.groups) / self.batch_dists))


def build_main_train_loader(raw_ds, standardizer, train_set, batch_size=64):
    ds = NormalizedDataset(raw_ds, standardizer)
    train_idx = [i for i, p in enumerate(raw_ds.data_pairs) if p["distribution_file"] in train_set]
    if not train_idx:
        raise RuntimeError("No training samples after group-level split")
    return DataLoader(ds, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=0, pin_memory=True), train_idx


def build_grouped_train_loader(raw_ds, standardizer, train_set):
    ds = NormalizedDataset(raw_ds, standardizer)
    sampler = GroupedGammaBatchSampler(raw_ds, train_set, batch_dists=BATCH_DISTS, seed=SEED)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, pin_memory=True), sampler


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def endpoint_loss(pred, target):
    return torch.mean((pred[:, 0, :2] - target[:, 0, :2]) ** 2) + torch.mean((pred[:, -1, :2] - target[:, -1, :2]) ** 2)


def denorm_traj_xy(x_norm, stats):
    t_mean = stats["trajectories"]["mean"].to(x_norm.device).view(1, 1, 4)
    t_std = stats["trajectories"]["std"].to(x_norm.device).view(1, 1, 4)
    safe = torch.where(t_std < 1e-6, torch.ones_like(t_std), t_std)
    return (x_norm * safe + t_mean)[..., :2]


def gamma_shape_loss(pred_norm, gt_norm, stats, groups_per_batch):
    if groups_per_batch <= 0 or pred_norm.shape[0] < 4:
        return pred_norm.new_tensor(0.0)
    B, T, D = pred_norm.shape
    usable = (B // 4) * 4
    pred = pred_norm[:usable].view(-1, 4, T, D)
    gt = gt_norm[:usable].view(-1, 4, T, D)
    t_mean = stats["trajectories"]["mean"].to(pred_norm.device).view(1, 1, 1, 4)
    t_std = stats["trajectories"]["std"].to(pred_norm.device).view(1, 1, 1, 4)
    safe = torch.where(t_std < 1e-6, torch.ones_like(t_std), t_std)
    pred_xy = (pred * safe + t_mean)[..., :2]
    gt_xy = (gt * safe + t_mean)[..., :2]
    losses = []
    for i, j in itertools.combinations(range(4), 2):
        d_pred = torch.linalg.norm(pred_xy[:, i] - pred_xy[:, j], dim=-1).mean(dim=-1)
        d_gt = torch.linalg.norm(gt_xy[:, i] - gt_xy[:, j], dim=-1).mean(dim=-1)
        losses.append(torch.abs(d_pred / torch.clamp(d_gt, min=1e-4) - 1.0))
    return torch.stack(losses, dim=0).mean()


def gmm_information_integral(xy, gaussian_packed, padding_mask):
    centers = gaussian_packed[..., 0:2]
    cov_flat = gaussian_packed[..., 2:6]
    weights = gaussian_packed[..., 6]
    valid = (~padding_mask).to(xy.dtype)
    sx2 = torch.clamp(cov_flat[..., 0].abs(), min=1e-3)
    sy2 = torch.clamp(cov_flat[..., 3].abs(), min=1e-3)
    diff = xy.unsqueeze(2) - centers.unsqueeze(1)
    exponent = -0.5 * ((diff[..., 0] ** 2) / sx2.unsqueeze(1) + (diff[..., 1] ** 2) / sy2.unsqueeze(1))
    density = torch.exp(torch.clamp(exponent, min=-50.0, max=0.0)) * weights.unsqueeze(1) * valid.unsqueeze(1)
    return density.sum(dim=-1).mean(dim=-1)


def gmm_hotspot_coverage_components(xy, gaussian_packed, padding_mask, temperature=16.0):
    centers = gaussian_packed[..., 0:2]
    cov_flat = gaussian_packed[..., 2:6]
    weights = gaussian_packed[..., 6]
    valid = (~padding_mask).to(xy.dtype)
    sx2 = torch.clamp(cov_flat[..., 0].abs() * HOTSPOT_COV_SCALE, min=1e-4)
    sy2 = torch.clamp(cov_flat[..., 3].abs() * HOTSPOT_COV_SCALE, min=1e-4)
    diff = xy.unsqueeze(2) - centers.unsqueeze(1)
    exponent = -0.5 * ((diff[..., 0] ** 2) / sx2.unsqueeze(1) + (diff[..., 1] ** 2) / sy2.unsqueeze(1))
    proximity = torch.exp(torch.clamp(exponent, min=-50.0, max=0.0))
    soft_select = torch.softmax(temperature * proximity, dim=1)
    soft_max = (soft_select * proximity).sum(dim=1)
    norm_weights = weights * valid
    norm_weights = norm_weights / torch.clamp(norm_weights.sum(dim=-1, keepdim=True), min=1e-8)
    return soft_max, norm_weights, valid


def gmm_hotspot_coverage_score(xy, gaussian_packed, padding_mask, temperature=12.0):
    soft_max, norm_weights, _ = gmm_hotspot_coverage_components(xy, gaussian_packed, padding_mask, temperature=temperature)
    return (soft_max * norm_weights).sum(dim=-1)


def ergodic_margin_loss(pred_norm, gt_norm, batch, stats, margin=ERGODIC_MARGIN):
    pred_xy = denorm_traj_xy(pred_norm, stats)
    gt_xy = denorm_traj_xy(gt_norm, stats)
    packed = batch["gaussian_packed"].to(pred_norm.device)
    mask = batch["gaussian_padding_mask"].to(pred_norm.device)
    pred_cov, norm_weights, valid = gmm_hotspot_coverage_components(pred_xy, packed, mask)
    oracle_cov, _, _ = gmm_hotspot_coverage_components(gt_xy, packed, mask)
    oracle_cov = oracle_cov.detach()
    pred_info = (pred_cov * norm_weights).sum(dim=-1)
    oracle_info = (oracle_cov * norm_weights).sum(dim=-1).detach()
    raw_deficit = torch.relu(oracle_cov - pred_cov - margin) * valid
    per_hotspot_deficit = raw_deficit * norm_weights
    loss = (((raw_deficit / max(margin, 1e-6)) ** 2) * norm_weights).sum(dim=-1).mean()
    return loss, {
        "pred_info_mean": float(pred_info.detach().mean().cpu()),
        "oracle_info_mean": float(oracle_info.detach().mean().cpu()),
        "info_deficit_mean": float(torch.relu(oracle_info - pred_info).detach().mean().cpu()),
        "hotspot_deficit_mean": float(per_hotspot_deficit.detach().sum(dim=-1).mean().cpu()),
        "ergodic_margin_loss": float(loss.detach().cpu()),
    }


def smoothness_loss(pred_norm, stats):
    xy = denorm_traj_xy(pred_norm, stats)
    accel = xy[:, 2:, :] - 2.0 * xy[:, 1:-1, :] + xy[:, :-2, :]
    return (accel ** 2).sum(dim=-1).mean()


def make_train_inputs(batch, device):
    traj = batch["trajectories"].to(device)
    return {
        "distribution": batch["distribution"].to(device),
        "robot_state": batch["robot_state"].to(device),
        "end_state": traj[:, -1, :],
        "gaussian_packed": batch["gaussian_packed"].to(device),
        "gaussian_padding_mask": batch["gaussian_padding_mask"].to(device),
        "gamma": batch["gamma"].to(device),
        "ergodic_metric": batch["ergodic_metric"].to(device),
        "trajectories": traj,
        "diffusion_time": torch.rand(traj.shape[0], device=device),
    }


@torch.no_grad()
def eval_direct_gamma_morphology(model, raw_ds, standardizer, by_dist, selected, normalizer, device, max_groups=12):
    model.eval()
    direct_pairwise, oracle_pairwise, direct_low_high, oracle_low_high = [], [], [], []
    for dist_i, dfile in enumerate(selected[:max_groups]):
        items = by_dist[dfile]
        base_idx = [x for x in items if abs(x[1] - 0.1) < 1e-6][0][0]
        base_sample = standardizer(raw_ds[base_idx])
        base_x0 = base_sample["trajectories"].unsqueeze(0).to(device)
        t = torch.full((1,), T_FIXED, device=device)
        torch.manual_seed(991 + dist_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(991 + dist_i)
        mean, std = model.sde.marginal_prob(base_x0, t)
        fixed_xt = mean + std * torch.randn_like(base_x0)
        pred_by_gamma, oracle_by_gamma = {}, {}
        for gamma in GAMMAS:
            item = [x for x in items if abs(x[1] - gamma) < 1e-6][0]
            oracle_by_gamma[gamma] = np.asarray(item[3]["states"], dtype=float)[:, :2]
            inp = make_model_input(raw_ds, standardizer, base_idx, gamma, device)
            enc = model.encoder(model._build_enc_in(inp))
            direct = model.decoder(
                fixed_xt,
                t,
                enc["encoding"],
                conditions={0: inp["robot_state"], model.decoder.T - 1: inp["end_state"]},
                gamma=inp["gamma"],
                context_mask=enc.get("encoding_mask"),
            )
            pred_by_gamma[gamma] = normalizer.denormalize_traj(direct).cpu().numpy()[0, :, :2]
        for ga, gb in itertools.combinations(GAMMAS, 2):
            pd = mean_point_dist(pred_by_gamma[ga], pred_by_gamma[gb])
            od = mean_point_dist(oracle_by_gamma[ga], oracle_by_gamma[gb])
            direct_pairwise.append(pd)
            oracle_pairwise.append(od)
            if abs(ga - 0.01) < 1e-9 and abs(gb - 0.2) < 1e-9:
                direct_low_high.append(pd)
                oracle_low_high.append(od)
    return {
        "direct_pairwise_error": float(np.mean(np.abs(np.asarray(direct_pairwise) - np.asarray(oracle_pairwise)))),
        "direct_pred_pairwise_mean": float(np.mean(direct_pairwise)),
        "oracle_pairwise_mean": float(np.mean(oracle_pairwise)),
        "direct_pred_low_high_mean": float(np.mean(direct_low_high)),
        "oracle_low_high_mean": float(np.mean(oracle_low_high)),
        "direct_low_high_error": float(np.mean(np.abs(np.asarray(direct_low_high) - np.asarray(oracle_low_high)))),
        "direct_oracle_pairwise_corr": corrcoef(direct_pairwise, oracle_pairwise),
    }


def save_checkpoint(path, model, opt, epoch, metrics, config_dict, gamma_metrics=None, score=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "metrics": metrics,
        "config": config_dict,
    }
    if gamma_metrics is not None:
        payload["gamma_morphology_metrics"] = gamma_metrics
    if score is not None:
        payload["mixed_score"] = float(score)
    torch.save(payload, path)


def main():
    import yaml
    try:
        import wandb
    except Exception:
        wandb = None

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = torch.load(os.path.join(STATS_DATA_DIR, "dataset_stats.pt"), map_location="cpu")
    raw_ds = ErgodicDataset(DATA_DIR, transform=None, max_trajectory_len=101)
    standardizer = Standardizer(stats)
    normalizer = EvalNormalizer(stats, device)
    train_set, val_set, test_set = split_distribution_sets(raw_ds, seed=42)
    train_loader, train_idx = build_main_train_loader(raw_ds, standardizer, train_set)
    shape_loader, shape_sampler = build_grouped_train_loader(raw_ds, standardizer, train_set)
    val_loader, val_idx = build_groupcomplete_val_loader(raw_ds, standardizer, val_set)

    pairs = json.load(open(os.path.join(DATA_DIR, "dataset_index.json")))
    gamma_counts = collections.Counter()
    by_dist = collections.defaultdict(list)
    for idx, pair in enumerate(pairs):
        traj = json.load(open(os.path.join(DATA_DIR, "trajectories", pair["trajectory_file"])))
        gamma = float(traj["gamma"])
        gamma_counts[f"{gamma:.6f}"] += 1
        by_dist[pair["distribution_file"]].append((idx, gamma, pair, traj))
    selected = select_distributions(pairs, by_dist)
    complete_dists = complete_distribution_files(raw_ds)

    assert len(pairs) == 1356, len(pairs)
    assert dict(sorted(gamma_counts.items())) == {"0.010000": 339, "0.030000": 339, "0.100000": 339, "0.200000": 339}
    assert len(complete_dists) >= 4, len(complete_dists)
    assert len(val_idx) > 0, len(val_idx)
    assert abs(float(stats["robot_state"]["mean"][0]) - 1.5) < 0.01, stats["robot_state"]["mean"].tolist()
    assert abs(float(stats["robot_state"]["mean"][1]) + 0.8) < 0.01, stats["robot_state"]["mean"].tolist()

    init_ckpt = torch.load(INIT_CKPT, map_location=device)
    cfg = to_ns(init_ckpt["config"])
    cfg.data.data_dir = DATA_DIR
    cfg.data_dir = DATA_DIR
    cfg.data.trajectory_len = 101
    cfg.trajectory_len = 101
    cfg.normalizer.robot_state.mean = stats["robot_state"]["mean"].to(device)
    cfg.normalizer.robot_state.std = stats["robot_state"]["std"].to(device)
    cfg.normalizer.trajectories = SimpleNamespace(
        mean=stats["trajectories"]["mean"].to(device),
        std=stats["trajectories"]["std"].to(device),
    )
    model = ErgodicDiffusionModel(cfg).to(device)
    model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)

    def _to_dict(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        return obj

    config_dict = _to_dict(cfg)
    config_dict["v7b_conditioning_fixes"] = [
        "start_and_endpoint_conditioning",
        "per_timestep_decoder_tokens",
        "block_level_gamma_film",
        "grouped_gamma_batches",
        "relative_oracle_pairwise_shape_loss",
        "gmm_information_margin_loss",
        "trajectory_smoothness_loss",
        "best_ade_best_morphology_best_mixed_best_ergodic_checkpoints",
    ]
    config_dict["shape_loss_weight"] = SHAPE_LOSS_WEIGHT
    config_dict["ergodic_loss_weight"] = ERGODIC_LOSS_WEIGHT
    config_dict["smooth_loss_weight"] = SMOOTH_LOSS_WEIGHT
    config_dict["endpoint_loss_weight"] = ENDPOINT_LOSS_WEIGHT
    config_dict["ergodic_margin"] = ERGODIC_MARGIN
    config_dict["hotspot_coverage_scale"] = HOTSPOT_COV_SCALE
    config_dict["mixed_pairwise_weight"] = MIXED_PAIRWISE_WEIGHT
    config_dict["mixed_ergodic_weight"] = MIXED_ERGODIC_WEIGHT
    config_dict["mixed_smooth_weight"] = MIXED_SMOOTH_WEIGHT
    config_dict["wandb_mode"] = os.environ.get("WANDB_MODE", "online")
    with open(OUT_DIR / "model_config.yaml", "w") as f:
        yaml.safe_dump(config_dict, f)

    run = None
    if wandb is not None:
        run = wandb.init(
            project="diffusion-ergodic",
            name="toes_continuation_balanced_v7b_ergodic_margin_strictbalanced",
            config=config_dict,
            reinit=True,
            mode=os.environ.get("WANDB_MODE", "online"),
        )

    print("[v7b preflight] data_dir:", DATA_DIR)
    print("[v7b preflight] stats_data_dir:", STATS_DATA_DIR)
    print("[v7b preflight] dataset size:", len(pairs))
    print("[v7b preflight] gamma counts:", dict(sorted(gamma_counts.items())))
    print("[v7b preflight] complete 4-gamma distributions:", len(complete_dists))
    print("[v7b preflight] train distributions:", len(train_set), "val distributions:", len(val_set), "test distributions:", len(test_set))
    print("[v7b preflight] train samples:", len(train_idx), "steps/epoch:", len(train_loader))
    print("[v7b preflight] complete shape-train groups:", len(shape_sampler.groups), "shape steps/epoch:", len(shape_loader))
    print("[v7b preflight] complete val trajectories:", len(val_idx), "complete val groups:", len(val_idx) // 4)
    print("[v7b preflight] split mode: distribution group split; gamma variants stay in the same split")
    print("[v7b preflight] stats robot_state mean:", stats["robot_state"]["mean"].tolist())
    print("[v7b preflight] init checkpoint:", INIT_CKPT)
    print("[v7b preflight] output dir:", OUT_DIR)
    print("[v7b preflight] wandb mode:", os.environ.get("WANDB_MODE", "online"))
    print("[v7b preflight] endpoint conditioning: start index 0, end index 100")
    print("[v7b preflight] loss weights:", {
        "shape": SHAPE_LOSS_WEIGHT,
        "ergodic": ERGODIC_LOSS_WEIGHT,
        "smooth": SMOOTH_LOSS_WEIGHT,
        "endpoint": ENDPOINT_LOSS_WEIGHT,
        "ergodic_margin": ERGODIC_MARGIN,
    })

    best_ade = float("inf")
    best_morph = float("inf")
    best_mixed = float("inf")
    best_ergodic = float("inf")
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_mse = total_shape = total_erg = total_smooth = total_endpoint = total_loss = 0.0
        last_info_stats = {}
        n_steps = 0
        shape_iter = iter(shape_loader)
        for batch in train_loader:
            X = make_train_inputs(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(X, training=True)
            pred = out["prediction"]
            mse = mse_loss(pred, X["trajectories"])
            endp = endpoint_loss(pred, X["trajectories"])
            erg, info_stats = ergodic_margin_loss(pred, X["trajectories"], batch, stats, margin=ERGODIC_MARGIN)
            smooth = smoothness_loss(pred, stats)

            try:
                shape_batch = next(shape_iter)
            except StopIteration:
                shape_iter = iter(shape_loader)
                shape_batch = next(shape_iter)
            X_shape = make_train_inputs(shape_batch, device)
            shape_out = model(X_shape, training=True)
            shape = gamma_shape_loss(shape_out["prediction"], X_shape["trajectories"], stats, shape_out["prediction"].shape[0] // 4)

            loss = (
                mse
                + SHAPE_LOSS_WEIGHT * shape
                + ERGODIC_LOSS_WEIGHT * erg
                + SMOOTH_LOSS_WEIGHT * smooth
                + ENDPOINT_LOSS_WEIGHT * endp
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_mse += float(mse.detach().cpu())
            total_shape += float(shape.detach().cpu())
            total_erg += float(erg.detach().cpu())
            total_smooth += float(smooth.detach().cpu())
            total_endpoint += float(endp.detach().cpu())
            total_loss += float(loss.detach().cpu())
            last_info_stats = info_stats
            n_steps += 1

        val = evaluate_validation(model, val_loader, normalizer, device, max_samples=512)
        gm = eval_direct_gamma_morphology(model, raw_ds, standardizer, by_dist, selected, normalizer, device, max_groups=12)
        ade = val["dpm_inference_xy_mae_m"]["mean"]
        direct_ade = val["direct_xy_mae_m"]["mean"]
        morph = gm["direct_pairwise_error"]
        erg_val = val["direct_info_deficit"]["mean"]
        smooth_val = val["direct_smoothness"]["mean"]
        mixed = (
            ade
            + MIXED_PAIRWISE_WEIGHT * morph
            + MIXED_ERGODIC_WEIGHT * erg_val
            + MIXED_SMOOTH_WEIGHT * smooth_val
        )
        print(
            f"[v7b epoch {epoch}] loss={total_loss/max(n_steps,1):.4f} mse={total_mse/max(n_steps,1):.4f} "
            f"shape={total_shape/max(n_steps,1):.4f} erg={total_erg/max(n_steps,1):.5f} "
            f"smooth={total_smooth/max(n_steps,1):.5f} direct_ade={direct_ade:.4f} dpm_ade={ade:.4f} "
            f"morph_err={morph:.4f} pred_pair={gm['direct_pred_pairwise_mean']:.4f} "
            f"oracle_pair={gm['oracle_pairwise_mean']:.4f} info_def={erg_val:.4f} mixed={mixed:.4f}"
        )

        if wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": total_loss / max(n_steps, 1),
                "train/mse": total_mse / max(n_steps, 1),
                "train/shape_loss": total_shape / max(n_steps, 1),
                "train/ergodic_margin_loss": total_erg / max(n_steps, 1),
                "train/smoothness_loss": total_smooth / max(n_steps, 1),
                "train/endpoint_loss": total_endpoint / max(n_steps, 1),
                "train/info_pred_mean": last_info_stats.get("pred_info_mean", 0.0),
                "train/info_oracle_mean": last_info_stats.get("oracle_info_mean", 0.0),
                "train/info_deficit_mean": last_info_stats.get("info_deficit_mean", 0.0),
                "val/direct_ade": direct_ade,
                "val/dpm_ade": ade,
                "val/dpm_fde": val["dpm_fde_m"]["mean"],
                "val/info_deficit": erg_val,
                "val/info_ratio": val["direct_info_ratio"]["mean"],
                "val/direct_smoothness": smooth_val,
                "val_gamma/direct_pairwise_error": morph,
                "val_gamma/direct_pred_pairwise_mean": gm["direct_pred_pairwise_mean"],
                "val_gamma/oracle_pairwise_mean": gm["oracle_pairwise_mean"],
                "val_gamma/direct_low_high": gm["direct_pred_low_high_mean"],
                "val_gamma/oracle_low_high": gm["oracle_low_high_mean"],
                "val_gamma/direct_oracle_pairwise_corr": gm["direct_oracle_pairwise_corr"],
                "val/mixed_score": mixed,
            })

        metrics = {
            "direct_ade": direct_ade,
            "dpm_ade": ade,
            "dpm_fde": val["dpm_fde_m"]["mean"],
            "validation": val,
            "train_loss": total_loss / max(n_steps, 1),
            "train_mse": total_mse / max(n_steps, 1),
            "train_shape_loss": total_shape / max(n_steps, 1),
            "train_ergodic_margin_loss": total_erg / max(n_steps, 1),
            "train_smoothness_loss": total_smooth / max(n_steps, 1),
            "train_endpoint_loss": total_endpoint / max(n_steps, 1),
            "val_info_deficit": erg_val,
            "val_info_ratio": val["direct_info_ratio"]["mean"],
            "val_direct_smoothness": smooth_val,
        }

        improved = False
        if ade < best_ade:
            best_ade = ade
            no_improve = 0
            improved = True
            save_checkpoint(OUT_DIR / "best_model.pth", model, opt, epoch, metrics, config_dict, gm, score=mixed)
            print(f"[v7b] saved best ADE: {best_ade:.4f}")
        if morph < best_morph:
            best_morph = morph
            save_checkpoint(OUT_DIR / "best_morphology_model.pth", model, opt, epoch, metrics, config_dict, gm, score=mixed)
            print(f"[v7b] saved best morphology: {best_morph:.4f}")
        if erg_val < best_ergodic:
            best_ergodic = erg_val
            save_checkpoint(OUT_DIR / "best_ergodic_model.pth", model, opt, epoch, metrics, config_dict, gm, score=mixed)
            print(f"[v7b] saved best ergodic: {best_ergodic:.4f}")
        if mixed < best_mixed:
            best_mixed = mixed
            no_improve = 0
            improved = True
            save_checkpoint(OUT_DIR / "best_mixed_model.pth", model, opt, epoch, metrics, config_dict, gm, score=mixed)
            print(f"[v7b] saved best mixed: {best_mixed:.4f}")
        if not improved:
            no_improve += 1
        if epoch % 20 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config_dict}, OUT_DIR / f"checkpoint_epoch_{epoch}.pth")
        if no_improve >= PATIENCE:
            print(f"[v7b] early stop at epoch {epoch}; best_ade={best_ade:.4f} best_morph={best_morph:.4f} best_ergodic={best_ergodic:.4f} best_mixed={best_mixed:.4f}")
            break

    eval_dir = OUT_DIR / "gamma_morphology_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    final_summary = {
        "data_dir": DATA_DIR,
        "output_dir": str(OUT_DIR),
        "best_ade": best_ade,
        "best_morphology_error": best_morph,
        "best_ergodic_error": best_ergodic,
        "best_mixed": best_mixed,
        "shape_loss_weight": SHAPE_LOSS_WEIGHT,
        "ergodic_loss_weight": ERGODIC_LOSS_WEIGHT,
        "smooth_loss_weight": SMOOTH_LOSS_WEIGHT,
        "endpoint_loss_weight": ENDPOINT_LOSS_WEIGHT,
        "ergodic_margin": ERGODIC_MARGIN,
        "mixed_pairwise_weight": MIXED_PAIRWISE_WEIGHT,
        "mixed_ergodic_weight": MIXED_ERGODIC_WEIGHT,
        "mixed_smooth_weight": MIXED_SMOOTH_WEIGHT,
        "checkpoints": {},
    }
    for label, ckpt in {
        "best_ade": OUT_DIR / "best_model.pth",
        "best_morphology": OUT_DIR / "best_morphology_model.pth",
        "best_ergodic": OUT_DIR / "best_ergodic_model.pth",
        "best_mixed": OUT_DIR / "best_mixed_model.pth",
    }.items():
        if ckpt.exists():
            final_summary["checkpoints"][label] = evaluate_checkpoint(label, str(ckpt), raw_ds, standardizer, normalizer, val_loader, by_dist, selected, stats, device)
    with open(OUT_DIR / "v7b_training_and_eval_summary.json", "w") as f:
        json.dump(jsonable(final_summary), f, indent=2)
    if wandb is not None:
        wandb.finish()
    print("[v7b] summary:", OUT_DIR / "v7b_training_and_eval_summary.json")


if __name__ == "__main__":
    main()
