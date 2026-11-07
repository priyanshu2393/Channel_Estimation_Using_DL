"""Classical pilot interpolation and train-only genie 2D-LMMSE baselines."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import solve
import torch

from attrnet_ofdm.data import SavedOFDMDataset
from pilot_design.pilot_utils import pilot_coordinates


def manifest_coords(data: SavedOFDMDataset) -> np.ndarray:
    pc = data.manifest["pilot_configuration"]
    if "frequency_indices" in pc:
        rows = np.asarray(pc["frequency_indices"], dtype=int)
        cols = np.asarray(pc["time_indices"], dtype=int)
    else:
        rows = np.asarray(pc["frequency_indices_matlab_one_based"], dtype=int) - 1
        cols = np.asarray(pc["time_indices_matlab_one_based"], dtype=int) - 1
    return np.asarray([(f, t) for f in rows for t in cols], dtype=np.int64)


def complex_grid(ri: np.ndarray) -> np.ndarray:
    return ri[..., 0, :, :] + 1j * ri[..., 1, :, :]


def separable_covariance(train_y: np.ndarray) -> np.ndarray:
    """Estimate a separable covariance from clean training channels only."""
    channels = complex_grid(np.asarray(train_y)).astype(np.complex128, copy=False)
    rf = np.einsum("sft,sgt->fg", channels, channels.conj(), optimize=True) / (len(channels) * 14)
    rt = np.einsum("sft,sfu->tu", channels, channels.conj(), optimize=True) / len(channels)
    # The channel covariance is represented in C-order (f,t). Normalize the
    # frequency factor to unit trace; the time factor then carries total power.
    rf /= max(np.trace(rf).real, 1e-12)
    return np.kron(rf, rt)


def full_covariance(train_y: np.ndarray, separable: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    flat = complex_grid(np.asarray(train_y)).reshape(len(train_y), -1).astype(np.complex128, copy=False)
    cov = flat.T @ flat.conj() / len(flat)
    return (1.0 - shrinkage) * cov + shrinkage * separable


def interp_one(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rows, cols = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    if len(rows) * len(cols) != len(coords):
        fq, tq = np.meshgrid(np.arange(72), np.arange(14), indexing="ij")
        points = np.stack((fq.ravel(), tq.ravel()), axis=-1)
        out = griddata(coords, values, points, method="linear")
        missing = ~np.isfinite(out)
        out[missing] = griddata(coords, values, points[missing], method="nearest")
        return out.reshape(72,14)
    grid = values.reshape(len(rows), len(cols))
    fq, tq = np.meshgrid(np.arange(72), np.arange(14), indexing="ij")
    fi = np.clip(np.searchsorted(rows, fq)-1, 0, len(rows)-2)
    ti = np.clip(np.searchsorted(cols, tq)-1, 0, len(cols)-2)
    f0, f1 = rows[fi], rows[fi+1]; t0, t1 = cols[ti], cols[ti+1]
    af = ((fq-f0)/np.maximum(f1-f0,1)).astype(np.float64)
    at = ((tq-t0)/np.maximum(t1-t0,1)).astype(np.float64)
    return ((1-af)*(1-at)*grid[fi,ti] + af*(1-at)*grid[fi+1,ti] + (1-af)*at*grid[fi,ti+1] + af*at*grid[fi+1,ti+1])


def lmmse_one(cov: np.ndarray, coords: np.ndarray, values: np.ndarray, sigma2: float) -> np.ndarray:
    pilot_idx = coords[:, 0] * 14 + coords[:, 1]
    cross = cov[:, pilot_idx]
    pp = cov[np.ix_(pilot_idx, pilot_idx)] + sigma2 * np.eye(len(pilot_idx))
    return (cross @ solve(pp, values, assume_a="her", check_finite=False)).reshape(72, 14)


def lmmse_filter(cov: np.ndarray, coords: np.ndarray, sigma2: float) -> np.ndarray:
    pilot_idx = coords[:, 0] * 14 + coords[:, 1]
    cross = cov[:, pilot_idx]
    pp = cov[np.ix_(pilot_idx, pilot_idx)] + sigma2 * np.eye(len(pilot_idx))
    return solve(pp, cross.conj().T, assume_a="her", check_finite=False).conj().T


def score(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = np.abs(pred - target) ** 2
    power = np.abs(target) ** 2
    return float(error.mean()), float(error.sum() / max(power.sum(), 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/classical_baselines.csv"))
    p.add_argument("--shrinkage", type=float, default=0.1)
    args = p.parse_args()
    train = SavedOFDMDataset(args.data_dir, "train")
    test = SavedOFDMDataset(args.data_dir, "test")
    coords = manifest_coords(train)
    cov_sep = separable_covariance(train.y)
    cov_full = full_covariance(train.y, cov_sep, args.shrinkage)
    train_ls = train.x.transpose(0, 2, 3, 1).reshape(len(train), -1, 2)
    train_true = train.y.transpose(0, 2, 3, 1)[:, coords[:, 0], coords[:, 1]]
    noise_by_snr = {}
    for snr in np.unique(train.snr_db):
        mask = train.snr_db == snr
        diff = train_ls[mask] - train_true[mask]
        noise_by_snr[float(snr)] = float(np.mean(np.sum(diff**2, axis=-1)))
    rows = []
    for snr in np.unique(test.snr_db):
        ix = np.flatnonzero(test.snr_db == snr)
        filt_sep=lmmse_filter(cov_sep,coords,noise_by_snr[float(snr)])
        filt_full=lmmse_filter(cov_full,coords,noise_by_snr[float(snr)])
        predictions = {"LS+linear_interp": [], "genie_separable_LMMSE": [], "genie_full_LMMSE": []}
        for i in ix:
            ls = test.x[i].transpose(1, 2, 0).reshape(-1, 2)
            vals = ls[:, 0] + 1j*ls[:, 1]
            predictions["LS+linear_interp"].append(interp_one(vals, coords))
            predictions["genie_separable_LMMSE"].append((filt_sep @ vals).reshape(72,14))
            predictions["genie_full_LMMSE"].append((filt_full @ vals).reshape(72,14))
        target = complex_grid(np.asarray(test.y[ix]))
        for name, pred in predictions.items():
            mse, nmse = score(np.asarray(pred), target)
            rows.append({"model": name, "snr_db": float(snr), "mse": mse, "nmse": nmse, "nmse_db": 10*np.log10(max(nmse, 1e-30)), "samples": len(ix), "shrinkage": args.shrinkage if "full" in name else "", "noise_variance_source": "training samples at matching SNR"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {args.output}; train-only genie covariance, shrinkage={args.shrinkage}; noise variances={noise_by_snr}")


if __name__ == "__main__":
    main()
