"""Evaluate a trained AttRNet checkpoint on a fixed simulated data split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File-only rendering; avoids a GUI backend on Windows/headless runs.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.model import AttRNet


@torch.no_grad()
def mse_nmse(model: AttRNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return complex-channel MSE and pooled energy-ratio NMSE."""
    model.eval()
    mse_total, error_energy, target_energy, count = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        error = (model(x) - y).square().sum(dim=1)
        target = y.square().sum(dim=1)
        mse_total += error.mean().item() * x.size(0)
        error_energy += error.sum().item()
        target_energy += target.sum().item()
        count += x.size(0)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return mse_total / count, error_energy / max(target_energy, 1e-30)


def update_comparison_csv(path: Path, pilots: int, model_name: str, checkpoint: Path, values: dict[float, float]) -> None:
    """Insert one real test result per SNR into the paper-comparison CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Comparison CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []
    updated = 0
    for row in rows:
        snr = float(row["snr_db"])
        if int(row["pilot_count"]) == pilots and row["model"] == model_name and snr in values:
            simulated = values[snr]
            paper = float(row["paper_mse_digitized_approx"])
            row["simulation_mse"] = f"{simulated:.8g}"
            row["relative_difference_percent"] = f"{100 * (simulated - paper) / paper:.4f}"
            row["simulation_checkpoint"] = str(checkpoint)
            row["simulation_status"] = "complete"
            updated += 1
    if updated != len(values):
        raise ValueError(f"Updated {updated} rows but expected {len(values)}; check model/pilot/SNR values in {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paper_reference(path: Path, pilots: int, model_name: str) -> dict[float, float]:
    """Read digitized paper reference points for one model configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Paper reference CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        result = {
            float(row["snr_db"]): float(row["paper_mse_digitized_approx"])
            for row in rows
            if int(row["pilot_count"]) == pilots and row["model"] == model_name
        }
    return result


def save_comparison_plot(path: Path, pilots: int, model_name: str, paper: dict[float, float], simulated: dict[float, float]) -> None:
    """Save a paper-style logarithmic MSE-versus-SNR comparison plot."""
    snrs = sorted(set(paper) & set(simulated)) if paper else sorted(simulated)
    if not snrs:
        raise ValueError("No shared SNR points between paper reference and simulated result")
    color = "tab:red" if model_name == "AttRNet-FC" else "tab:cyan"
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    if paper:
        ax.semilogy(snrs, [paper[s] for s in snrs], "--", color=color, marker="*", markersize=9, linewidth=2, label="Paper (digitized)")
    ax.semilogy(snrs, [simulated[s] for s in snrs], "-", color="black", marker="o", markerfacecolor="white", markersize=7, linewidth=2, label="Our simulation")
    ax.set_xticks(snrs)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    title = f"{model_name}: MSE comparison with {pilots} pilots" if paper else f"{model_name}: MSE versus SNR with {pilots} pilots (simulation only)"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.65)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None, help="Dataset created by simulate_data.py. Defaults from checkpoint pilot count.")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reference-csv", type=Path, default=Path("paper_vs_simulation_metrics.csv"), help="Digitized paper metrics used in the plot.")
    p.add_argument("--comparison-csv", type=Path, default=None, help="Optional paper_vs_simulation_metrics.csv to update with this test result.")
    p.add_argument("--plot", type=Path, default=None, help="PNG output path. Defaults beside the checkpoint.")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pilots = checkpoint["pilots"]
    data_dir = args.data_dir or Path("data") / f"epa7_{pilots}pilots"
    dataset = SavedOFDMDataset(data_dir, args.split)
    if dataset.pilots != pilots:
        p.error(f"Checkpoint expects {pilots} pilots but dataset manifest contains {dataset.pilots}.")
    model = AttRNet(checkpoint["model"], checkpoint["features"], output_kernel=tuple(checkpoint["output_kernel"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    total_mse, total_nmse = mse_nmse(model, DataLoader(dataset, batch_size=args.batch_size), device)
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {data_dir} ({args.split}, {len(dataset):,} samples)")
    print(f"overall_mse: {total_mse:.8g}")
    print(f"overall_nmse: {total_nmse:.8g}")
    print(f"overall_nmse_db: {10 * np.log10(max(total_nmse, 1e-30)):.6g}")
    print("snr_db,mse,nmse,nmse_db,samples")
    snrs = np.asarray(dataset.snr_db)
    per_snr: dict[float, float] = {}
    for snr in np.unique(snrs):
        indices = np.flatnonzero(snrs == snr).tolist()
        value, nmse = mse_nmse(model, DataLoader(Subset(dataset, indices), batch_size=args.batch_size), device)
        per_snr[float(snr)] = value
        print(f"{snr:g},{value:.8g},{nmse:.8g},{10 * np.log10(max(nmse, 1e-30)):.6g},{len(indices)}")
    if args.comparison_csv is not None:
        display_name = {"fc": "AttRNet-FC", "conv": "AttRNet-Conv"}[checkpoint["model"]]
        update_comparison_csv(args.comparison_csv, pilots, display_name, args.checkpoint, per_snr)
        print(f"comparison CSV updated: {args.comparison_csv}")
    else:
        display_name = {"fc": "AttRNet-FC", "conv": "AttRNet-Conv"}[checkpoint["model"]]
    plot_path = args.plot or args.checkpoint.parent / f"comparison_{checkpoint['model']}_{pilots}pilots.png"
    save_comparison_plot(plot_path, pilots, display_name, paper_reference(args.reference_csv, pilots, display_name), per_snr)
    print(f"comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
