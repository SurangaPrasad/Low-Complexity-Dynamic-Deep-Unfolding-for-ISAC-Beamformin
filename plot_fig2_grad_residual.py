"""
Fig. 2 — Analog gradient residual vs outer-layer index (motivation for C1/C2).

Records r_{i,0} = ||∇_F g||_F at the first inner step of each outer layer for
Fixed-UPGA with J = 5 and J = 10, then saves a journal-quality EPS figure.
"""

from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from PGA_models import PGA_Unfold_JX
from system_config import (
    OMEGA,
    Nt,
    device,
    directory_result,
    init_scheme,
    load_model_file,
    n_iter_inner_J5,
    n_iter_inner_J10,
    n_iter_outer,
    snr,
    snr_dB,
    step_size_UPGA_J5,
    step_size_UPGA_J10,
    test_size,
    theta_desire,
    xi_0,
    R_N_inv,
)
from utility import get_data_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FIG_STEM = "fig2_grad_residual"
CACHE_NAME = f"{FIG_STEM}_{Nt}_{OMEGA}.mat"


def _configure_matplotlib():
    # Font sizes are chosen for IEEE single-column width after
    # \includegraphics[width=\columnwidth] / \linewidth.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.usetex": False,
    })


def _load_or_init_model(stem, step_size):
    model = PGA_Unfold_JX(step_size.clone().detach())
    path = load_model_file(stem)
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded trained step sizes from {path}")
    else:
        print(f"No trained model at {path}; using initial step sizes.")
    model.to(device)
    model.eval()
    return model


def collect_residual(model, H, n_inner, label):
    print(f"Collecting residual for {label} (J = {n_inner})...")
    with torch.no_grad():
        model.execute_PGA(
            H, xi_0, theta_desire, R_N_inv, snr, n_iter_outer, n_inner, track_metrics=True
        )
    residual = np.asarray(model.residual_history, dtype=np.float64)
    if residual.size != n_iter_outer:
        raise RuntimeError(
            f"{label}: expected {n_iter_outer} residual samples, got {residual.size}"
        )
    return residual


def run_and_cache(cache_path, max_samples=None):
    from system_config import data_source

    H_train, _ = get_data_tensor(data_source)
    n_use = test_size if max_samples is None else min(test_size, max_samples)
    H = H_train[:, :n_use, :, :].contiguous()

    model_j5 = _load_or_init_model("UPGA_J5", step_size_UPGA_J5)
    model_j10 = _load_or_init_model("UPGA_J10", step_size_UPGA_J10)

    residual_j5 = collect_residual(model_j5, H, n_iter_inner_J5, "Fixed-UPGA (J=5)")
    residual_j10 = collect_residual(model_j10, H, n_iter_inner_J10, "Fixed-UPGA (J=10)")

    outer_idx = np.arange(n_iter_outer, dtype=np.int64)
    payload = {
        "outer_idx": outer_idx,
        "residual_j5": residual_j5,
        "residual_j10": residual_j10,
        "Nt": Nt,
        "OMEGA": OMEGA,
        "snr_dB": snr_dB,
        "n_iter_outer": n_iter_outer,
        "init_scheme": init_scheme,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    scipy.io.savemat(cache_path, payload)
    print(f"Saved residual cache to {cache_path}")
    return payload


def load_cache(cache_path):
    data = scipy.io.loadmat(cache_path, simplify_cells=True)
    return {
        "outer_idx": np.asarray(data["outer_idx"]).ravel(),
        "residual_j5": np.asarray(data["residual_j5"]).ravel(),
        "residual_j10": np.asarray(data["residual_j10"]).ravel(),
    }


def plot_residual(outer_idx, residual_j5, residual_j10, out_dir):
    _configure_matplotlib()
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.35))

    markevery = max(1, len(outer_idx) // 12)
    ax.plot(
        outer_idx, residual_j5,
        linestyle="-", marker="o", markevery=markevery,
        color="#1f4e79", markerfacecolor="white", markeredgewidth=0.9,
        label=r"Fixed-UPGA, $J=5$",
    )
    ax.plot(
        outer_idx, residual_j10,
        linestyle="-", marker="D", markevery=markevery,
        color="#c45c26", markerfacecolor="white", markeredgewidth=0.9,
        label=r"Fixed-UPGA, $J=10$",
    )

    ax.set_xlabel(r"Outer-layer index $i$")
    ax.set_ylabel(r"Analog gradient residual $\|\nabla_{\mathbf{F}} g\|_F$")
    ax.set_xlim(float(outer_idx[0]), float(outer_idx[-1]))

    y_max = float(max(np.max(residual_j5), np.max(residual_j10)))
    ax.set_ylim(0.0, y_max * 1.08)

    ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="0.75")
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.5",
        framealpha=1.0,
        borderpad=0.3,
        handlelength=1.6,
        labelspacing=0.25,
    )

    fig.tight_layout(pad=0.15)

    eps_path = os.path.join(out_dir, f"{FIG_STEM}.eps")
    png_path = os.path.join(out_dir, f"{FIG_STEM}.png")
    pdf_path = os.path.join(out_dir, f"{FIG_STEM}.pdf")

    fig.savefig(eps_path, format="eps")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)

    print(f"Saved {eps_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return eps_path


def main():
    parser = argparse.ArgumentParser(description="Plot Fig. 2: analog gradient residual.")
    parser.add_argument("--load-cache", action="store_true", help="Plot from cached .mat only.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit batch size for a quick run.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: sim_results/<system>/ and journal_paper/figures).",
    )
    args = parser.parse_args()

    cache_path = os.path.join(directory_result, CACHE_NAME)
    out_dirs = []
    if args.out_dir:
        out_dirs = [args.out_dir]
    else:
        out_dirs = [
            directory_result,
            os.path.join("journal_paper", "figures"),
        ]

    if args.load_cache:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        data = load_cache(cache_path)
    else:
        data = run_and_cache(cache_path, max_samples=args.max_samples)

    for out_dir in out_dirs:
        plot_residual(data["outer_idx"], data["residual_j5"], data["residual_j10"], out_dir)


if __name__ == "__main__":
    main()
