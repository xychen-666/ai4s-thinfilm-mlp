"""All article-ready figures generated from this experiment's actual arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .config import Config


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [
        ("Four-layer film", "thickness d1...d4"),
        ("TMM physics", "5000 spectra"),
        ("MLP training", "4 data sizes"),
        ("Screening", "10000 candidates"),
        ("TMM check", "Top 10 to Top 5"),
    ]
    for i, (title, subtitle) in enumerate(stages):
        x = 0.15 + 2.45 * i
        ax.add_patch(
            Rectangle((x, 0.85), 2.05, 1.3, facecolor="#e7f0f7", edgecolor="#245477", lw=1.6)
        )
        ax.text(x + 1.025, 1.65, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + 1.025, 1.27, subtitle, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.42, 1.5), xytext=(x + 2.08, 1.5), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#245477"})
    ax.set_title("Figure 1. AI4S thin-film study workflow", fontsize=13, pad=5)
    _save(fig, path)


def plot_physics_data(path: Path, config: Config, spectra: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [0.8, 1.4]})
    colors = ["#d8edf8", "#799bc7", "#b4cfec", "#5b83bc", "#e5e5e5", "#c7e8ee"]
    labels = ["Air (n = 1.00)", "H (n = 2.30)", "L (n = 1.45)", "H (n = 2.30)", "L (n = 1.45)", "Glass (n = 1.52)"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        y = 5 - i
        ax1.add_patch(Rectangle((0, y), 3.5, 1, facecolor=color, edgecolor="white"))
        ax1.text(1.75, y + 0.5, label, ha="center", va="center", fontsize=10)
    ax1.annotate("incident light", xy=(0.55, 4.5), xytext=(0.55, 5.6), ha="center", fontsize=9, arrowprops={"arrowstyle": "->"})
    ax1.set_xlim(-0.15, 3.65)
    ax1.set_ylim(-0.1, 6.4)
    ax1.axis("off")
    ax1.set_title("(a) Four-layer optical stack")
    for i, spectrum in enumerate(spectra[:5]):
        ax2.plot(config.wavelengths_nm, spectrum, lw=1.7, label=f"Sample {i + 1}")
    ax2.axvline(config.target_wavelength_nm, color="black", ls="--", lw=1, alpha=0.65)
    ax2.set(xlabel="Wavelength (nm)", ylabel="Reflectance", xlim=(400, 800), ylim=(0, 1), title="(b) Example TMM-generated data")
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    widths = [4, 128, 128, 64, 41]
    labels = ["Thickness\n4 inputs", "ReLU\n128", "ReLU\n128", "ReLU\n64", "Spectrum\n41 outputs"]
    for i, (width, label) in enumerate(zip(widths, labels)):
        x = 0.22 + 2.0 * i
        ax.add_patch(Rectangle((x, 0.65), 1.55, 1.55, facecolor="#ecf4fa" if i in (0, 4) else "#b8d6e9", edgecolor="#245477", lw=1.5))
        ax.text(x + 0.775, 1.45, label, ha="center", va="center", fontsize=10)
        if i < 4:
            ax.annotate("", xy=(x + 1.95, 1.43), xytext=(x + 1.57, 1.43), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(5, 0.3, "Input scaling: 40-180 nm to [-1, 1]     Loss: mean squared error", ha="center", fontsize=9)
    ax.set_title("Figure 3. MLP surrogate model architecture", fontsize=13)
    _save(fig, path)


def plot_losses(path: Path, history: dict[str, list[float]], best_epoch: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(history["train_mse"]) + 1)
    ax.plot(epochs, history["train_mse"], label="Train MSE", lw=1.8)
    ax.plot(epochs, history["validation_mse"], label="Validation MSE", lw=1.8)
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Selected epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Figure 4. MLP training and validation losses (n = 4000)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_test_spectra(path: Path, config: Config, actual: np.ndarray, predicted: np.ndarray, indices: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, idx in zip(axes, indices):
        ax.plot(config.wavelengths_nm, actual[idx], label="TMM", color="#155e75", lw=2)
        ax.plot(config.wavelengths_nm, predicted[idx], label="MLP", color="#c2410c", lw=1.6, ls="--")
        ax.set(xlabel="Wavelength (nm)", title=f"Test sample {idx} | MAE {np.mean(np.abs(actual[idx]-predicted[idx])):.4f}", xlim=(400, 800), ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Reflectance")
    axes[-1].legend()
    fig.suptitle("Figure 5. TMM and MLP spectra on three representative test samples", fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def plot_training_sizes(path: Path, sizes: list[int], test_mse: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    ax.plot(sizes, test_mse, marker="o", lw=2, color="#245477")
    for position, (x, y) in enumerate(zip(sizes, test_mse)):
        align = "left" if position == 0 else "right" if position == len(sizes) - 1 else "center"
        offset = 5 if position == 0 else -5 if position == len(sizes) - 1 else 0
        ax.annotate(f"{y:.3g}", (x, y), xytext=(offset, 8), textcoords="offset points", ha=align, fontsize=9)
    ax.set(xlabel="Training samples", ylabel="Held-out test MSE", title="Figure 6. Effect of training-set size")
    ax.set_xticks(sizes)
    ax.set_xlim(min(sizes) - 160, max(sizes) + 160)
    ax.set_ylim(0, max(test_mse) * 1.18)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def plot_design(path: Path, config: Config, predicted: np.ndarray, verified: np.ndarray, target_pred: float, target_true: float) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(config.wavelengths_nm, verified, label="TMM verified", color="#155e75", lw=2.2)
    ax.plot(config.wavelengths_nm, predicted, label="MLP predicted", color="#c2410c", lw=1.8, ls="--")
    ax.axvline(config.target_wavelength_nm, color="black", lw=1, ls=":", label=f"Target: {config.target_wavelength_nm} nm")
    ax.scatter([config.target_wavelength_nm] * 2, [target_true, target_pred], color=["#155e75", "#c2410c"], zorder=3)
    ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", ylim=(0, 1), xlim=(400, 800), title="Figure 7. Selected high-reflectance design")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def plot_failure(path: Path, config: Config, actual: np.ndarray, predicted: np.ndarray, index: int) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(config.wavelengths_nm, actual, label="TMM", color="#155e75", lw=2)
    ax.plot(config.wavelengths_nm, predicted, label="MLP", color="#c2410c", lw=1.8, ls="--")
    ax.fill_between(config.wavelengths_nm, actual, predicted, alpha=0.17, color="#c2410c")
    ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", ylim=(0, 1), xlim=(400, 800), title=f"Figure 8. Largest-error test case (sample {index})")
    ax.text(0.02, 0.03, f"Spectrum MAE = {np.mean(np.abs(actual-predicted)):.4f}", transform=ax.transAxes, fontsize=10)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def plot_screening_audit(path: Path, predicted: np.ndarray, verified: np.ndarray, final_positions: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.1))
    ax.scatter(predicted, verified, s=70, color="#5888ac", label="MLP Top 10")
    ax.scatter(predicted[final_positions], verified[final_positions], s=95, color="#c2410c", label="TMM Top 5")
    bounds = (min(predicted.min(), verified.min()) - 0.02, max(predicted.max(), verified.max()) + 0.02)
    ax.plot(bounds, bounds, ls="--", color="gray", lw=1, label="Perfect prediction")
    ax.set(xlabel="MLP-predicted target R", ylabel="TMM-verified target R", xlim=bounds, ylim=bounds, title="Screening audit at the target wavelength")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
