"""Chinese publication figures for the inverse-design experiment.

All numerical marks are drawn from the supplied experiment arrays.  The
workflow alone contains explicitly schematic thumbnails.  This module does
not train a model, alter predictions, or run a new optical calculation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .config import Config


BLUE = "#3879AB"
APRICOT = "#CB895E"
GREEN = "#7D9B87"
INK = "#253340"
MUTED = "#697985"
LIGHT = "#D5E0E7"
PALE_BLUE = "#E4EFF6"
PALE_APRICOT = "#F7E8DC"
PALE_GREEN = "#EAF1EB"

FIGURE_FILENAMES = (
    "figure_1_inverse_workflow.png",
    "figure_2_inverse_losses.png",
    "figure_3_inverse_thickness.png",
    "figure_4_inverse_spectra.png",
    "figure_5_inverse_training_sizes.png",
    "figure_6_inverse_distribution_refinement.png",
    "figure_7_inverse_failure.png",
)

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _save(fig: plt.Figure, path: Path) -> None:
    """Deliver a 600 dpi bitmap plus editable vector versions."""
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def _axis(ax: plt.Axes, panel: str | None = None, title: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left", pad=10)
    if panel:
        ax.text(-0.14, 1.07, panel, transform=ax.transAxes, fontsize=12,
                weight="bold", va="bottom", ha="left")


def _safe_log_values(values: np.ndarray) -> np.ndarray:
    """Use the numerical floor only for zero loss on a logarithmic axis."""
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("A loss or absolute error cannot be negative")
    return np.maximum(values, np.finfo(float).tiny)


def _validate(config: Config, arrays: dict[str, np.ndarray], summary: dict) -> None:
    samples = len(np.asarray(arrays["true_thickness"]))
    if samples < 1:
        raise ValueError("The inverse test set is empty")
    required_thickness = ["true_thickness", "supervised_raw_thickness",
                          "supervised_thickness", "physics_thickness",
                          "refined_thickness", "nearest_thickness"]
    required_spectra = ["target_spectra", "supervised_spectra", "physics_spectra",
                       "refined_spectra", "nearest_spectra"]
    for key in required_thickness:
        value = np.asarray(arrays[key])
        if value.shape != (samples, 4) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be a finite ({samples}, 4) array")
    for key in required_spectra:
        value = np.asarray(arrays[key])
        if value.shape != (samples, len(config.wavelengths_nm)) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} has invalid spectral dimensions or values")
    sizes = np.asarray(arrays["training_sizes"])
    for key in ["size_thickness_mae_nm", "size_spectral_mae"]:
        if np.asarray(arrays[key]).shape != sizes.shape:
            raise ValueError(f"{key} must match training_sizes")
    for method in ["supervised", "physics"]:
        train = np.asarray(arrays[f"loss_{method}_train"])
        validation = np.asarray(arrays[f"loss_{method}_validation"])
        if train.ndim != 1 or train.shape != validation.shape:
            raise ValueError(f"Invalid {method} loss history")
        if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
            raise ValueError(f"Non-finite {method} loss history")
    refinement = np.asarray(arrays["refinement_best_mse_history"])
    if refinement.ndim != 2 or refinement.shape[1] != samples or not np.all(np.isfinite(refinement)):
        raise ValueError("refinement_best_mse_history must have (steps + 1, samples) shape")
    if refinement.shape[0] != int(summary["refinement_steps"]) + 1:
        raise ValueError("Refinement history does not match refinement_steps")
    indices = list(summary["representative_test_indices"])
    if len(indices) != 3 or any(not 0 <= int(i) < samples for i in indices):
        raise ValueError("Exactly three valid representative indices are required")
    if not 0 <= int(summary["failure_test_index"]) < samples:
        raise ValueError("Invalid failure_test_index")


def _workflow(path: Path, config: Config, arrays: dict[str, np.ndarray], summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.85))
    ax.set(xlim=(0, 10), ylim=(0, 2.25))
    ax.axis("off")
    xs = [0.72, 2.83, 4.94, 7.05, 9.16]
    titles = ["目标光谱", "逆向 MLP", "四层膜厚", "TMM 回算", "重建光谱"]
    notes = ["41 个反射率点", "41 → 128 → 128 → 64 → 4",
             "40–180 nm", "H / L / H / L", "与目标逐点比较"]
    for i, (x, title, note) in enumerate(zip(xs, titles, notes)):
        ax.text(x, 0.62, title, ha="center", weight="bold", fontsize=10)
        ax.text(x, 0.32, note, ha="center", color=MUTED, fontsize=6.9 if i == 1 else 8)
        if i < 4:
            ax.annotate("", xy=(x + 1.53, 1.34), xytext=(x + 0.53, 1.34),
                        arrowprops={"arrowstyle": "-|>", "color": LIGHT, "lw": 1.2,
                                    "mutation_scale": 10})
    # The first/last thumbnails are genuine spectra, rescaled solely to fit
    # a schematic workflow (they have no quantitative axes).
    idx = int(summary["representative_test_indices"][1])
    local_x = np.linspace(-0.41, 0.41, len(config.wavelengths_nm))
    for x, key, color in [(xs[0], "target_spectra", INK), (xs[4], "refined_spectra", BLUE)]:
        y = np.asarray(arrays[key][idx])
        ax.plot(x + local_x, 1.08 + 0.49 * y, color=color, lw=1.6, clip_on=False)
    # A sparse icon indicates a fully connected model without fabricating
    # a rendered node for every hidden unit.
    nodes = [np.array([1.10, 1.34, 1.58])] * 3
    columns = [xs[1] - 0.30, xs[1], xs[1] + 0.30]
    for j in range(2):
        for y1 in nodes[j]:
            for y2 in nodes[j + 1]:
                ax.plot([columns[j], columns[j + 1]], [y1, y2], color=LIGHT, lw=0.55, zorder=1)
    for x, ys in zip(columns, nodes):
        ax.scatter([x] * len(ys), ys, s=12, color=GREEN, zorder=2)
    for i in range(4):
        ax.add_patch(Rectangle((xs[2] - 0.35, 1.06 + i * 0.15), 0.70, 0.12,
                               facecolor=BLUE if i % 2 == 0 else APRICOT, edgecolor="none"))
    ax.text(xs[3], 1.35, "$M_1M_2M_3M_4$", ha="center", va="center", fontsize=12)
    ax.text(5, 0.00, "直接逆向训练  ·  可微物理约束  ·  膜厚局部优化",
            ha="center", fontsize=8, color=MUTED)
    _save(fig, path)


def _losses(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    descriptions = [
        ("supervised", "归一化膜厚 MSE", "直接逆向 MLP"),
        ("physics", "反射率 MSE", "物理约束 MLP"),
    ]
    for ax, panel, (method, ylabel, title) in zip(axes, "ab", descriptions):
        train = np.asarray(arrays[f"loss_{method}_train"])
        val = np.asarray(arrays[f"loss_{method}_validation"])
        epochs = np.arange(1, len(train) + 1)
        ax.plot(epochs, _safe_log_values(train), lw=1.4, color=BLUE, label="训练")
        ax.plot(epochs, _safe_log_values(val), lw=1.4, color=APRICOT, label="验证")
        ax.set(xlabel="训练轮次", ylabel=ylabel, yscale="log", xlim=(1, len(train)))
        ax.legend(loc="upper right")
        _axis(ax, panel, title)
    fig.tight_layout(w_pad=2.3)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.51, 0.015, "两个损失对应不同物理量，数值不可直接比较", ha="center", color=MUTED, fontsize=8)
    _save(fig, path)


def _thickness(path: Path, config: Config, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True, sharey=True)
    actual = np.asarray(arrays["true_thickness"])
    bounds = (config.thickness_min_nm - 4, config.thickness_max_nm + 4)
    for j, (ax, panel) in enumerate(zip(axes.flat, "abcd")):
        ax.plot(bounds, bounds, color=MUTED, lw=0.9, ls="--", label="一致线", zorder=1)
        ax.scatter(actual[:, j], arrays["supervised_thickness"][:, j], s=6,
                   color=APRICOT, alpha=0.40, edgecolors="none", label="直接 MLP", zorder=2)
        ax.scatter(actual[:, j], arrays["refined_thickness"][:, j], s=6,
                   color=BLUE, alpha=0.43, edgecolors="none", label="MLP 多起点 + 优化", zorder=3)
        ax.set(xlim=bounds, ylim=bounds, aspect="equal", xticks=[40, 80, 120, 180],
               yticks=[40, 80, 120, 180])
        _axis(ax, panel, f"第 {j + 1} 层（{'H' if j % 2 == 0 else 'L'}）")
        if j >= 2:
            ax.set_xlabel("原始膜厚 (nm)")
        if j % 2 == 0:
            ax.set_ylabel("逆向膜厚 (nm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.50, 0.002),
               markerscale=2, handlelength=2.5, columnspacing=1.8)
    fig.tight_layout(w_pad=2, h_pad=1.5, rect=(0, 0.055, 1, 1))
    _save(fig, path)


def _spectrum_lines(ax: plt.Axes, config: Config, arrays: dict[str, np.ndarray], idx: int) -> None:
    ax.plot(config.wavelengths_nm, arrays["target_spectra"][idx], color=INK, lw=1.6, label="目标")
    ax.plot(config.wavelengths_nm, arrays["supervised_spectra"][idx], color=APRICOT,
            lw=1.25, ls="--", label="直接 MLP → TMM")
    ax.plot(config.wavelengths_nm, arrays["physics_spectra"][idx], color=GREEN,
            lw=1.35, ls=":", label="物理 MLP → TMM")
    ax.plot(config.wavelengths_nm, arrays["refined_spectra"][idx], color=BLUE,
            lw=1.35, label="优化 → TMM")
    ax.set(xlim=(config.wavelength_min_nm, config.wavelength_max_nm), ylim=(-0.015, 1.015),
           xlabel="波长 (nm)", xticks=[400, 500, 600, 700, 800], yticks=[0, 0.25, 0.5, 0.75, 1])


def _spectra(path: Path, config: Config, arrays: dict[str, np.ndarray], summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75), sharey=True)
    for ax, panel, idx in zip(axes, "abc", summary["representative_test_indices"]):
        idx = int(idx)
        _spectrum_lines(ax, config, arrays, idx)
        mae = float(np.mean(np.abs(arrays["target_spectra"][idx] - arrays["refined_spectra"][idx])))
        _axis(ax, panel, f"测试样本 {idx}")
        ax.text(0.02, 0.97, f"优化 MAE = {mae:.3g}", transform=ax.transAxes,
                va="top", fontsize=7.5, color=MUTED)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("反射率")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.51, -0.005),
               handlelength=2.3, columnspacing=1.0, fontsize=7.5)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.11, 1, 1))
    _save(fig, path)


def _training_sizes(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    sizes = np.asarray(arrays["training_sizes"])
    metrics = [
        ("size_thickness_mae_nm", "膜厚 MAE (nm)", BLUE),
        ("size_spectral_mae", "回算光谱 MAE", APRICOT),
    ]
    for ax, panel, (key, ylabel, color) in zip(axes, "ab", metrics):
        value = np.asarray(arrays[key])
        ax.plot(sizes, value, marker="o", markersize=4, lw=1.4, color=color)
        for x, y in zip(sizes, value):
            ax.annotate(f"{y:.3g}", (x, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set(xlabel="训练样本数", ylabel=ylabel, xticks=sizes,
               xlim=(float(sizes.min()) * 0.65, float(sizes.max()) * 1.075))
        top = float(value.max())
        ax.set_ylim(0, top * 1.23 if top > 0 else 1)
        _axis(ax, panel)
    fig.tight_layout(w_pad=2.8)
    _save(fig, path)


def _distribution(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    target = np.asarray(arrays["target_spectra"])
    definitions = [
        ("nearest", "最近训练样本", MUTED, "-."),
        ("supervised", "直接逆向 MLP", APRICOT, "--"),
        ("physics", "物理约束 MLP", GREEN, ":"),
        ("refined", "MLP 多起点 + 优化", BLUE, "-"),
    ]
    any_zero = False
    for key, label, color, linestyle in definitions:
        errors = np.sort(np.mean(np.abs(target - np.asarray(arrays[f"{key}_spectra"])), axis=1))
        probabilities = np.arange(1, len(errors) + 1) / len(errors)
        any_zero = any_zero or bool(np.any(errors == 0))
        axes[0].step(errors, probabilities, where="post", color=color, lw=1.5,
                     ls=linestyle, label=label)
    # A symmetric-log scale preserves exact zeros if an exactly matching
    # spectrum occurs; otherwise the conventional positive log axis is used.
    axes[0].set_xscale("symlog", linthresh=1e-6) if any_zero else axes[0].set_xscale("log")
    axes[0].set(xlabel="单样本光谱 MAE", ylabel="累计比例", ylim=(0, 1.02),
                yticks=[0, 0.25, 0.5, 0.75, 1])
    _axis(axes[0], "a", f"全部 {len(target)} 个测试目标")
    history = np.asarray(arrays["refinement_best_mse_history"])
    steps = np.arange(len(history))
    mean = history.mean(axis=1)
    median = np.median(history, axis=1)
    axes[1].plot(steps, _safe_log_values(mean), color=BLUE, lw=1.6, label="均值")
    axes[1].plot(steps, _safe_log_values(median), color=GREEN, ls="--", lw=1.3, label="中位数")
    axes[1].set(xlabel="膜厚优化步数", ylabel="历史最优光谱 MSE", yscale="log",
                xlim=(0, len(history) - 1))
    axes[1].legend(loc="lower left")
    _axis(axes[1], "b", "TMM 物理优化")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.51, 0.002), fontsize=7,
               handlelength=2.3, columnspacing=1.0)
    fig.tight_layout(w_pad=2.5, rect=(0, 0.13, 1, 1))
    _save(fig, path)


def _failure(path: Path, config: Config, arrays: dict[str, np.ndarray], summary: dict) -> None:
    idx = int(summary["failure_test_index"])
    fig = plt.figure(figsize=(7.2, 3.65))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 0.42],
                           hspace=0.14, wspace=0.4)
    ax = fig.add_subplot(grid[0, 0])
    residual = fig.add_subplot(grid[1, 0], sharex=ax)
    table_ax = fig.add_subplot(grid[:, 1])
    _spectrum_lines(ax, config, arrays, idx)
    ax.set_xlabel("")
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("反射率")
    _axis(ax, "a", f"最差测试样本 {idx}")
    ax.legend(loc="upper right", fontsize=6.8, handlelength=2)
    error = np.abs(np.asarray(arrays["target_spectra"][idx]) - np.asarray(arrays["refined_spectra"][idx]))
    residual.plot(config.wavelengths_nm, error, color=BLUE, lw=1.2)
    residual.fill_between(config.wavelengths_nm, 0, error, color=PALE_BLUE)
    residual.set(xlabel="波长 (nm)", ylabel="绝对误差", ylim=(0, max(float(error.max()) * 1.16, 1e-8)))
    residual.tick_params(axis="x", labelsize=7.5)
    _axis(residual)
    table_ax.axis("off")
    table_ax.text(0, 1.015, "b", transform=table_ax.transAxes, fontsize=12, weight="bold")
    table_ax.text(0.12, 1.015, "四层膜厚 (nm)", transform=table_ax.transAxes, fontsize=10)
    rows = [
        ("原始结构", "true_thickness", INK),
        ("直接 MLP", "supervised_thickness", APRICOT),
        ("物理 MLP", "physics_thickness", GREEN),
        ("优化后", "refined_thickness", BLUE),
    ]
    table = table_ax.table(
        cellText=[[label] + [f"{x:.2f}" for x in arrays[key][idx]] for label, key, _ in rows],
        colLabels=["", "$H_1$", "$L_2$", "$H_3$", "$L_4$"],
        colWidths=[0.31, 0.17, 0.17, 0.17, 0.18],
        cellLoc="center", bbox=[0.0, 0.34, 1.02, 0.49],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_facecolor("white")
        cell.set_text_props(color=INK)
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.7)
            cell.set_edgecolor(LIGHT)
        if row == len(rows):
            cell.visible_edges = "B"
            cell.set_linewidth(0.7)
            cell.set_edgecolor(LIGHT)
        if row > 0 and col == 0:
            cell.set_text_props(color=INK, ha="left")
    mae = float(error.mean())
    max_error = float(error.max())
    table_ax.text(0.0, 0.22, f"优化后光谱 MAE  {mae:.4f}\n最大单点误差  {max_error:.4f}",
                  transform=table_ax.transAxes, fontsize=8.5, color=INK, linespacing=1.6)
    table_ax.text(0.0, 0.065, "膜厚不同也可能形成相近光谱；\n逆向解以 TMM 回算响应评价。",
                  transform=table_ax.transAxes, fontsize=7.5, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.15)
    _save(fig, path)


def make_inverse_figures(
    output: Path,
    config: Config,
    arrays: dict[str, np.ndarray],
    summary: dict,
) -> list[str]:
    """Create seven figures under ``output/figures``; return PNG filenames.

    ``supervised_spectra``, ``physics_spectra``, and ``refined_spectra`` must
    already be TMM calculations for their corresponding feasible thicknesses.
    The method named ``nearest`` is a training-set nearest-spectrum baseline.
    Representative indices and the failure index are supplied by the experiment
    so that selection is explicit and reproducible.
    """
    _validate(config, arrays, summary)
    destination = Path(output) / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / name for name in FIGURE_FILENAMES]
    with plt.rc_context(STYLE):
        _workflow(paths[0], config, arrays, summary)
        _losses(paths[1], arrays)
        _thickness(paths[2], config, arrays)
        _spectra(paths[3], config, arrays, summary)
        _training_sizes(paths[4], arrays)
        _distribution(paths[5], arrays)
        _failure(paths[6], config, arrays, summary)
    return list(FIGURE_FILENAMES)
