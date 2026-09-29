"""Run the complete coursework experiment and regenerate every result."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch

from .config import Config
from .data import make_dataset, make_design_candidates
from .model import predict, regression_metrics, train_model
from .physics import tmm_reflectance
from .plots import (
    plot_architecture,
    plot_design,
    plot_failure,
    plot_losses,
    plot_physics_data,
    plot_screening_audit,
    plot_test_spectra,
    plot_training_sizes,
    plot_workflow,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, epochs: int, batch_size: int, learning_rate: float) -> dict:
    config = Config()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "histories").mkdir(exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    dataset = make_dataset(config)
    dataset.save(str(output / "data" / "dataset.npz"))
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)

    figdir = output / "figures"
    plot_workflow(figdir / "figure_1_workflow.png")
    plot_physics_data(figdir / "figure_2_tmm_data.png", config, dataset.y_train[:5])
    plot_architecture(figdir / "figure_3_mlp_architecture.png")

    training_rows: list[dict] = []
    main_fit = None
    main_prediction = None
    for size in config.training_sizes:
        print(f"Training MLP with {size} samples for {epochs} epochs...", flush=True)
        start = perf_counter()
        fit = train_model(
            dataset.x_train[:size],
            dataset.y_train[:size],
            dataset.x_validation,
            dataset.y_validation,
            low=config.thickness_min_nm,
            high=config.thickness_max_nm,
            seed=config.seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        elapsed = perf_counter() - start
        test_prediction = predict(fit.model, dataset.x_test, config.thickness_min_nm, config.thickness_max_nm)
        metrics = regression_metrics(dataset.y_test, test_prediction)
        target_error = test_prediction[:, config.target_index] - dataset.y_test[:, config.target_index]
        row = {
            "training_samples": size,
            "best_epoch": fit.best_epoch,
            "validation_mse": fit.best_validation_mse,
            "test_mse": metrics["mse"],
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "test_r2": metrics["r2"],
            "target_520nm_mae": float(np.mean(np.abs(target_error))),
            "predictions_outside_0_1": int(np.count_nonzero((test_prediction < 0) | (test_prediction > 1))),
            "training_seconds": elapsed,
        }
        training_rows.append(row)
        print(f"  test MSE={metrics['mse']:.7f}, MAE={metrics['mae']:.5f}, best epoch={fit.best_epoch}", flush=True)
        torch.save(fit.model.state_dict(), output / "models" / f"mlp_{size}.pt")
        with (output / "histories" / f"loss_{size}.json").open("w", encoding="utf-8") as handle:
            json.dump(fit.history, handle)
        if size == config.n_train:
            main_fit = fit
            main_prediction = test_prediction

    assert main_fit is not None and main_prediction is not None
    write_csv(output / "training_size_metrics.csv", training_rows)
    plot_losses(figdir / "figure_4_training_losses.png", main_fit.history, main_fit.best_epoch)
    sample_maes = np.mean(np.abs(dataset.y_test - main_prediction), axis=1)
    representative = [int(np.argsort(sample_maes)[round(q * (len(sample_maes) - 1))]) for q in (0.25, 0.50, 0.75)]
    failure_index = int(np.argmax(sample_maes))
    plot_test_spectra(figdir / "figure_5_test_spectra.png", config, dataset.y_test, main_prediction, representative)
    plot_training_sizes(figdir / "figure_6_training_size.png", [r["training_samples"] for r in training_rows], [r["test_mse"] for r in training_rows])
    plot_failure(figdir / "figure_8_failure_case.png", config, dataset.y_test[failure_index], main_prediction[failure_index], failure_index)

    # New independent candidates; rank by *predicted high reflectance* at 520 nm.
    candidates = make_design_candidates(config)
    start = perf_counter()
    candidate_prediction = predict(main_fit.model, candidates, config.thickness_min_nm, config.thickness_max_nm)
    mlp_screening_seconds = perf_counter() - start
    target_predictions = candidate_prediction[:, config.target_index]
    top10_indices = np.argsort(-target_predictions, kind="stable")[:10]
    top10_structures = candidates[top10_indices]
    top10_verified_spectra = tmm_reflectance(
        top10_structures,
        config.wavelengths_nm,
        config.refractive_indices,
        config.ambient_index,
        config.substrate_index,
    )
    top10_verified_target = top10_verified_spectra[:, config.target_index]
    final_order = np.argsort(-top10_verified_target, kind="stable")[:5]
    final_rows: list[dict] = []
    top10_rows: list[dict] = []
    for predicted_rank, (global_idx, structure, mlp_r, tmm_r) in enumerate(
        zip(top10_indices, top10_structures, target_predictions[top10_indices], top10_verified_target), start=1
    ):
        top10_rows.append({
            "mlp_rank": predicted_rank,
            "candidate_index": int(global_idx),
            "d1_nm": float(structure[0]), "d2_nm": float(structure[1]),
            "d3_nm": float(structure[2]), "d4_nm": float(structure[3]),
            "mlp_r_target": float(mlp_r), "tmm_r_target": float(tmm_r),
            "absolute_error": float(abs(mlp_r - tmm_r)),
        })
    for verified_rank, position in enumerate(final_order, start=1):
        final_rows.append({"tmm_rank": verified_rank, **top10_rows[int(position)]})
    write_csv(output / "top10_tmm_verification.csv", top10_rows)
    write_csv(output / "top5_designs.csv", final_rows)
    best_position = int(final_order[0])
    plot_design(
        figdir / "figure_7_selected_design.png", config,
        candidate_prediction[top10_indices[best_position]], top10_verified_spectra[best_position],
        float(target_predictions[top10_indices[best_position]]), float(top10_verified_target[best_position]),
    )
    plot_screening_audit(figdir / "screening_audit.png", target_predictions[top10_indices], top10_verified_target, final_order.tolist())

    # Audit all 10,000 candidates with the same physical model, without using
    # their labels in MLP selection. This quantifies the screening outcome.
    audit_start = perf_counter()
    audit_target = tmm_reflectance(
        candidates, np.array([config.target_wavelength_nm], dtype=np.float64),
        config.refractive_indices, config.ambient_index, config.substrate_index,
    )[:, 0]
    audit_seconds = perf_counter() - audit_start
    physical_best_index = int(np.argmax(audit_target))
    verified_best = float(top10_verified_target[best_position])
    physical_best = float(audit_target[physical_best_index])
    best_rank_in_pool = int(1 + np.count_nonzero(audit_target > verified_best))
    summary = {
        "identity": {"student_name": config.student_name, "student_id": config.student_id},
        "target_wavelength_nm": config.target_wavelength_nm,
        "seed": config.seed,
        "design_seed": config.design_seed,
        "design_objective": "maximize reflectance at target wavelength",
        "setup": {"n_samples": config.n_samples, "split": [config.n_train, config.n_validation, config.n_test], "n_candidates": config.n_design_candidates, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "normalization": "(thickness - 40) * 2 / 140 - 1"},
        "training_size_results": training_rows,
        "representative_test_indices": representative,
        "failure_test_index": failure_index,
        "failure_spectrum_mae": float(sample_maes[failure_index]),
        "best_design": final_rows[0],
        "screening_audit": {
            "best_tmm_r_among_all_10000": physical_best,
            "best_tmm_candidate_index": physical_best_index,
            "selected_best_tmm_rank_among_all_10000": best_rank_in_pool,
            "selected_to_pool_best_ratio": verified_best / physical_best,
            "mlp_10000_prediction_seconds": mlp_screening_seconds,
            "tmm_10000_single_wavelength_seconds": audit_seconds,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Done. Best TMM-verified candidate R({config.target_wavelength_nm} nm)={verified_best:.6f}; results at {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    run(args.output, args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
