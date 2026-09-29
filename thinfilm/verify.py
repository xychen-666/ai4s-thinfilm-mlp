"""Independently check all saved data, model metrics, and design rankings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import make_dataset, make_design_candidates
from .model import SpectralMLP, predict, regression_metrics
from .physics import tmm_rt, tmm_reflectance


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(output: Path) -> None:
    config = Config()
    torch.set_num_threads(1)
    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["target_wavelength_nm"] == 520
    assert summary["seed"] == 270069 and summary["design_seed"] == 270070

    saved = np.load(output / "data" / "dataset.npz")
    regenerated = make_dataset(config)
    for key, value in regenerated.__dict__.items():
        np.testing.assert_array_equal(saved[key], value)
    x_all = np.concatenate((saved["x_train"], saved["x_validation"], saved["x_test"]))
    y_all = np.concatenate((saved["y_train"], saved["y_validation"], saved["y_test"]))
    r, t = tmm_rt(x_all, config.wavelengths_nm)
    np.testing.assert_allclose(y_all, r, atol=1e-12, rtol=0)
    np.testing.assert_allclose(r + t, 1, atol=1e-12, rtol=0)
    assert np.all((r >= 0) & (r <= 1))

    metric_rows = _read_csv(output / "training_size_metrics.csv")
    assert [int(row["training_samples"]) for row in metric_rows] == list(config.training_sizes)
    main_model = None
    for row in metric_rows:
        size = int(row["training_samples"])
        model = SpectralMLP()
        model.load_state_dict(torch.load(output / "models" / f"mlp_{size}.pt", map_location="cpu", weights_only=True))
        predictions = predict(model, saved["x_test"], 40, 180)
        metrics = regression_metrics(saved["y_test"], predictions)
        for name in ("mse", "mae", "rmse", "r2"):
            np.testing.assert_allclose(metrics[name], float(row[f"test_{name}"]), atol=1e-7, rtol=0)
        if size == config.n_train:
            main_model = model
    assert main_model is not None

    candidates = make_design_candidates(config)
    predictions = predict(main_model, candidates, 40, 180)
    target_pred = predictions[:, config.target_index]
    top10_indices = np.argsort(-target_pred, kind="stable")[:10]
    top10_rows = _read_csv(output / "top10_tmm_verification.csv")
    assert len(top10_rows) == 10
    verified = tmm_reflectance(candidates[top10_indices], config.wavelengths_nm)
    for rank, (index, row) in enumerate(zip(top10_indices, top10_rows), start=1):
        assert int(row["mlp_rank"]) == rank and int(row["candidate_index"]) == index
        for layer in range(4):
            np.testing.assert_allclose(float(row[f"d{layer + 1}_nm"]), candidates[index, layer], atol=1e-12)
        np.testing.assert_allclose(float(row["mlp_r_target"]), target_pred[index], atol=1e-6)
        np.testing.assert_allclose(float(row["tmm_r_target"]), verified[rank - 1, config.target_index], atol=1e-12)
    top5_rows = _read_csv(output / "top5_designs.csv")
    expected_order = np.argsort(-verified[:, config.target_index], kind="stable")[:5]
    assert len(top5_rows) == 5
    for verified_rank, position in enumerate(expected_order, start=1):
        assert int(top5_rows[verified_rank - 1]["tmm_rank"]) == verified_rank
        assert int(top5_rows[verified_rank - 1]["candidate_index"]) == top10_indices[position]

    audit = tmm_reflectance(candidates, np.array([config.target_wavelength_nm]))[:, 0]
    chosen_r = float(verified[expected_order[0], config.target_index])
    np.testing.assert_allclose(chosen_r, summary["best_design"]["tmm_r_target"], atol=1e-12)
    np.testing.assert_allclose(float(np.max(audit)), summary["screening_audit"]["best_tmm_r_among_all_10000"], atol=1e-12)
    assert int(1 + np.count_nonzero(audit > chosen_r)) == summary["screening_audit"]["selected_best_tmm_rank_among_all_10000"]
    print("Verification passed: 5000 TMM spectra, energy conservation, four test metrics, Top 10, Top 5, and 10000-candidate audit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
