"""Four-input, 41-output MLP and fixed training/evaluation routines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class SpectralMLP(nn.Module):
    """Course-suggested 4-128-128-64-41 architecture with ReLU hidden units."""

    def __init__(self, output_points: int = 41) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_points),
        )

    def forward(self, thickness: torch.Tensor) -> torch.Tensor:
        return self.layers(thickness)


def scale_thickness(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map the permitted thickness interval to [-1, 1]."""
    return ((2.0 * (np.asarray(x, dtype=np.float32) - low) / (high - low)) - 1.0).astype(
        np.float32
    )


@dataclass
class FitResult:
    model: SpectralMLP
    history: dict[str, list[float]]
    best_epoch: int
    best_validation_mse: float


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    low: float,
    high: float,
    seed: int,
    epochs: int = 300,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> FitResult:
    """Start every training-size run from the same seeded initialization.

    All runs use identical optimizer settings and epoch count. The checkpoint
    with lowest fixed-validation MSE is evaluated once on the held-out test set.
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = SpectralMLP(y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    x_tensor = torch.from_numpy(scale_thickness(x_train, low, high))
    y_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    xv_tensor = torch.from_numpy(scale_thickness(x_validation, low, high))
    yv_tensor = torch.from_numpy(np.asarray(y_validation, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    history: dict[str, list[float]] = {"train_mse": [], "validation_mse": []}
    best_validation = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_tensor), generator=generator)
        for batch_indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_tensor[batch_indices])
            loss = loss_fn(prediction, y_tensor[batch_indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            training_loss = loss_fn(model(x_tensor), y_tensor).item()
            validation_loss = loss_fn(model(xv_tensor), yv_tensor).item()
        history["train_mse"].append(training_loss)
        history["validation_mse"].append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, history, best_epoch, best_validation)


def predict(model: SpectralMLP, x: np.ndarray, low: float, high: float) -> np.ndarray:
    model.eval()
    x_scaled = torch.from_numpy(scale_thickness(x, low, high))
    with torch.no_grad():
        return model(x_scaled).numpy().astype(np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
