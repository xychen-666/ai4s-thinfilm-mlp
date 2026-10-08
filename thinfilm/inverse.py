"""Inverse MLPs and physically evaluated, bounded spectral reconstruction.

The supervised baseline preserves the forward experiment's hidden widths,
linear output, optimizer and schedule. A separate bounded MLP is trained with
the exact differentiable TMM spectrum loss. Refinement is an explicitly
separate inference-time optimization, never included in one-pass MLP scores.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import Config
from .model import scale_thickness
from .physics import tmm_reflectance


class InverseMLP(nn.Module):
    """41-128-128-64-4; raw reflectance input, normalized thickness output."""

    def __init__(self, input_points: int = 41, bounded: bool = False) -> None:
        super().__init__()
        self.bounded = bounded
        self.layers = nn.Sequential(
            nn.Linear(input_points, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        output = self.layers(spectra)
        return torch.tanh(output) if self.bounded else output


def decode_thickness(normalized: torch.Tensor, low: float = 40., high: float = 180.) -> torch.Tensor:
    if high <= low:
        raise ValueError('Thickness interval must have positive width')
    return low + (normalized + 1.) * ((high - low) / 2.)


def differentiable_tmm(
    thicknesses_nm: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    refractive_indices: tuple[float, ...] = (2.30, 1.45, 2.30, 1.45),
    ambient_index: float = 1.,
    substrate_index: float = 1.52,
) -> torch.Tensor:
    """Exact characteristic-matrix R using real-valued differentiable algebra.

    M11=a and M22=e are real; M12=i*b, M21=i*c for this lossless model.
    Keeping those four real quantities avoids complex autodiff overhead.
    This is algebraically identical to physics.tmm_rt, not a learned proxy.
    """
    if thicknesses_nm.ndim != 2 or thicknesses_nm.shape[1] != len(refractive_indices):
        raise ValueError('Expected a batch with one thickness per film')
    if wavelengths_nm.ndim != 1 or wavelengths_nm.numel() == 0:
        raise ValueError('Expected a nonempty 1D wavelength grid')
    # Callers validate finite, positive optical constants and feasible designs.
    wave = wavelengths_nm.to(dtype=thicknesses_nm.dtype, device=thicknesses_nm.device)
    shape = (thicknesses_nm.shape[0], wave.numel())
    a = torch.ones(shape, dtype=thicknesses_nm.dtype, device=thicknesses_nm.device)
    b = torch.zeros_like(a); c = torch.zeros_like(a); e = torch.ones_like(a)
    for j, index in enumerate(refractive_indices):
        phase = 2. * torch.pi * index * thicknesses_nm[:, j, None] / wave[None, :]
        cosine, sine = torch.cos(phase), torch.sin(phase)
        a, b, c, e = (
            a * cosine - b * index * sine,
            a * sine / index + b * cosine,
            c * cosine + e * index * sine,
            e * cosine - c * sine / index,
        )
    nr = ambient_index * a - substrate_index * e
    ni = ambient_index * substrate_index * b - c
    dr = ambient_index * a + substrate_index * e
    di = ambient_index * substrate_index * b + c
    return (nr.square() + ni.square()) / (dr.square() + di.square())


@dataclass
class InverseFit:
    model: InverseMLP
    history: dict[str, list[float]]
    best_epoch: int
    best_validation_loss: float
    loss_kind: str


def train_inverse(
    spectra_train: np.ndarray,
    thickness_train: np.ndarray,
    spectra_validation: np.ndarray,
    thickness_validation: np.ndarray,
    config: Config,
    *, physics_loss: bool = False,
    epochs: int = 300, batch_size: int = 128, learning_rate: float = 1e-3,
) -> InverseFit:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('Training settings must be positive')
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = InverseMLP(len(config.wavelengths_nm), bounded=physics_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    inputs = torch.from_numpy(np.asarray(spectra_train, dtype=np.float32))
    validation_inputs = torch.from_numpy(np.asarray(spectra_validation, dtype=np.float32))
    targets = torch.from_numpy(scale_thickness(thickness_train, config.thickness_min_nm, config.thickness_max_nm))
    validation_targets = torch.from_numpy(scale_thickness(thickness_validation, config.thickness_min_nm, config.thickness_max_nm))
    wave = torch.tensor(config.wavelengths_nm, dtype=torch.float32)

    def objective(x: torch.Tensor, target_d: torch.Tensor) -> torch.Tensor:
        inverse_output = model(x)
        if physics_loss:
            d = decode_thickness(inverse_output, config.thickness_min_nm, config.thickness_max_nm)
            reconstructed = differentiable_tmm(d, wave, config.refractive_indices, config.ambient_index, config.substrate_index)
            return loss_fn(reconstructed, x)
        return loss_fn(inverse_output, target_d)

    generator = torch.Generator().manual_seed(config.seed)
    history: dict[str, list[float]] = {'train_loss': [], 'validation_loss': []}
    best = float('inf'); best_epoch = 0; best_state = deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(inputs[indices], targets[indices])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = float(objective(inputs, targets))
            validation_loss = float(objective(validation_inputs, validation_targets))
        history['train_loss'].append(training_loss)
        history['validation_loss'].append(validation_loss)
        if validation_loss < best:
            best = validation_loss; best_epoch = epoch; best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 50 == 0:
            print(f'  epoch {epoch}/{epochs}: train={training_loss:.6g}, validation={validation_loss:.6g}', flush=True)
    model.load_state_dict(best_state); model.eval()
    return InverseFit(model, history, best_epoch, best,
                      'exact_TMM_spectral_MSE' if physics_loss else 'normalized_thickness_MSE')


def predict_inverse(model: InverseMLP, spectra: np.ndarray, config: Config) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(np.asarray(spectra, dtype=np.float32)))
        return decode_thickness(output, config.thickness_min_nm, config.thickness_max_nm).numpy().astype(np.float64)


def evaluate_inverse(target_spectra: np.ndarray, true_thickness: np.ndarray,
                     predicted_thickness: np.ndarray, config: Config) -> tuple[dict, np.ndarray]:
    reconstructed = tmm_reflectance(predicted_thickness, config.wavelengths_nm,
                                   config.refractive_indices, config.ambient_index, config.substrate_index)
    de = np.asarray(predicted_thickness, dtype=np.float64) - true_thickness
    se = reconstructed - target_spectra
    sample_mae = np.mean(np.abs(se), axis=1)
    denominator = float(np.sum((true_thickness - np.mean(true_thickness, axis=0))**2))
    metrics = {
        'thickness_mae_nm': float(np.mean(np.abs(de))),
        'thickness_rmse_nm': float(np.sqrt(np.mean(de**2))),
        'thickness_r2': float(1. - np.sum(de**2) / denominator),
        'layer_mae_nm': np.mean(np.abs(de), axis=0).tolist(),
        'layer_rmse_nm': np.sqrt(np.mean(de**2, axis=0)).tolist(),
        'spectral_mse': float(np.mean(se**2)), 'spectral_mae': float(np.mean(np.abs(se))),
        'spectral_rmse': float(np.sqrt(np.mean(se**2))),
        'target_520nm_mae': float(np.mean(np.abs(se[:, config.target_index]))),
        'mean_spectrum_mae': float(np.mean(sample_mae)),
        'median_spectrum_mae': float(np.median(sample_mae)),
        'p90_spectrum_mae': float(np.quantile(sample_mae, .9)),
        'max_spectrum_mae': float(np.max(sample_mae)),
        'spectra_mae_below_0_01_fraction': float(np.mean(sample_mae < .01)),
        'spectra_mae_below_0_02_fraction': float(np.mean(sample_mae < .02)),
        'out_of_bounds_thickness_count': int(np.count_nonzero((predicted_thickness < config.thickness_min_nm) | (predicted_thickness > config.thickness_max_nm))),
    }
    return metrics, reconstructed


def nearest_training_structure(target_spectra: np.ndarray, training_spectra: np.ndarray,
                               training_thickness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A training-only lookup baseline; target thickness is never consulted."""
    targets = np.asarray(target_spectra, dtype=np.float64)
    bank = np.asarray(training_spectra, dtype=np.float64)
    indices = []
    for batch in np.array_split(targets, max(1, (len(targets) + 127) // 128)):
        distance = ((batch[:, None, :] - bank[None, :, :])**2).mean(axis=2)
        indices.append(np.argmin(distance, axis=1))
    idx = np.concatenate(indices)
    return np.asarray(training_thickness)[idx], idx


def refine_inverse(target_spectra: np.ndarray, starts_nm: np.ndarray, config: Config,
                   steps: int = 400, learning_rate: float = .02) -> dict[str, np.ndarray]:
    """Optimize only supplied target spectra; no reference thickness input.

    Each start is a separate bounded four-parameter optimization. The best
    iterate for each start is retained, then the lowest spectral MSE start is
    selected per target. initial_mse means the minimum across all initial starts.
    """
    target = np.asarray(target_spectra, dtype=np.float64)
    starts = np.asarray(starts_nm, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != len(config.wavelengths_nm):
        raise ValueError('Target spectra must have shape (N, 41)')
    if starts.ndim != 3 or starts.shape[1:] != (len(target), 4) or starts.shape[0] == 0:
        raise ValueError('Initial structures must have shape (K, N, 4)')
    if not np.all(np.isfinite(target)) or np.any((target < 0) | (target > 1)):
        raise ValueError('Target reflectance must be finite and in [0, 1]')
    if not np.all(np.isfinite(starts)) or np.any((starts < config.thickness_min_nm) | (starts > config.thickness_max_nm)):
        raise ValueError('Initial thicknesses must be finite and within bounds')
    if steps < 0 or learning_rate <= 0:
        raise ValueError('Invalid refinement settings')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    k, n, _ = starts.shape
    low, high = config.thickness_min_nm, config.thickness_max_nm
    initial_q = 2. * (starts.reshape(-1, 4) - low) / (high - low) - 1.
    q = nn.Parameter(torch.tensor(initial_q, dtype=torch.float64))
    goal = torch.tensor(np.tile(target, (k, 1)), dtype=torch.float64)
    wave = torch.tensor(config.wavelengths_nm, dtype=torch.float64)
    optimizer = torch.optim.Adam([q], lr=learning_rate)

    def losses() -> torch.Tensor:
        d = decode_thickness(q, low, high)
        r = differentiable_tmm(d, wave, config.refractive_indices, config.ambient_index, config.substrate_index)
        return (r - goal).square().mean(dim=1)

    with torch.no_grad():
        best_loss = losses(); best_q = q.detach().clone()
    history = [best_loss.reshape(k, n).amin(dim=0).numpy().copy()]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # Sum keeps gradients of independent designs independent of batch size.
        loss = losses().sum(); loss.backward(); optimizer.step()
        with torch.no_grad():
            q.clamp_(-1., 1.)
            current = losses(); improved = current < best_loss
            best_q[improved] = q[improved]; best_loss = torch.minimum(best_loss, current)
            history.append(best_loss.reshape(k, n).amin(dim=0).numpy().copy())
        if step == 1 or step % 100 == 0:
            print(f'  TMM refine {step}/{steps}: mean best MSE={np.mean(history[-1]):.6g}', flush=True)
    best_matrix = best_loss.reshape(k, n).numpy()
    chosen = np.argmin(best_matrix, axis=0)
    structures = decode_thickness(best_q, low, high).numpy().reshape(k, n, 4)
    final_d = structures[chosen, np.arange(n)]
    final_r = tmm_reflectance(final_d, config.wavelengths_nm, config.refractive_indices, config.ambient_index, config.substrate_index)
    final_mse = np.mean((final_r - target)**2, axis=1)
    return {'thickness_nm': final_d, 'spectra': final_r, 'initial_mse': history[0],
            'final_mse': final_mse, 'best_mse_history': np.asarray(history), 'start_index': chosen}


def make_mlp_starts(supervised: np.ndarray, physics: np.ndarray, config: Config) -> np.ndarray:
    """Two neural initializations and three fixed-seed local perturbations."""
    low, high = config.thickness_min_nm, config.thickness_max_nm
    rng = np.random.default_rng(config.design_seed)
    starts = [np.clip(supervised, low, high), np.clip(physics, low, high)]
    for sigma in (5., 10., 20.):
        starts.append(np.clip(physics + rng.normal(0., sigma, size=physics.shape), low, high))
    return np.asarray(starts)
