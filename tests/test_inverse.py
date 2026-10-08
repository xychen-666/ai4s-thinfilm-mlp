"""Physical, gradient, and artifact checks for spectrum-to-thickness inversion.

These checks use NumPy's independently validated TMM and analytic optical
limits. They do not train an inverse model or assume a unique inverse solution.
"""

import numpy as np
import pytest
import torch

from thinfilm.config import Config
from thinfilm.inverse import (
    InverseMLP,
    decode_thickness,
    differentiable_tmm,
    refine_inverse,
    train_inverse,
)
from thinfilm.physics import tmm_reflectance


@pytest.mark.parametrize(
    "indices,ambient,substrate",
    [
        ((2.30, 1.45, 2.30, 1.45), 1.0, 1.52),
        ((1.91, 1.38, 2.17, 1.61), 1.12, 1.74),
    ],
)
def test_differentiable_tmm_matches_numpy_batch(indices, ambient, substrate) -> None:
    """Catch matrix order, admittance, batch, and wavelength broadcasting errors."""
    thickness = np.random.default_rng(202327).uniform(40, 180, (7, 4))
    wavelengths = np.array([405.0, 463.0, 520.0, 647.0, 795.0])
    expected = tmm_reflectance(thickness, wavelengths, indices, ambient, substrate)
    actual = differentiable_tmm(
        torch.tensor(thickness, dtype=torch.float64),
        torch.tensor(wavelengths, dtype=torch.float64),
        refractive_indices=indices,
        ambient_index=ambient,
        substrate_index=substrate,
    )
    assert actual.shape == (7, 5)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=0, atol=2e-12)
    assert torch.all((actual >= 0) & (actual <= 1))


def test_differentiable_tmm_analytic_optical_limits() -> None:
    """Zero thickness and quarter-wave coating are independent closed-form cases."""
    wavelengths = torch.tensor([400.0, 520.0, 800.0], dtype=torch.float64)
    bare = differentiable_tmm(torch.zeros((2, 4), dtype=torch.float64), wavelengths)
    bare_expected = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
    np.testing.assert_allclose(bare.numpy(), bare_expected, rtol=0, atol=2e-13)

    # Keep the four-layer API, with only one nonzero optical layer.
    quarter_wave = torch.zeros((1, 4), dtype=torch.float64)
    quarter_wave[0, 0] = 520.0 / (4 * 2.30)
    actual = differentiable_tmm(quarter_wave, wavelengths[1:2])
    expected = ((1.0 * 1.52 - 2.30**2) / (1.0 * 1.52 + 2.30**2)) ** 2
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=2e-13)

    matched_index = 1.70
    matched = differentiable_tmm(
        torch.tensor([[45.0, 92.0, 137.0, 171.0]], dtype=torch.float64),
        wavelengths,
        refractive_indices=(matched_index,) * 4,
        ambient_index=matched_index,
        substrate_index=matched_index,
    )
    np.testing.assert_allclose(matched.numpy(), 0, rtol=0, atol=2e-13)


def test_all_four_thickness_gradients_match_numpy_finite_differences() -> None:
    """Check every layer of two asymmetric stacks against an external objective."""
    thickness = np.array([[73.2, 121.4, 56.7, 163.8], [147.3, 62.1, 132.7, 95.4]])
    wavelengths = np.array([415.0, 520.0, 635.0, 780.0])
    weights = np.array([[0.7, 1.3, 0.5, 1.1], [1.1, 0.4, 1.7, 0.9]])
    variable = torch.tensor(thickness, dtype=torch.float64, requires_grad=True)
    reflectance = differentiable_tmm(variable, torch.tensor(wavelengths))
    objective = (reflectance * torch.tensor(weights)).sum()
    (gradient,) = torch.autograd.grad(objective, variable)

    finite_difference = np.empty_like(thickness)
    step_nm = 1e-3
    for sample in range(2):
        for layer in range(4):
            plus = thickness.copy()
            minus = thickness.copy()
            plus[sample, layer] += step_nm
            minus[sample, layer] -= step_nm
            finite_difference[sample, layer] = (
                np.sum(tmm_reflectance(plus, wavelengths) * weights)
                - np.sum(tmm_reflectance(minus, wavelengths) * weights)
            ) / (2 * step_nm)

    # The chosen case must exercise each derivative, including the deepest layer.
    assert np.all(np.abs(finite_difference) > 1e-6)
    assert torch.isfinite(gradient).all()
    np.testing.assert_allclose(gradient.numpy(), finite_difference, rtol=5e-6, atol=2e-9)


def test_decode_thickness_preserves_range_units_and_gradient() -> None:
    normalized = torch.tensor([[-1.0, 0.0, 1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    physical = decode_thickness(normalized)
    np.testing.assert_allclose(physical.detach().numpy(), [[40.0, 110.0, 180.0, 145.0]])
    (gradient,) = torch.autograd.grad(physical.sum(), normalized)
    np.testing.assert_allclose(gradient.numpy(), 70.0)
    np.testing.assert_allclose(
        decode_thickness(normalized.detach(), low=20.0, high=220.0).numpy(),
        [[20.0, 120.0, 220.0, 170.0]],
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_inverse_model_shape_and_saved_weights_are_reproducible(tmp_path, bounded) -> None:
    torch.manual_seed(270069)
    model = InverseMLP(input_points=41, bounded=bounded).eval()
    spectra = torch.linspace(0, 1, 3 * 41).reshape(3, 41)
    with torch.no_grad():
        expected = model(spectra)
    assert expected.shape == (3, 4)
    assert torch.isfinite(expected).all()
    if bounded:
        # A second input checks the physical range beyond the nominal spectrum range.
        with torch.no_grad():
            extreme = model(torch.full((2, 41), 1000.0))
        assert torch.all((extreme >= -1) & (extreme <= 1))
        physical = decode_thickness(extreme)
        assert torch.all((physical >= 40) & (physical <= 180))

    checkpoint = tmp_path / "inverse_weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = InverseMLP(input_points=41, bounded=bounded).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(spectra), expected, rtol=0, atol=0)


def test_refinement_uses_only_spectra_and_bounded_starts_and_keeps_best() -> None:
    """A different structure can be valid: test spectral error, not inverse uniqueness."""
    config = Config()
    # Ground truth generates targets only; it is deliberately not passed to refinement.
    source_thickness = np.array(
        [[62.0, 143.0, 91.0, 170.0], [158.0, 59.0, 137.0, 81.0], [109.0, 76.0, 165.0, 54.0]]
    )
    targets = tmm_reflectance(source_thickness, config.wavelengths_nm)
    starts = np.array(
        [
            [[85.0, 135.0, 100.0, 145.0], [137.0, 78.0, 111.0, 98.0], [129.0, 93.0, 143.0, 72.0]],
            [[121.0, 69.0, 154.0, 92.0], [75.0, 161.0, 79.0, 150.0], [67.0, 155.0, 88.0, 133.0]],
        ]
    )
    targets_before = targets.copy()
    starts_before = starts.copy()
    start_spectra = tmm_reflectance(starts.reshape(-1, 4), config.wavelengths_nm).reshape(2, 3, 41)
    initial_expected = np.mean((start_spectra - targets[None, :, :]) ** 2, axis=-1).min(axis=0)

    steps = 40
    result = refine_inverse(targets, starts, config, steps=steps, learning_rate=0.02)
    thickness = np.asarray(result["thickness_nm"])
    spectra = np.asarray(result["spectra"])
    history = np.asarray(result["best_mse_history"])
    final_mse = np.asarray(result["final_mse"])
    start_index = np.asarray(result["start_index"])
    assert thickness.shape == (3, 4)
    assert spectra.shape == (3, 41)
    assert history.shape == (steps + 1, 3)
    assert final_mse.shape == (3,)
    assert start_index.shape == (3,)
    assert np.issubdtype(start_index.dtype, np.integer)
    assert np.all((start_index >= 0) & (start_index < 2))
    assert np.isfinite(thickness).all() and np.isfinite(spectra).all()
    assert np.all((thickness >= 40) & (thickness <= 180))

    independently_verified = tmm_reflectance(thickness, config.wavelengths_nm)
    independently_computed_mse = np.mean((independently_verified - targets) ** 2, axis=1)
    np.testing.assert_allclose(spectra, independently_verified, rtol=0, atol=2e-6)
    np.testing.assert_allclose(final_mse, independently_computed_mse, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(result["initial_mse"], initial_expected, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(history[0], initial_expected, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(history[-1], final_mse, rtol=1e-5, atol=1e-9)
    assert np.all(np.diff(history, axis=0) <= 1e-12)
    assert np.all(final_mse <= initial_expected + 1e-9)
    assert np.any(final_mse < initial_expected - 1e-6), "Refinement must do useful optimization."
    np.testing.assert_array_equal(targets, targets_before)
    np.testing.assert_array_equal(starts, starts_before)


@pytest.mark.parametrize("invalid_thickness", [39.999, 180.001, np.nan])
def test_refinement_rejects_infeasible_initial_structures(invalid_thickness) -> None:
    config = Config()
    targets = tmm_reflectance(np.full((1, 4), 110.0), config.wavelengths_nm)
    starts = np.full((1, 1, 4), 110.0)
    starts[0, 0, 2] = invalid_thickness
    with pytest.raises(ValueError, match="thickness|bounds"):
        refine_inverse(targets, starts, config, steps=0)


def test_physics_loss_training_is_independent_of_thickness_labels() -> None:
    """A tiny deterministic run detects accidental supervision or label leakage."""
    config = Config()
    thickness = np.random.default_rng(26).uniform(40, 180, (16, 4))
    spectra = tmm_reflectance(thickness, config.wavelengths_nm)
    alternative_labels = np.full_like(thickness, 40.0)
    fits = [
        train_inverse(
            spectra[:12], labels[:12], spectra[12:], labels[12:], config,
            physics_loss=True, epochs=2, batch_size=4,
        )
        for labels in (thickness, alternative_labels)
    ]
    assert fits[0].loss_kind == "exact_TMM_spectral_MSE"
    assert fits[0].model.bounded and fits[1].model.bounded
    assert fits[0].history == fits[1].history
    assert fits[0].best_epoch == fits[1].best_epoch
    for name, value in fits[0].model.state_dict().items():
        torch.testing.assert_close(value, fits[1].model.state_dict()[name], rtol=0, atol=0)
