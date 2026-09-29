"""Independent physical checks of the characteristic-matrix calculation."""

import numpy as np

from thinfilm.config import Config
from thinfilm.data import make_dataset, make_design_candidates
from thinfilm.physics import tmm_rt


def fresnel_recursion(thicknesses: np.ndarray, wavelength: float, indices: tuple[float, ...]) -> float:
    """Independent recursive interface calculation for normal incidence."""
    media = (1.0, *indices, 1.52)
    result = (media[-2] - media[-1]) / (media[-2] + media[-1])
    for layer in reversed(range(len(indices))):
        interface = (media[layer] - media[layer + 1]) / (media[layer] + media[layer + 1])
        phase = np.exp(2j * 2.0 * np.pi * indices[layer] * thicknesses[layer] / wavelength)
        result = (interface + result * phase) / (1 + interface * result * phase)
    return float(abs(result) ** 2)


def test_zero_thickness_is_bare_air_glass_interface() -> None:
    wave = np.array([400.0, 520.0, 800.0])
    r, t = tmm_rt(np.zeros(4), wave)
    expected = ((1.0 - 1.52) / (1.0 + 1.52)) ** 2
    np.testing.assert_allclose(r[0], expected, atol=1e-13)
    np.testing.assert_allclose(r + t, 1, atol=1e-13)


def test_single_layer_quarter_wave_analytic_result() -> None:
    wave = 520.0
    film_index = 2.30
    r, t = tmm_rt(np.array([wave / (4 * film_index)]), np.array([wave]), (film_index,))
    expected = ((1.0 * 1.52 - film_index**2) / (1.0 * 1.52 + film_index**2)) ** 2
    np.testing.assert_allclose(r[0, 0], expected, atol=1e-13)
    np.testing.assert_allclose(r + t, 1, atol=1e-13)


def test_batch_matches_independent_fresnel_recursion_and_conserves_energy() -> None:
    config = Config()
    rng = np.random.default_rng(123)
    thickness = rng.uniform(40, 180, (24, 4))
    waves = config.wavelengths_nm
    r, t = tmm_rt(thickness, waves)
    reference = np.array([
        [fresnel_recursion(row, wavelength, config.refractive_indices) for wavelength in waves]
        for row in thickness
    ])
    np.testing.assert_allclose(r, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r + t, 1, rtol=0, atol=1e-12)
    assert np.all((r >= 0) & (r <= 1))


def test_personal_parameters_and_fixed_independent_data() -> None:
    config = Config()
    assert (config.target_wavelength_nm, config.seed, config.design_seed) == (520, 270069, 270070)
    first = make_dataset(config)
    second = make_dataset(config)
    for key in first.__dict__:
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    assert first.x_train.shape == (4000, 4)
    assert first.y_test.shape == (500, 41)
    candidates = make_design_candidates(config)
    assert candidates.shape == (10000, 4)
    assert not np.array_equal(candidates[:500], first.x_train[:500])
    assert np.all((candidates >= 40) & (candidates < 180))
