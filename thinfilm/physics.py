"""Normal-incidence characteristic-matrix method for lossless thin films.

At normal incidence the optical admittance equals refractive index. For layer
j, M_j = [[cos(delta), i sin(delta)/n_j],
          [i n_j sin(delta), cos(delta)]],
where delta = 2*pi*n_j*d_j/lambda. Matrices are multiplied from air to glass.
The substrate is semi-infinite and is not one of the four design layers.
"""

from __future__ import annotations

import numpy as np


def tmm_rt(
    thicknesses_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
    refractive_indices: tuple[float, ...] = (2.30, 1.45, 2.30, 1.45),
    ambient_index: float = 1.0,
    substrate_index: float = 1.52,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R and T, each shaped (samples, wavelengths).

    ``thicknesses_nm`` accepts one structure (shape 4,) or a batch (N, 4).
    No artificial clipping is applied: conservation tests can detect mistakes.
    """
    d = np.asarray(thicknesses_nm, dtype=np.float64)
    if d.ndim == 1:
        d = d[None, :]
    wave = np.asarray(wavelengths_nm, dtype=np.float64)
    indices = np.asarray(refractive_indices, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != len(indices):
        raise ValueError("Expected one thickness per optical layer")
    if wave.ndim != 1 or len(wave) == 0 or np.any(wave <= 0):
        raise ValueError("Wavelengths must be a nonempty positive 1D array")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Thicknesses must be finite and nonnegative")
    if np.any(indices <= 0) or ambient_index <= 0 or substrate_index <= 0:
        raise ValueError("Refractive indices must be positive")

    shape = (len(d), len(wave))
    m11 = np.ones(shape, dtype=np.complex128)
    m12 = np.zeros(shape, dtype=np.complex128)
    m21 = np.zeros(shape, dtype=np.complex128)
    m22 = np.ones(shape, dtype=np.complex128)

    for layer, index in enumerate(indices):
        delta = 2.0 * np.pi * index * d[:, layer, None] / wave[None, :]
        cosine = np.cos(delta)
        sine = np.sin(delta)
        a = cosine
        b = 1j * sine / index
        c = 1j * index * sine
        e = cosine
        m11, m12, m21, m22 = (
            m11 * a + m12 * c,
            m11 * b + m12 * e,
            m21 * a + m22 * c,
            m21 * b + m22 * e,
        )

    b_total = m11 + substrate_index * m12
    c_total = m21 + substrate_index * m22
    denominator = ambient_index * b_total + c_total
    r = (ambient_index * b_total - c_total) / denominator
    t = 2.0 * ambient_index / denominator
    reflectance = np.abs(r) ** 2
    transmittance = (substrate_index / ambient_index) * np.abs(t) ** 2
    return reflectance, transmittance


def tmm_reflectance(
    thicknesses_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
    refractive_indices: tuple[float, ...] = (2.30, 1.45, 2.30, 1.45),
    ambient_index: float = 1.0,
    substrate_index: float = 1.52,
) -> np.ndarray:
    """Reflectance in [0, 1] for this lossless physical setup."""
    return tmm_rt(
        thicknesses_nm,
        wavelengths_nm,
        refractive_indices,
        ambient_index,
        substrate_index,
    )[0]
