"""Deterministic course dataset and separate design-candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .physics import tmm_reflectance


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.__dict__)


def make_dataset(config: Config) -> Dataset:
    """Generate once, permute once, then hold validation/test fixed."""
    if config.n_train + config.n_validation + config.n_test != config.n_samples:
        raise ValueError("Split sizes do not add up to the dataset size")
    rng = np.random.default_rng(config.seed)
    x = rng.uniform(
        config.thickness_min_nm,
        config.thickness_max_nm,
        size=(config.n_samples, len(config.refractive_indices)),
    )
    y = tmm_reflectance(
        x,
        config.wavelengths_nm,
        config.refractive_indices,
        config.ambient_index,
        config.substrate_index,
    )
    order = rng.permutation(config.n_samples)
    x, y = x[order], y[order]
    n1 = config.n_train
    n2 = n1 + config.n_validation
    return Dataset(x[:n1], y[:n1], x[n1:n2], y[n1:n2], x[n2:], y[n2:])


def make_design_candidates(config: Config) -> np.ndarray:
    """Use an independent seed so design data cannot copy training samples."""
    rng = np.random.default_rng(config.design_seed)
    return rng.uniform(
        config.thickness_min_nm,
        config.thickness_max_nm,
        size=(config.n_design_candidates, len(config.refractive_indices)),
    )
