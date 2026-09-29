"""Course-defined physical and student-specific parameters."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    student_name: str = "陈新堉"
    student_id: str = "2023270069"
    refractive_indices: tuple[float, ...] = (2.30, 1.45, 2.30, 1.45)
    ambient_index: float = 1.0
    substrate_index: float = 1.52
    thickness_min_nm: float = 40.0
    thickness_max_nm: float = 180.0
    wavelength_min_nm: int = 400
    wavelength_max_nm: int = 800
    wavelength_step_nm: int = 10
    n_samples: int = 5000
    n_train: int = 4000
    n_validation: int = 500
    n_test: int = 500
    n_design_candidates: int = 10000
    training_sizes: tuple[int, ...] = (500, 1000, 2000, 4000)

    @property
    def target_wavelength_nm(self) -> int:
        last_two = int(self.student_id[-2:])
        return 450 + 10 * (last_two % 31)

    @property
    def seed(self) -> int:
        return int(self.student_id[-6:])

    @property
    def design_seed(self) -> int:
        return self.seed + 1

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return np.arange(
            self.wavelength_min_nm,
            self.wavelength_max_nm + 1,
            self.wavelength_step_nm,
            dtype=np.float64,
        )

    @property
    def target_index(self) -> int:
        locations = np.flatnonzero(self.wavelengths_nm == self.target_wavelength_nm)
        if len(locations) != 1:
            raise ValueError("Target wavelength must occur exactly once in the grid")
        return int(locations[0])

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            target_wavelength_nm=self.target_wavelength_nm,
            seed=self.seed,
            design_seed=self.design_seed,
            wavelengths_nm=self.wavelengths_nm.tolist(),
        )
        return result
