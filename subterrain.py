"""Local terrain utilities replacing the small Isaac-style subset HTB used.

This module avoids any simulator dependency. `SubTerrain` stores raw int16
height units, and `random_uniform_terrain` adds low-resolution roughness using
numpy/scipy interpolation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class SubTerrain:
    """A rectangular height field tile in raw int16 height units."""

    name: str
    width: int
    length: int
    vertical_scale: float
    horizontal_scale: float
    height_field_raw: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Validate scales and allocate the tile height array."""
        if self.width <= 0 or self.length <= 0:
            raise ValueError("SubTerrain width and length must be positive")
        if self.vertical_scale <= 0.0 or self.horizontal_scale <= 0.0:
            raise ValueError("SubTerrain scales must be positive")
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def _rng_choice(
    rng: Optional[np.random.Generator], values: np.ndarray, size: tuple
) -> np.ndarray:
    """Sample with an optional generator while preserving global RNG behavior."""
    if rng is not None:
        return rng.choice(values, size=size)
    return np.random.choice(values, size=size)


def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float,
    max_height: float,
    step: float = 0.005,
    downsampled_scale: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> SubTerrain:
    """Add low-resolution random roughness to a height field.

    This replaces Isaac Gym terrain_utils.random_uniform_terrain for this
    terrain-only project. Heights are sampled in meters, interpolated to the
    terrain grid, converted to raw int16 height units, and added in place.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    if max_height < min_height:
        raise ValueError("max_height must be greater than or equal to min_height")

    # The original HTB path used simulator terrain utilities for roughness.
    # Here we sample coarse meter-scale heights and interpolate them to the
    # full terrain grid before converting meters back to raw height units.
    downsampled_scale = downsampled_scale or terrain.horizontal_scale
    downsampled_scale = max(float(downsampled_scale), terrain.horizontal_scale)

    values = np.arange(min_height, max_height + step * 0.5, step, dtype=np.float64)
    if values.size == 0:
        values = np.array([min_height], dtype=np.float64)

    full_x = max((terrain.width - 1) * terrain.horizontal_scale, terrain.horizontal_scale)
    full_y = max((terrain.length - 1) * terrain.horizontal_scale, terrain.horizontal_scale)
    low_rows = max(2, int(np.ceil(full_x / downsampled_scale)) + 1)
    low_cols = max(2, int(np.ceil(full_y / downsampled_scale)) + 1)

    low_heights = _rng_choice(rng, values, (low_rows, low_cols))
    low_x = np.linspace(0.0, terrain.width - 1, low_rows)
    low_y = np.linspace(0.0, terrain.length - 1, low_cols)
    interp = RegularGridInterpolator(
        (low_x, low_y), low_heights, bounds_error=False, fill_value=None
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(terrain.width), np.arange(terrain.length), indexing="ij"
    )
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    roughness_m = interp(points).reshape(terrain.width, terrain.length)

    roughness_raw = np.rint(roughness_m / terrain.vertical_scale).astype(np.int32)
    updated = terrain.height_field_raw.astype(np.int32) + roughness_raw
    terrain.height_field_raw = np.clip(updated, -32768, 32767).astype(np.int16)
    return terrain
