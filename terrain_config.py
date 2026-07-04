"""Configuration values for HTB terrain generation and MuJoCo export.

The defaults mirror the compact terrain-only conversion: one grid cell is
`horizontal_scale` meters in x/y, and one raw height unit is `vertical_scale`
meters in z.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


TerrainProportion = Tuple[str, int, float]


@dataclass(frozen=True)
class TerrainConfig:
    """Immutable configuration for tiled terrain generation."""

    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    border_size: float = 5.0

    terrain_length: float = 10.0
    terrain_width: float = 4.0
    platform_size: float = 2.5
    num_rows: int = 10
    num_cols: int = 20
    num_goals: int = 10

    curriculum: bool = True
    origin_zero_z: bool = True
    add_roughness: bool = True
    roughness_height: Tuple[float, float] = (0.02, 0.06)
    roughness_step: float = 0.005
    downsampled_scale: float = 0.075

    selected_terrain: Optional[str] = None
    forced_difficulty: Optional[float] = None
    terrain_proportions: Tuple[TerrainProportion, ...] = field(
        default_factory=lambda: (
            ("single", 0, 1.0),  # parkour
            ("single", 1, 1.0),  # hurdle
            ("single", 2, 1.0),  # bridge
            ("single", 3, 1.0),  # flat
            ("single", 4, 1.0),  # uneven
            ("single", 5, 1.0),  # stair
            ("single", 6, 1.0),  # wave
            ("single", 7, 1.0),  # slope
            ("single", 8, 1.0),  # gap
            ("single", 9, 1.0),  # plot
        )
    )

    def with_updates(self, **kwargs: object) -> "TerrainConfig":
        """Return a copy of the config with selected fields replaced."""
        return replace(self, **kwargs)
