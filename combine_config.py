"""Terrain selection and composition helpers.

The original HTB generator can choose individual terrain types or compose
several terrain functions. This module keeps that logic without package-level
dependencies on the old training stack.
"""

import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from single_terrain import (
    bridge,
    flat,
    gap,
    hurdle,
    parkour,
    plot,
    slope,
    stair,
    uneven,
    wave,
)
from subterrain import SubTerrain
from terrain_config import TerrainConfig


TerrainFunction = Callable[..., Tuple[SubTerrain, np.ndarray, int]]

SINGLE_TERRAINS: List[Tuple[str, TerrainFunction]] = [
    ("parkour", parkour),
    ("hurdle", hurdle),
    ("bridge", bridge),
    ("flat", flat),
    ("uneven", uneven),
    ("stair", stair),
    ("wave", wave),
    ("slope", slope),
    ("gap", gap),
    ("plot", plot),
]

TERRAIN_NAME_TO_ID: Dict[str, int] = {
    name: index for index, (name, _func) in enumerate(SINGLE_TERRAINS)
}

MULTIPLICATION_TERRAINS: List[List[TerrainFunction]] = [
    [gap, bridge],
    [wave, gap],
    [wave, bridge],
    [gap, wave, stair, bridge],
    [stair, wave, bridge],
]

ADDITION_TERRAINS: List[List[TerrainFunction]] = [[stair, bridge, uneven, gap]]


class CombineConfig:
    """Static terrain function registries."""

    single = [func for _name, func in SINGLE_TERRAINS]
    multiplication = MULTIPLICATION_TERRAINS
    addition = ADDITION_TERRAINS


class Generator:
    """Create terrain tiles from names, ids, or weighted config choices."""

    def __init__(self, cfg: TerrainConfig) -> None:
        """Store the terrain generation config."""
        self.cfg = cfg

    @property
    def terrain_names(self) -> List[str]:
        """Return available single terrain names."""
        return [name for name, _func in SINGLE_TERRAINS]

    def create_by_name(
        self, terrain: SubTerrain, name: str, difficulty: float = 0.5
    ) -> SubTerrain:
        """Create a tile using a public terrain name."""
        normalized = name.lower().strip()
        if normalized not in TERRAIN_NAME_TO_ID:
            allowed = ", ".join(self.terrain_names)
            raise ValueError(f"Unknown terrain '{name}'. Expected one of: {allowed}")
        return self.single_create(terrain, TERRAIN_NAME_TO_ID[normalized], difficulty)

    def single_create(
        self, terrain: SubTerrain, terrain_id: int = 0, difficulty: float = 0.5
    ) -> SubTerrain:
        """Create one tile using a single HTB terrain function."""
        length_x = self.cfg.terrain_length
        length_y = self.cfg.terrain_width
        num_goals = self.cfg.num_goals
        platform_size = self.cfg.platform_size
        func = CombineConfig.single[terrain_id]
        terrain, goals, _final_x = func(
            terrain=terrain,
            length_x=length_x,
            length_y=length_y,
            num_goals=num_goals,
            platform_size=platform_size,
            difficulty=difficulty,
        )
        terrain.goals = goals * terrain.horizontal_scale
        terrain.idx = terrain_id
        terrain.terrain_name = SINGLE_TERRAINS[terrain_id][0]
        return terrain

    def addition_create(
        self, terrain: SubTerrain, terrain_id: int = 0, difficulty: float = 0.5
    ) -> SubTerrain:
        """Create a tile by laying several terrain segments along x."""
        terrain_list = CombineConfig.addition[terrain_id]
        num_terrain = len(terrain_list)
        platform_grid = round(self.cfg.platform_size / terrain.horizontal_scale)
        segment_length = self.cfg.terrain_length / max(num_terrain, 1) + self.cfg.platform_size
        num_goals = max(1, self.cfg.num_goals // max(num_terrain, 1))
        goals = []
        start_x = 0
        for index, func in enumerate(terrain_list):
            current_goals = num_goals
            if index == num_terrain - 1:
                current_goals = self.cfg.num_goals - index * num_goals
            terrain, sub_goals, final_x = func(
                terrain=terrain,
                length_x=segment_length,
                length_y=self.cfg.terrain_width,
                num_goals=current_goals,
                start_x=start_x,
                platform_size=self.cfg.platform_size,
                difficulty=difficulty,
            )
            goals.append(sub_goals)
            start_x = max(0, final_x - platform_grid)
        terrain.goals = np.vstack(goals) * terrain.horizontal_scale
        terrain.idx = terrain_id + len(CombineConfig.single)
        terrain.terrain_name = f"addition_{terrain_id}"
        return terrain

    def multiplication_create(
        self, terrain: SubTerrain, terrain_id: int = 0, difficulty: float = 0.5
    ) -> SubTerrain:
        """Create a tile by applying several terrain functions over the same area."""
        terrain_list = CombineConfig.multiplication[terrain_id]
        goals = None
        start_x = round(self.cfg.platform_size / terrain.horizontal_scale)
        for func in terrain_list:
            terrain, goals, _final_x = func(
                terrain=terrain,
                length_x=self.cfg.terrain_length - self.cfg.platform_size,
                length_y=self.cfg.terrain_width,
                num_goals=self.cfg.num_goals,
                start_x=start_x,
                platform_size=self.cfg.platform_size,
                difficulty=difficulty,
            )
        terrain.goals = np.asarray(goals) * terrain.horizontal_scale
        terrain.idx = terrain_id + len(CombineConfig.single) + len(CombineConfig.addition)
        terrain.terrain_name = f"multiplication_{terrain_id}"
        return terrain

    def random_create(self, terrain: SubTerrain, difficulty: float = 0.5) -> SubTerrain:
        """Create a tile from weighted terrain proportions."""
        pairs = []
        weights = []
        for terrain_type, index, weight in self.cfg.terrain_proportions:
            pairs.append((terrain_type, index))
            weights.append(weight)
        if not pairs:
            raise ValueError("TerrainConfig.terrain_proportions must not be empty")
        selected_type, selected_index = random.choices(pairs, weights=weights, k=1)[0]
        if selected_type == "single":
            return self.single_create(terrain, selected_index, difficulty)
        if selected_type == "addition":
            return self.addition_create(terrain, selected_index, difficulty)
        if selected_type == "multiplication":
            return self.multiplication_create(terrain, selected_index, difficulty)
        raise ValueError(f"Unsupported terrain type group: {selected_type}")


# Backward-compatible names for scripts that used the original module casing.
combine_config = CombineConfig
generator = Generator
