"""Tile and stitch HTB terrain patches into one exportable height field."""

import random
from typing import Optional

import numpy as np

from combine_config import Generator
from subterrain import SubTerrain, random_uniform_terrain
from terrain_config import TerrainConfig


class Terrain:
    """Build a full terrain map plus HTB metadata arrays."""

    def __init__(self, cfg: Optional[TerrainConfig] = None) -> None:
        """Generate all terrain tiles immediately from the supplied config."""
        self.cfg = cfg or TerrainConfig()
        self.env_length = self.cfg.terrain_length
        self.env_width = self.cfg.terrain_width
        self.num_goals = self.cfg.num_goals

        # env_origins marks each tile's nominal robot spawn origin in meters.
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype=np.float64)
        # terrain_type stores the generator id selected for each tile.
        self.terrain_type = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        # goals stores per-tile waypoint coordinates in global terrain meters.
        self.goals = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.cfg.num_goals, 3),
            dtype=np.float64,
        )

        self.width_per_env_pixels = int(round(self.env_width / self.cfg.horizontal_scale))
        self.length_per_env_pixels = int(round(self.env_length / self.cfg.horizontal_scale))
        self.border = int(round(self.cfg.border_size / self.cfg.horizontal_scale))
        self.tot_cols = int(self.cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.generator = Generator(self.cfg)

        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = self._tile_difficulty(row)
                sub_terrain = self.make_terrain(difficulty)
                self.add_terrain_to_map(sub_terrain, row, col)

        self.heightsamples = self.height_field_raw

    def _tile_difficulty(self, row: int) -> float:
        """Return the curriculum or forced difficulty for a row."""
        if self.cfg.forced_difficulty is not None:
            return float(np.clip(self.cfg.forced_difficulty, 0.0, 1.0))
        if self.cfg.curriculum:
            return float(row / max(self.cfg.num_rows, 1))
        return float(np.random.uniform(0.0, 1.0))

    def add_roughness(self, terrain: SubTerrain, difficulty: float = 1.0) -> None:
        """Add stochastic surface roughness to one terrain tile."""
        if not self.cfg.add_roughness:
            return
        min_h, max_h = self.cfg.roughness_height
        max_height = (max_h - min_h) * difficulty + min_h
        height = random.uniform(min_h, max_height)
        random_uniform_terrain(
            terrain,
            min_height=-height,
            max_height=height,
            step=self.cfg.roughness_step,
            downsampled_scale=self.cfg.downsampled_scale,
        )

    def make_terrain(self, difficulty: float) -> SubTerrain:
        """Create one tile from either a named terrain or weighted random choice."""
        terrain = SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        if self.cfg.selected_terrain:
            terrain = self.generator.create_by_name(
                terrain, self.cfg.selected_terrain, difficulty=difficulty
            )
        else:
            terrain = self.generator.random_create(terrain, difficulty=difficulty)
        self.add_roughness(terrain, difficulty=difficulty)
        return terrain

    def add_terrain_to_map(self, terrain: SubTerrain, row: int, col: int) -> None:
        """Copy one tile into the global height field and record metadata."""
        start_x = self.border + row * self.length_per_env_pixels
        end_x = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = self.border + (col + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = row * self.env_length + 1.0
        env_origin_y = (col + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0.0
        else:
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            )
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[row, col] = int(getattr(terrain, "idx", -1))
        self.goals[row, col, :, :2] = terrain.goals + [
            row * self.env_length,
            col * self.env_width,
        ]
