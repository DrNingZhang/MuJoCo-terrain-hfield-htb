"""Individual HTB terrain primitives implemented with numpy height fields.

Each function mutates a `SubTerrain.height_field_raw` array and returns the
tile, local goal positions, and the final x index touched by the primitive.
Coordinates are in grid indices inside the function and converted to meters by
the generator when metadata is stored.
"""

import math
import random
from typing import Tuple

import numpy as np

from subterrain import SubTerrain


def _set_region(
    terrain: SubTerrain,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    value: object,
) -> None:
    """Safely assign a scalar or array into a clipped height-field region."""
    x0 = int(round(x0))
    x1 = int(round(x1))
    y0 = int(round(y0))
    y1 = int(round(y1))
    xs = max(0, x0)
    xe = min(terrain.height_field_raw.shape[0], x1)
    ys = max(0, y0)
    ye = min(terrain.height_field_raw.shape[1], y1)
    if xe <= xs or ye <= ys:
        return

    if np.isscalar(value):
        terrain.height_field_raw[xs:xe, ys:ye] = int(round(float(value)))
        return

    arr = np.asarray(value)
    src_x = xs - x0
    src_y = ys - y0
    terrain.height_field_raw[xs:xe, ys:ye] = arr[
        src_x : src_x + (xe - xs), src_y : src_y + (ye - ys)
    ].astype(np.int16)


def parkour(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 6,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 2.5,
    difficulty: float = 0.5,
    x_range: Tuple[float, float] = (0.5, 1.0),
    y_range: Tuple[float, float] = (0.3, 0.4),
    stone_len_range: Tuple[float, float] = (0.8, 1.0),
    stone_width_range: Tuple[float, float] = (0.6, 0.8),
    incline_height: float = 0.1,
    pit_depth: Tuple[float, float] = (0.5, 1.0),
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create stepping stones over a lowered pit for parkour-style traversal."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    pit_depth_grid = -round(random.uniform(*pit_depth) / terrain.vertical_scale)

    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale
    length_y_grid = round(length_y / h_scale)
    mid_y = length_y_grid // 2
    length_x_grid = round(length_x / h_scale)

    stone_len = round(
        ((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1])
        / h_scale
    )
    stone_width = round(
        (
            (stone_width_range[0] - stone_width_range[1]) * difficulty
            + stone_width_range[1]
        )
        / h_scale
    )
    gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)
    gap_y = round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)
    platform_size_grid = int(round(platform_size / h_scale))
    incline_height_grid = int(round(incline_height / v_scale))

    _set_region(
        terrain,
        start_x + platform_size_grid,
        start_x + length_x_grid,
        start_y,
        start_y + length_y_grid,
        pit_depth_grid,
    )

    dis_x = start_x + platform_size_grid - gap_x + stone_len // 2
    goals[0] = [start_x + platform_size_grid - stone_len // 2, start_y + mid_y]
    left_right_flag = random.randint(0, 1)

    for i in range(max(0, num_goals - 2)):
        dis_x += gap_x
        pos_neg = 2 * (left_right_flag - 0.5)
        dis_y = mid_y + pos_neg * gap_y
        x_start = int(dis_x - stone_len // 2)
        x_end = x_start + stone_len
        y_start = int(start_y + dis_y - stone_width // 2)
        y_end = y_start + stone_width

        heights = (
            np.tile(
                np.linspace(-incline_height_grid, incline_height_grid, stone_width),
                (stone_len, 1),
            )
            * pos_neg
        ).astype(np.int16)
        _set_region(terrain, x_start, x_end, y_start, y_end, heights)
        goals[i + 1] = [dis_x, start_y + dis_y]
        left_right_flag = 1 - left_right_flag

    final_dis_x = int(dis_x + gap_x)
    goals[-1] = [final_dis_x, start_y + mid_y]
    return terrain, goals, final_dis_x


def hurdle(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    hurdle_range: Tuple[float, float] = (0.1, 0.2),
    hurdle_height_range: Tuple[float, float] = (0.05, 0.15),
    flat_size: float = 0.6,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create repeated transverse hurdles across the walking corridor."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    per_x = (round(length_x / terrain.horizontal_scale) - round(platform_size)) // num_goals
    hurdle_size = round(
        ((hurdle_range[1] - hurdle_range[0]) * difficulty + hurdle_range[0])
        / terrain.horizontal_scale
    )
    hurdle_height = round(
        (
            (hurdle_height_range[1] - hurdle_height_range[0]) * difficulty
            + hurdle_height_range[0]
        )
        / terrain.vertical_scale
    )
    platform_grid = round(platform_size / terrain.horizontal_scale)
    _set_region(
        terrain,
        start_x,
        start_x + round(length_x / terrain.horizontal_scale),
        start_y,
        start_y + mid_y * 2,
        0,
    )

    flat_grid = round(flat_size / terrain.horizontal_scale)
    dis_x = start_x + platform_grid
    for i in range(num_goals):
        goals[i] = [dis_x + per_x * i, start_y + mid_y]
    for _ in range(num_goals):
        _set_region(
            terrain,
            dis_x - hurdle_size // 2,
            dis_x + hurdle_size // 2,
            start_y,
            start_y + mid_y * 2,
            hurdle_height,
        )
        dis_x += flat_grid + hurdle_size
    return terrain, goals, int(dis_x)


def bridge(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    bridge_width_range: Tuple[float, float] = (0.3, 0.4),
    bridge_height: float = 0.7,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create a narrow bridge by lowering the terrain on both sides."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    bridge_width = round(
        (
            (bridge_width_range[1] - bridge_width_range[0]) * difficulty
            + bridge_width_range[0]
        )
        / terrain.horizontal_scale
    )
    bridge_height_raw = round(bridge_height / terrain.vertical_scale)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    bridge_start_x = start_x + platform_grid
    bridge_length = round(length_x / terrain.horizontal_scale)
    bridge_end_x = start_x + bridge_length

    _set_region(terrain, start_x, start_x + platform_grid, start_y, start_y + mid_y * 2, 0)
    for i in range(num_goals):
        goals[i] = [bridge_start_x + bridge_length / num_goals * i, start_y + mid_y]

    left_y2 = int(start_y + mid_y - bridge_width // 2)
    right_y1 = int(start_y + mid_y + bridge_width // 2)
    _set_region(terrain, bridge_start_x, bridge_end_x, start_y, left_y2, -bridge_height_raw)
    _set_region(
        terrain,
        bridge_start_x,
        bridge_end_x,
        right_y1,
        start_y + mid_y * 2,
        -bridge_height_raw,
    )
    return terrain, goals, bridge_end_x


def flat(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create a flat tile while preserving evenly spaced goal metadata."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    length_x_grid = round(length_x / terrain.horizontal_scale)
    length_y_grid = round(length_y / terrain.horizontal_scale)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    for i in range(num_goals):
        goals[i] = [
            start_x + platform_grid + length_x_grid / num_goals * i,
            start_y + length_y_grid // 2,
        ]
    return terrain, goals, start_x + length_x_grid


def uneven(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    num_range: Tuple[int, int] = (150, 200),
    size_range: Tuple[float, float] = (0.4, 0.7),
    height_range: Tuple[float, float] = (0.1, 0.2),
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create blocky uneven terrain from random rectangular height patches."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    per_x = (round(length_x / terrain.horizontal_scale) - platform_grid) // num_goals
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    for i in range(num_goals):
        goals[i] = [start_x + platform_grid + per_x * i, start_y + mid_y]

    height = round(
        ((height_range[1] - height_range[0]) * difficulty + height_range[0])
        / terrain.vertical_scale
    )
    min_size = round(size_range[0] / terrain.horizontal_scale)
    max_size = max(min_size, round(size_range[1] / terrain.horizontal_scale))
    discrete_start_x = start_x + platform_grid
    discrete_start_y = start_y
    discrete_end_x = discrete_start_x + round(length_x / terrain.horizontal_scale) - platform_grid
    discrete_end_y = discrete_start_y + round(length_y / terrain.horizontal_scale)
    num_rects = round((num_range[1] - num_range[0]) * difficulty + num_range[0])

    for _ in range(num_rects):
        width = random.randint(min_size, max_size)
        length = random.randint(min_size, max_size)
        if discrete_end_x - width <= discrete_start_x or discrete_end_y - length <= discrete_start_y:
            continue
        rect_x = random.randint(discrete_start_x, discrete_end_x - width)
        rect_y = random.randint(discrete_start_y, discrete_end_y - length)
        value = random.randint(-height // 2, height)
        _set_region(terrain, rect_x, rect_x + width, rect_y, rect_y + length, value)

    _set_region(terrain, start_x, start_x + platform_grid, start_y, start_y + mid_y * 2, 0)
    _set_region(
        terrain,
        discrete_end_x,
        discrete_end_x + platform_grid,
        start_y,
        start_y + mid_y * 2,
        0,
    )
    return terrain, goals, discrete_end_x + platform_grid


def stair(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    height_range: Tuple[float, float] = (0.08, 0.2),
    size_range: Tuple[float, float] = (0.4, 0.5),
    upstair: bool = True,
    start_z: float = 3.0,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create a staircase with difficulty-controlled step height and depth."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    per_x = (round(length_x / terrain.horizontal_scale) - platform_grid) // num_goals
    per_y = round(length_y / terrain.horizontal_scale) // 2
    step_height = round(
        ((height_range[1] - height_range[0]) * difficulty + height_range[0])
        / terrain.vertical_scale
    )
    step_x = round(
        ((size_range[0] - size_range[1]) * difficulty + size_range[1])
        / terrain.horizontal_scale
    )
    total_step_height = 0 if upstair else round(start_z / terrain.vertical_scale)
    dis_x = start_x + platform_grid

    for i in range(num_goals):
        goals[i] = [dis_x + per_x * i, start_y + per_y]
    for _ in range(num_goals):
        total_step_height += step_height if upstair else -step_height
        _set_region(
            terrain, dis_x, dis_x + step_x, start_y, start_y + per_y * 2, total_step_height
        )
        dis_x += step_x

    _set_region(
        terrain,
        dis_x,
        start_x + round(length_x / terrain.horizontal_scale),
        start_y,
        start_y + per_y * 2,
        total_step_height,
    )
    return terrain, goals, start_x + round(length_x / terrain.horizontal_scale)


def wave(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    amplitude_range: Tuple[float, float] = (0.05, 0.1),
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create a sinusoidal wave along the x direction."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    platform_grid = round(1.5 / terrain.horizontal_scale)
    mid_x = (round(length_x / terrain.horizontal_scale) - platform_grid) // num_goals
    for i in range(num_goals):
        goals[i] = [start_x + platform_grid + mid_x * i, start_y + mid_y]

    x_indices = np.arange(start_x, start_x + mid_x * num_goals + platform_grid)
    amplitude = round(
        ((amplitude_range[1] - amplitude_range[0]) * difficulty + amplitude_range[0])
        / terrain.vertical_scale
    )
    wave_pattern = amplitude * np.sin(2.0 * np.pi * x_indices / max(length_x, 1e-6))
    for offset, wave_height in enumerate(wave_pattern):
        x = int(x_indices[offset])
        _set_region(terrain, x, x + 1, start_y, start_y + mid_y * 2, round(wave_height))

    _set_region(terrain, start_x, start_x + platform_grid, start_y, start_y + mid_y * 2, 0)
    return terrain, goals, start_x + mid_x * num_goals


def slope(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    angle_range: Tuple[float, float] = (4.1, 10.0),
    uphill: bool = False,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create a linear ramp using angle-based slope height."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    length_x_grid = round((length_x - platform_size) / terrain.horizontal_scale)
    length_y_grid = round(length_y / terrain.horizontal_scale)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    for i in range(num_goals):
        goals[i] = [
            start_x + platform_grid + length_x_grid / num_goals * i,
            start_y + length_y_grid // 2,
        ]

    slope_angle = (angle_range[1] - angle_range[0]) * difficulty + angle_range[0]
    total_height_units = length_x * math.tan(math.radians(slope_angle)) / terrain.vertical_scale
    ramp_start = start_x + platform_grid
    for x in range(ramp_start, ramp_start + length_x_grid):
        progress = (x - ramp_start) / max(length_x_grid, 1)
        height = progress * total_height_units if uphill else (1.0 - progress) * total_height_units
        _set_region(terrain, x, x + 1, start_y, start_y + length_y_grid, round(height))
    return terrain, goals, ramp_start + length_x_grid


def gap(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    gap_height: float = 2.0,
    gap_low_range: Tuple[float, float] = (0.15, 0.3),
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create repeated gaps by carving deep negative-height strips."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    mid_x = round((length_x - platform_size) / terrain.horizontal_scale) // num_goals
    platform_grid = round(platform_size / terrain.horizontal_scale)
    for i in range(num_goals):
        goals[i] = [start_x + platform_grid + mid_x * i, start_y + mid_y]

    gap_size = round(
        ((gap_low_range[0] - gap_low_range[1]) * difficulty + gap_low_range[1])
        / terrain.horizontal_scale
    )
    gap_dis_x = start_x + platform_grid + gap_size
    gap_depth = -round(gap_height / terrain.vertical_scale)
    for _ in range(num_goals):
        _set_region(
            terrain,
            gap_dis_x,
            gap_dis_x + gap_size,
            start_y,
            start_y + mid_y * 2,
            gap_depth,
        )
        gap_dis_x += 3 * gap_size

    _set_region(terrain, start_x, start_x + platform_grid, start_y, start_y + mid_y * 2, 0)
    return terrain, goals, start_x + mid_x * num_goals


def plot(
    terrain: SubTerrain,
    length_x: float = 18.0,
    length_y: float = 4.0,
    num_goals: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    platform_size: float = 1.0,
    difficulty: float = 0.5,
    hurdle_range: Tuple[float, float] = (0.1, 0.15),
    hurdle_height: float = 1.2,
    flat_size: float = 1.0,
) -> Tuple[SubTerrain, np.ndarray, int]:
    """Create compact raised square obstacles for plotting/diagnostics."""
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    mid_y = round(length_y / terrain.horizontal_scale) // 2
    per_x = (round(length_x / terrain.horizontal_scale) - round(platform_size)) // num_goals
    hurdle_size = (
        round(
            ((hurdle_range[1] - hurdle_range[0]) * difficulty + hurdle_range[0])
            / terrain.horizontal_scale
        )
        // 2
    )
    hurdle_height_raw = round(hurdle_height / terrain.vertical_scale)
    platform_grid = round(platform_size / terrain.horizontal_scale)
    _set_region(
        terrain,
        start_x,
        start_x + round(length_x / terrain.horizontal_scale),
        start_y,
        start_y + mid_y * 2,
        0,
    )

    flat_grid = round(flat_size / terrain.horizontal_scale)
    dis_x = start_x + platform_grid
    for i in range(num_goals):
        goals[i] = [dis_x + per_x * i, start_y + mid_y]
    for _ in range(num_goals):
        _set_region(
            terrain,
            dis_x - hurdle_size,
            dis_x + hurdle_size,
            start_y + mid_y - hurdle_size,
            start_y + mid_y + hurdle_size,
            hurdle_height_raw,
        )
        dis_x += flat_grid + hurdle_size * 2
    return terrain, goals, int(dis_x)


class single_terrain:
    """Backward-compatible namespace matching the original HTB class name."""

    parkour = staticmethod(parkour)
    hurdle = staticmethod(hurdle)
    bridge = staticmethod(bridge)
    flat = staticmethod(flat)
    uneven = staticmethod(uneven)
    stair = staticmethod(stair)
    wave = staticmethod(wave)
    slope = staticmethod(slope)
    gap = staticmethod(gap)
    plot = staticmethod(plot)
