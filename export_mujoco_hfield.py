"""Export HTB-derived terrain height fields as MuJoCo hfield assets.

The exporter keeps the terrain data as a MuJoCo hfield instead of converting
the full grid to a mesh. MuJoCo stores hfield elevation values normalized to
[0, 1], while the XML hfield size and geom position restore the metric height
range at load time.
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from combine_config import TERRAIN_NAME_TO_ID
from terrain import Terrain
from terrain_config import TerrainConfig


EPSILON = 1e-6


def json_ready(value: Any) -> Any:
    """Convert numpy containers/scalars into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def apply_orientation(
    height_field_raw: np.ndarray, transpose: bool, flip_x: bool, flip_y: bool
) -> np.ndarray:
    """Apply export-time orientation fixes for matching MuJoCo and preview views."""
    oriented = height_field_raw
    # These switches are intentionally explicit so users can correct viewer
    # orientation without changing the terrain generator itself.
    if transpose:
        oriented = oriented.T
    if flip_x:
        oriented = np.flip(oriented, axis=0)
    if flip_y:
        oriented = np.flip(oriented, axis=1)
    return np.ascontiguousarray(oriented.astype(np.int16, copy=False))


def write_hfield(path: Path, hfield_data: np.ndarray) -> Tuple[int, int, int]:
    """Write MuJoCo hfield binary data and return rows, cols, and byte size."""
    nrow, ncol = hfield_data.shape
    with path.open("wb") as handle:
        # MuJoCo hfield binary format used here:
        # int32 nrow, int32 ncol, then row-major float32 elevation[nrow*ncol].
        np.asarray([nrow, ncol], dtype="<i4").tofile(handle)
        np.asarray(hfield_data, dtype="<f4").ravel(order="C").tofile(handle)
    return nrow, ncol, path.stat().st_size


def compute_light_height(full_x: float, full_y: float, h_range: float) -> float:
    """Choose a light height that scales with the exported terrain footprint."""
    return max(8.0, 0.35 * max(full_x, full_y), h_range + 6.0)


def write_xml(
    path: Path,
    hfield_file: str,
    full_x: float,
    full_y: float,
    h_range: float,
    base_z: float,
    h_min: float,
    light_height: float,
    add_test_ball: bool = False,
) -> None:
    """Write a minimal MJCF scene containing the terrain hfield.

    The terrain geom is centered at full_x/2, full_y/2 because MuJoCo hfield
    size stores half extents. The z position is h_min so normalized hfield
    elevations recover the original metric height range.
    """
    test_ball_xml = ""
    if add_test_ball:
        test_ball_xml = """    <body name="test_ball" pos="2 2 2">
      <freejoint/>
      <geom type="sphere" size="0.15" mass="1"/>
    </body>
"""

    xml = f'''<mujoco model="htb_terrain">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="htb_terrain" file="{hfield_file}" size="{full_x / 2.0:.9g} {full_y / 2.0:.9g} {h_range:.9g} {base_z:.9g}"/>
  </asset>
  <worldbody>
    <light name="light_corner_00" pos="0 0 {light_height:.9g}" dir="1 1 -2" diffuse="0.8 0.8 0.8"/>
    <light name="light_corner_10" pos="{full_x:.9g} 0 {light_height:.9g}" dir="-1 1 -2" diffuse="0.8 0.8 0.8"/>
    <light name="light_corner_01" pos="0 {full_y:.9g} {light_height:.9g}" dir="1 -1 -2" diffuse="0.8 0.8 0.8"/>
    <light name="light_corner_11" pos="{full_x:.9g} {full_y:.9g} {light_height:.9g}" dir="-1 -1 -2" diffuse="0.8 0.8 0.8"/>
    <geom name="terrain" type="hfield" hfield="htb_terrain" pos="{full_x / 2.0:.9g} {full_y / 2.0:.9g} {h_min:.9g}" friction="1.0 0.005 0.0001"/>
{test_ball_xml.rstrip()}
  </worldbody>
</mujoco>
'''
    path.write_text(xml, encoding="utf-8")


def write_preview(path: Path, height_m: np.ndarray, full_x: float, full_y: float) -> None:
    """Render a static PNG preview of the metric height field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    image = ax.imshow(
        height_m.T,
        origin="lower",
        extent=[0.0, full_x, 0.0, full_y],
        aspect="auto",
        cmap="terrain",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("HTB MuJoCo hfield height [m]")
    fig.colorbar(image, ax=ax, label="height [m]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_config(args: argparse.Namespace) -> TerrainConfig:
    """Build a terrain generator config from parsed CLI arguments."""
    rows = 1 if args.single else args.rows
    cols = 1 if args.single else args.cols
    difficulty = args.difficulty
    if args.single and difficulty is None:
        difficulty = 0.5
    return TerrainConfig(
        num_rows=rows,
        num_cols=cols,
        horizontal_scale=args.horizontal_scale,
        vertical_scale=args.vertical_scale,
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
        selected_terrain=args.terrain,
        forced_difficulty=difficulty,
    )


def export(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate terrain, write MuJoCo hfield artifacts, and return metadata."""
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    terrain = Terrain(cfg)
    raw = apply_orientation(terrain.height_field_raw, args.transpose, args.flip_x, args.flip_y)
    # Raw HTB heights are int16 units; vertical_scale converts each unit to meters.
    height_m = raw.astype(np.float64) * cfg.vertical_scale

    h_min = float(height_m.min())
    h_max = float(height_m.max())
    h_range = max(h_max - h_min, EPSILON)
    # MuJoCo hfield data is normalized; XML size[2] stores the metric height span.
    hfield_data = ((height_m - h_min) / h_range).astype(np.float32)

    nrow, ncol = height_m.shape
    # With grid spacing s, samples cover world coordinates x=[0, full_x],
    # y=[0, full_y]. MuJoCo hfield size stores half extents.
    full_x = float((nrow - 1) * cfg.horizontal_scale)
    full_y = float((ncol - 1) * cfg.horizontal_scale)
    light_height = compute_light_height(full_x, full_y, h_range)

    hfield_path = out_dir / "htb_terrain.hfield"
    xml_path = out_dir / "htb_terrain.xml"
    raw_path = out_dir / "terrain_height_raw.npy"
    meters_path = out_dir / "terrain_height_m.npy"
    metadata_path = out_dir / "terrain_metadata.json"
    preview_path = out_dir / "preview.png"

    _nrow, _ncol, hfield_size = write_hfield(hfield_path, hfield_data)
    write_xml(
        xml_path,
        hfield_file=hfield_path.name,
        full_x=full_x,
        full_y=full_y,
        h_range=h_range,
        base_z=args.base_z,
        # geom pos z uses h_min so normalized elevation 0 lands at the original minimum.
        h_min=h_min,
        light_height=light_height,
        add_test_ball=args.add_test_ball,
    )
    np.save(raw_path, raw)
    np.save(meters_path, height_m)
    if args.preview:
        write_preview(preview_path, height_m, full_x, full_y)

    metadata = {
        "format": "mujoco_hfield",
        "seed": args.seed,
        "shape": [nrow, ncol],
        "horizontal_scale": cfg.horizontal_scale,
        "vertical_scale": cfg.vertical_scale,
        "full_x": full_x,
        "full_y": full_y,
        "h_min": h_min,
        "h_max": h_max,
        "h_range": h_range,
        "light_height": light_height,
        "base_z": args.base_z,
        "add_test_ball": args.add_test_ball,
        "hfield_file_size": hfield_size,
        "expected_hfield_file_size": 4 * (2 + nrow * ncol),
        "orientation": {
            "transpose": args.transpose,
            "flip_x": args.flip_x,
            "flip_y": args.flip_y,
        },
        "terrain": {
            "selected": args.terrain,
            "single": args.single,
            "difficulty": cfg.forced_difficulty,
            "available": sorted(TERRAIN_NAME_TO_ID.keys()),
        },
        "config": asdict(cfg),
        "env_origins": terrain.env_origins,
        "goals": terrain.goals,
        "terrain_type": terrain.terrain_type,
        "files": {
            "hfield": str(hfield_path),
            "xml": str(xml_path),
            "height_raw": str(raw_path),
            "height_m": str(meters_path),
            "metadata": str(metadata_path),
            "preview": str(preview_path) if args.preview else None,
        },
    }
    metadata_path.write_text(
        json.dumps(json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Humanoid-Terrain-Bench terrain as a MuJoCo hfield."
    )
    parser.add_argument("--out", default="generated", help="Output directory.")
    parser.add_argument("--rows", type=int, default=10, help="Number of terrain rows.")
    parser.add_argument("--cols", type=int, default=20, help="Number of terrain columns.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--preview", action="store_true", help="Write preview.png.")
    parser.add_argument(
        "--terrain",
        choices=sorted(TERRAIN_NAME_TO_ID.keys()),
        default=None,
        help="Force all tiles to one terrain type.",
    )
    parser.add_argument("--difficulty", type=float, default=None, help="Force difficulty [0, 1].")
    parser.add_argument("--single", action="store_true", help="Generate a single 1x1 tile.")
    parser.add_argument("--horizontal-scale", type=float, default=0.05, help="Grid spacing [m].")
    parser.add_argument("--vertical-scale", type=float, default=0.005, help="Raw height unit [m].")
    parser.add_argument("--terrain-length", type=float, default=10.0, help="Tile length [m].")
    parser.add_argument("--terrain-width", type=float, default=4.0, help="Tile width [m].")
    parser.add_argument("--base-z", type=float, default=0.05, help="MuJoCo hfield base size.")
    parser.add_argument("--transpose", action="store_true", help="Transpose height field.")
    parser.add_argument("--flip-x", action="store_true", help="Flip x/row direction.")
    parser.add_argument("--flip-y", action="store_true", help="Flip y/col direction.")
    parser.add_argument(
        "--add-test-ball",
        action="store_true",
        help="Add a free sphere for quick hfield collision checks.",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.horizontal_scale <= 0.0 or args.vertical_scale <= 0.0:
        parser.error("--horizontal-scale and --vertical-scale must be positive")
    if args.terrain_length <= 0.0 or args.terrain_width <= 0.0:
        parser.error("--terrain-length and --terrain-width must be positive")
    if args.difficulty is not None and not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0, 1]")
    return args


def main() -> None:
    """Command-line entry point."""
    metadata = export(parse_args())
    print(f"Wrote {metadata['files']['hfield']}")
    print(f"Wrote {metadata['files']['xml']}")
    print(f"height shape: {metadata['shape']}")
    print(f"hfield bytes: {metadata['hfield_file_size']}")


if __name__ == "__main__":
    main()
