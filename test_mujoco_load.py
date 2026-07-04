"""Validate generated MuJoCo hfield outputs.

The script checks the binary hfield header, file size, metadata shape, and
whether MuJoCo can load the generated XML.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_hfield_header(path: Path) -> tuple:
    """Read the int32 nrow/ncol header from an exported hfield file."""
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=2)
    if header.size != 2:
        raise RuntimeError(f"Could not read nrow/ncol header from {path}")
    return int(header[0]), int(header[1])


def parse_args() -> argparse.Namespace:
    """Parse validation CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate generated MuJoCo hfield XML.")
    parser.add_argument("--out", default="generated", help="Output directory to validate.")
    return parser.parse_args()


def main() -> None:
    """Run hfield consistency checks and MuJoCo XML loading."""
    args = parse_args()
    out_dir = Path(args.out)
    xml_path = out_dir / "htb_terrain.xml"
    hfield_path = out_dir / "htb_terrain.hfield"
    metadata_path = out_dir / "terrain_metadata.json"

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")
    if not hfield_path.exists():
        raise FileNotFoundError(f"Missing hfield: {hfield_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nrow, ncol = read_hfield_header(hfield_path)
    expected_size = 4 * (2 + nrow * ncol)
    actual_size = hfield_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Invalid hfield size: got {actual_size}, expected {expected_size}"
        )
    if list(metadata.get("shape", [])) != [nrow, ncol]:
        raise RuntimeError(
            f"Metadata shape {metadata.get('shape')} does not match hfield header {[nrow, ncol]}"
        )

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. Run: pip install mujoco") from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nhfield < 1:
        raise RuntimeError(f"Expected at least one hfield, got model.nhfield={model.nhfield}")
    if model.ngeom < 1:
        raise RuntimeError(f"Expected at least one geom, got model.ngeom={model.ngeom}")

    print(f"Loaded XML: {xml_path}")
    print(f"model.nhfield: {model.nhfield}")
    print(f"model.ngeom: {model.ngeom}")
    print(f"hfield header nrow/ncol: {nrow}/{ncol}")
    print(f"hfield bytes: {actual_size}")
    if hasattr(model, "hfield_nrow") and hasattr(model, "hfield_ncol"):
        print(f"model hfield_nrow: {model.hfield_nrow[:model.nhfield].tolist()}")
        print(f"model hfield_ncol: {model.hfield_ncol[:model.nhfield].tolist()}")
    if hasattr(model, "hfield_size"):
        print(f"model hfield_size: {model.hfield_size[:model.nhfield].tolist()}")


if __name__ == "__main__":
    main()
