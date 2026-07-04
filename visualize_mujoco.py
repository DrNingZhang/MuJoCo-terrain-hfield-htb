"""Open an exported MuJoCo hfield XML in the interactive MuJoCo viewer."""

import argparse
from pathlib import Path


def visualize_mjxml(xml_path: str | Path) -> None:
    """Load an MJCF file and start the passive MuJoCo viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. Run: pip install mujoco") from exc

    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def parse_args() -> argparse.Namespace:
    """Parse visualization CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo terrain XML.")
    parser.add_argument("--xml", default="generated/htb_terrain.xml", help="Path to MJCF XML.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    visualize_mjxml(parse_args().xml)


if __name__ == "__main__":
    main()
