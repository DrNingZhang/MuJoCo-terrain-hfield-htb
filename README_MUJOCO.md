# MuJoCo HTB Terrain HField Generator

`README.md` is the GitHub homepage for this project. This file is kept for
users who followed earlier conversion instructions and points to the same
terrain-only workflow.

Use:

```bash
pip install -r requirements.txt
python export_mujoco_hfield.py --out generated --rows 2 --cols 3 --seed 0 --preview
python test_mujoco_load.py --out generated
python visualize_mujoco.py --xml generated/htb_terrain.xml
```

The exporter creates MuJoCo hfield assets only. It does not include robot
models, controllers, rewards, training environments, or policies. The default
XML contains the hfield terrain and four auto-positioned visualization lights;
it has no robot and no test ball unless `--add-test-ball` is passed.

See `README.md` for the full option reference and release notes.
