"""MuJoCo scene helpers shared by the soccer scripts that run outside Isaac Sim.

Importing cyclotron.assets pulls in Isaac Lab, which needs Isaac Sim's `pxr`, so the robot model path and the soccer
foot-box constants are read from the source instead.
"""

import ast
import os
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
ASIMOV_1_MODEL_DIR = os.environ.get("ASIMOV_1_MODEL_DIR", str(REPO_ROOT / "third_party" / "asimov-1" / "sim-model"))
_SOCCER_SRC = REPO_ROOT / "source" / "cyclotron" / "cyclotron" / "assets" / "robots" / "asimov_1_soccer.py"
_CONSTANTS = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in ast.parse(_SOCCER_SRC.read_text()).body
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in ("FOOT_BOX_CENTER", "FOOT_BOX_SIZE", "FOOT_LINKS", "FOOT_MEDIAL_PATCH")
}
FOOT_BOX_CENTER = _CONSTANTS["FOOT_BOX_CENTER"]
FOOT_BOX_SIZE = _CONSTANTS["FOOT_BOX_SIZE"]
FOOT_LINKS = _CONSTANTS["FOOT_LINKS"]
FOOT_MEDIAL_PATCH = _CONSTANTS["FOOT_MEDIAL_PATCH"]


def build_spec(foot_boxes: bool = True, ball_radius: float | None = None) -> mujoco.MjSpec:
    """Asimov MJCF, optionally with the soccer foot boxes (translucent green) and a free-floating orange ball."""
    spec = mujoco.MjSpec.from_file(os.path.join(ASIMOV_1_MODEL_DIR, "xmls", "asimov_1.xml"))
    for link in FOOT_LINKS if foot_boxes else ():
        spec.body(link).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=FOOT_BOX_CENTER, size=[s / 2 for s in FOOT_BOX_SIZE],
            rgba=[0.2, 0.9, 0.3, 0.35], contype=0, conaffinity=0, group=1,
        )
    if ball_radius is not None:
        ball = spec.worldbody.add_body(name="ball")
        ball.add_freejoint()
        ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0],
                      rgba=[1.0, 0.55, 0.1, 1.0], contype=0, conaffinity=0, group=1)
    return spec
