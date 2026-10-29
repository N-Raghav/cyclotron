# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 and ball configurations for soccer tasks.

The stock URDF gives each foot only four 5 mm sole spheres, so a ball passes straight through the side of the foot.
The soccer robot is spawned from a copy of the URDF with a box collider added to each foot, written to a cache
directory so the upstream model stays untouched.
"""

import os
import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from .asimov_1 import ASIMOV_1_DELAYED_CFG, ASIMOV_1_URDF_PATH

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")

# Foot box in the ankle_roll_link frame. Lengthwise it stays within the sole spheres (x -0.045..0.122, r=0.005) and
# its bottom sits 8 mm above them, so the box never reaches the ground before a sole sphere does when the foot pitches.
# Width follows the visual mesh (ANKLE_B_NEW.STL: y +-0.038). The top stops at the ankle_roll origin, not the mesh top
# (z=0.023): the knee link's lowest collision sphere (r=0.04) reaches down to z=+0.022 here, and since knee and foot
# aren't parent and child the two would collide, the foot constantly pushed by its own shin.
FOOT_BOX_CENTER = (0.0385, 0.0, -0.013)
FOOT_BOX_SIZE = (0.177, 0.076, 0.026)
# Arch of the right foot's inside face (+y in its link frame), at the box's mid-height; mirror y for the left foot.
FOOT_MEDIAL_PATCH = (0.04, 0.038, -0.013)

BALL_RADIUS = 0.11
BALL_MASS = 0.43


def _foot_box_collision(center, size) -> str:
    xyz = " ".join(f"{v:g}" for v in center)
    size = " ".join(f"{v:g}" for v in size)
    return (
        f'    <collision name="foot_box">\n'
        f'      <origin xyz="{xyz}" rpy="0 0 0"/>\n'
        f'      <geometry>\n        <box size="{size}"/>\n      </geometry>\n'
        f"    </collision>\n"
    )


def write_soccer_urdf(
    source: str = ASIMOV_1_URDF_PATH,
    box_center=FOOT_BOX_CENTER,
    box_size=FOOT_BOX_SIZE,
    name: str = "asimov_1_soccer",
) -> str:
    """Write the soccer URDF (foot box colliders, absolute mesh paths) and return its path."""
    text = Path(source).read_text()
    model_dir = Path(source).resolve().parent
    text = re.sub(r'filename="(?!/)([^"]+)"', lambda m: f'filename="{(model_dir / m.group(1)).resolve()}"', text)
    for link in FOOT_LINKS:
        pattern = re.compile(rf'(<link name="{link}">.*?)(  </link>)', re.DOTALL)
        text, count = pattern.subn(lambda m: m.group(1) + _foot_box_collision(box_center, box_size) + m.group(2), text)
        if count != 1:
            raise RuntimeError(f"Expected one <link name=\"{link}\"> in {source}, found {count}.")

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cyclotron"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.urdf"
    if not path.exists() or path.read_text() != text:
        # Write then rename, so parallel (distributed) workers never read a half-written file.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    return str(path)


ASIMOV_1_SOCCER_CFG = ASIMOV_1_DELAYED_CFG.replace(
    spawn=ASIMOV_1_DELAYED_CFG.spawn.replace(asset_path=write_soccer_urdf())
)

BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=BALL_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Stand-in for grass rolling resistance, which PhysX spheres don't have.
            linear_damping=0.3,
            angular_damping=0.3,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.6, restitution=0.5, restitution_combine_mode="max"
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, BALL_RADIUS)),
)
