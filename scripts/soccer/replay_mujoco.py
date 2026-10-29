"""Replay a recorded Isaac Sim robot + ball trajectory in MuJoCo and render it offscreen.

Isaac Sim's RTX renderer crashes on this cluster's nodes, so recordings from
``check_foot_ball_contact.py --record`` or ``record_walk.py`` (no ball; the camera follows the robot) are posed frame by frame in Asimov's MJCF (kinematics only, no physics) and
rendered with MuJoCo's EGL backend to a GIF plus key-frame PNGs.

    MUJOCO_GL=egl python scripts/soccer/replay_mujoco.py outputs/soccer_checks/soccer_r0.05.npz
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mj_scene import build_spec  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording", help=".npz written by check_foot_ball_contact.py --record")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--azimuth", type=float, default=None, help="Camera azimuth in degrees (default: side-on).")
parser.add_argument("--distance", type=float, default=None, help="Camera distance (default 1.4 m, 2.5 m tracking).")
parser.add_argument("--track", action="store_true", help="Camera follows the pelvis (default without a ball).")
parser.add_argument("--slow", type=float, default=1.0, help="Play the GIF this many times slower than real time.")
args = parser.parse_args()

rec = np.load(args.recording)
has_ball = "ball_pos" in rec.files
track = args.track or not has_ball
joint_names = [str(n) for n in rec["joint_names"]]
body_names = [str(n) for n in rec["body_names"]]

# Asimov MJCF plus the soccer foot boxes (translucent green), the ball, and a red marker at the first contact.
robot = str(rec["robot"]) if "robot" in rec.files else os.path.basename(args.recording).split("_")[0]
spec = build_spec(foot_boxes=robot == "soccer", ball_radius=float(rec["ball_radius"]) if has_ball else None)
contact = rec["contact_point"] if has_ball else np.full(3, np.nan)
if np.all(np.isfinite(contact)):
    spec.worldbody.add_site(pos=contact, size=[0.012, 0, 0], rgba=[1, 0, 0, 1])
model = spec.compile()
data = mujoco.MjData(model)

joint_adr = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]
joint_ref = np.array([model.qpos0[a] for a in joint_adr])
ball_adr = model.jnt_qposadr[model.body("ball").jntadr[0]] if has_ball else None
mj_bodies = [(i, model.body(n).id) for i, n in enumerate(body_names) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]


def pose(frame: int, joint_offset: np.ndarray) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = rec["root_pos"][frame]
    data.qpos[3:7] = rec["root_quat"][frame]  # Isaac Lab and MuJoCo both use (w, x, y, z)
    data.qpos[joint_adr] = rec["joint_pos"][frame] + joint_offset
    if has_ball:
        data.qpos[ball_adr : ball_adr + 3] = rec["ball_pos"][frame]
        data.qpos[ball_adr + 3 : ball_adr + 7] = rec["ball_quat"][frame]
    mujoco.mj_kinematics(model, data)


def fk_error(joint_offset: np.ndarray) -> float:
    errors = []
    for frame in (0, len(rec["root_pos"]) // 2, len(rec["root_pos"]) - 1):
        pose(frame, joint_offset)
        errors += [np.linalg.norm(data.xpos[j] - rec["body_pos"][frame][i]) for i, j in mj_bodies]
    return float(np.max(errors))


# Isaac's joint angles are measured from the URDF zero pose; MJCF joints with a nonzero `ref` may or may not share it.
# Pick whichever mapping reproduces Isaac's body positions, and report the match.
candidates = {"q": np.zeros_like(joint_ref), "q + ref": joint_ref}
errors = {k: fk_error(v) for k, v in candidates.items()}
mapping = min(errors, key=errors.get)
print(f"[replay] max body position mismatch vs Isaac: {errors} -> using {mapping}")
offset = candidates[mapping]

foot = model.body("left_ankle_roll_link").id
pose(0, offset)
lookat = contact if np.all(np.isfinite(contact)) else data.xpos[foot].copy()
camera = mujoco.MjvCamera()
camera.lookat[:] = lookat + np.array([0.0, 0.0, 0.05])
camera.distance = args.distance if args.distance is not None else (2.5 if track else 1.4)
camera.elevation = -15.0
if args.azimuth is None and not has_ball:
    camera.azimuth = 135.0
elif args.azimuth is None:
    # Side-on to the ball's path.
    path = rec["ball_pos"][-1] - rec["ball_pos"][0]
    camera.azimuth = np.degrees(np.arctan2(path[1], path[0])) + 90.0
else:
    camera.azimuth = args.azimuth

renderer = mujoco.Renderer(model, height=args.height, width=args.width)
option = mujoco.MjvOption()
option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
frames = []
for frame in range(len(rec["root_pos"])):
    pose(frame, offset)
    if track:
        camera.lookat[:] = data.xpos[model.body("pelvis_link").id] - np.array([0.0, 0.0, 0.2])
    renderer.update_scene(data, camera=camera, scene_option=option)
    frames.append(renderer.render())

frame_dt = float(rec["frame_dt"])
base = os.path.splitext(args.recording)[0] + "_mujoco"
imageio.mimsave(f"{base}.gif", frames, duration=frame_dt * 1000 * args.slow, loop=0)
contact_t = float(rec["contact_t"]) if has_ball else -1.0
hit = int(contact_t / frame_dt) if contact_t >= 0 else len(frames) // 2
for label, k in (("before", hit - 5), ("contact", hit), ("after", hit + 10)):
    imageio.imwrite(f"{base}_{label}.png", frames[max(0, min(k, len(frames) - 1))])
print(f"[replay] wrote {base}.gif and {base}_before/_contact/_after.png ({len(frames)} frames)")
