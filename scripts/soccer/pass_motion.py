"""Inside-foot pass reference motion for Asimov: IK on the MuJoCo model along a target timeline, saved as an AMP clip.

Technique (right foot kicks, left foot plants, pass direction +x, ball at the origin):
  approach   0.00 s  last step before the plant, weight on the right foot
  plant      0.30 s  left foot flat beside the ball pointing at the target
  backswing  0.56 s  right foot raised behind the ball, turned 70 deg out (finishes turning on the way in)
  contact    0.76 s  inside-foot arch meets the middle of the ball
  follow     0.96 s  foot continues low along the target line
  right_back 1.22 s  right foot swings back along the same line, toes turning forward, and lands behind the plant foot
  shift      1.36 s  weight onto the right foot
  left_back  1.60 s  left foot steps back level with the right, at standing width
  steady     2.00 s  upright standing pose, weight shared, held
The arms hang relaxed and counter the kick: the left arm (opposite the kicking leg) swings out and a little forward,
the right arm a little back, both back at the sides once the robot is steady. Hands stay near hip height.
Foot targets are interpolated between these poses (left foot swings in an arc), the centre of mass is kept over the
support foot, the pelvis upright, the torso (and so the head: Asimov's neck is fixed) facing where the left foot
points, and joints off their limits. IK is solved every frame at 50 fps.

Outputs in --out_dir: pass_r<radius>.npz (same format as tasks/locomotion/motions/asimov_1_walking.npz),
pass_r<radius>.gif and pass_r<radius>_keyframes.png.

    MUJOCO_GL=egl python scripts/soccer/pass_motion.py --ball_radius 0.05
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mj_scene import FOOT_BOX_CENTER, FOOT_BOX_SIZE, FOOT_MEDIAL_PATCH, REPO_ROOT, build_spec  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ball_radius", type=float, default=0.05)
parser.add_argument("--ball_speed", type=float, default=1.5, help="Ball speed after contact, for the render only.")
parser.add_argument("--plant_gap", type=float, default=0.10, help="Ball surface to plant-foot inner edge (m).")
parser.add_argument("--plant_behind", type=float, default=0.03, help="Plant-foot centre behind the ball centre (m).")
parser.add_argument("--out_dir", default="outputs/soccer_pass")
parser.add_argument("--weight", action="append", default=[], metavar="TERM=W", help="Override an IK term weight.")
parser.add_argument("--tag", default="", help="Suffix for the output file names.")
args = parser.parse_args()

FPS = 50
WALK_CLIP = REPO_ROOT / "source" / "cyclotron" / "cyclotron" / "tasks" / "locomotion" / "motions" / "asimov_1_walking.npz"
KICK, PLANT, PELVIS, TORSO = "right_ankle_roll_link", "left_ankle_roll_link", "pelvis_link", "waist_yaw_link"
# Arch of the right foot's inside face (+y in its link frame), foot centre, and the foot-origin height with the sole on
# the ground (sole spheres at z=-0.029, r=0.005).
MEDIAL_PATCH = np.array(FOOT_MEDIAL_PATCH)
FOOT_CENTER = np.array([0.044, 0.0, 0.0])
SOLE_Z = 0.034
SWING_HEIGHT = 0.05
# Feet apart (y) in the standing pose, and the posture weight once the robot has recovered from the kick.
STANCE_WIDTH = 0.215
POSTURE_STAND = 1.0
# How far both feet step back after the kick, so the robot finishes square behind where the ball was.
STEP_BACK = 0.10
# Arm counter-swing at full extent, added to the standing arm pose: left arm out and forward, right arm back.
ARM_SWING = {"left_shoulder_pitch_joint": -0.30, "left_shoulder_roll_joint": -0.30,
             "right_shoulder_pitch_joint": -0.25, "right_shoulder_roll_joint": 0.15}
# Counter-swing amount (0..1) over time; zero means arms relaxed at the sides.
ARM_SWING_KEYS = {0.00: 0.0, 0.30: 0.4, 0.56: 1.0, 0.96: 1.0, 1.36: 0.3, 1.70: 0.0, 2.00: 0.0}
# Isaac Lab standing pose (ASIMOV_1_STANDING_INIT_STATE in assets/robots/asimov_1.py), used as the posture prior.
NOMINAL = {
    "left_hip_pitch_joint": -0.15, "right_hip_pitch_joint": 0.15,
    "left_knee_joint": 0.45, "right_knee_joint": -0.45,
    "left_shoulder_pitch_joint": -0.25, "right_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": -0.05, "right_shoulder_roll_joint": 0.05,
    "left_elbow_joint": 0.40, "right_elbow_joint": -0.40,
}

r = args.ball_radius
R_TOES_FORWARD = np.eye(3)
# Inside of the right foot (+y) faces the target (+x): toes point to -y.
R_INSIDE_TO_TARGET = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_KICK_STANCE = Rotation.from_euler("z", -45, degrees=True).as_matrix()
# With the leg drawn back, Asimov's hip yaw (+-45 deg) can't turn the foot fully out; it finishes turning on the way in.
R_KICK_BACKSWING = Rotation.from_euler("z", -70, degrees=True).as_matrix()
# Plant foot beside the ball with room for the kicking foot to swing through, its centre slightly behind the ball.
BOX_CENTER, BOX_HALF = np.array(FOOT_BOX_CENTER), np.array(FOOT_BOX_SIZE) / 2
PLANT_Y = r + args.plant_gap + BOX_HALF[1]
PLANT_POS = [-FOOT_CENTER[0] - args.plant_behind, PLANT_Y, SOLE_Z]


def kick_origin(patch_target, rot=R_INSIDE_TO_TARGET) -> np.ndarray:
    """Right-foot origin that puts the inside patch on `patch_target` with the foot at orientation `rot`."""
    return np.asarray(patch_target) - rot @ MEDIAL_PATCH


RIGHT_BACK_POS = [PLANT_POS[0] - STEP_BACK, PLANT_Y - STANCE_WIDTH, SOLE_Z]
LEFT_BACK_POS = [PLANT_POS[0] - STEP_BACK, PLANT_Y, SOLE_Z]

# name: (time, left-foot origin, left orientation, right-foot origin, right orientation, CoM weight on left foot)
KEYFRAMES = {
    "approach": (0.00, [-0.45 - args.plant_behind, PLANT_Y, SOLE_Z], R_TOES_FORWARD, [-0.35, -0.10, SOLE_Z], R_KICK_STANCE, 0.0),
    "plant": (0.30, PLANT_POS, R_TOES_FORWARD, [-0.35, -0.10, SOLE_Z], R_KICK_STANCE, 0.7),
    "backswing": (0.56, PLANT_POS, R_TOES_FORWARD,
                  kick_origin([-r - 0.25, 0.0, r + 0.04], R_KICK_BACKSWING), R_KICK_BACKSWING, 1.0),
    "contact": (0.76, PLANT_POS, R_TOES_FORWARD,
                kick_origin([-r, 0.0, r]), R_INSIDE_TO_TARGET, 1.0),
    "follow": (0.96, PLANT_POS, R_TOES_FORWARD,
               kick_origin([-r + 0.20, 0.0, r + 0.03]), R_INSIDE_TO_TARGET, 1.0),
    "right_back": (1.22, PLANT_POS, R_TOES_FORWARD, RIGHT_BACK_POS, R_TOES_FORWARD, 0.6),
    "shift": (1.36, PLANT_POS, R_TOES_FORWARD, RIGHT_BACK_POS, R_TOES_FORWARD, 0.0),
    "left_back": (1.60, LEFT_BACK_POS, R_TOES_FORWARD, RIGHT_BACK_POS, R_TOES_FORWARD, 0.0),
    "steady": (2.00, LEFT_BACK_POS, R_TOES_FORWARD, RIGHT_BACK_POS, R_TOES_FORWARD, 0.5),
}
T_CONTACT = KEYFRAMES["contact"][0]
key_t = np.array([k[0] for k in KEYFRAMES.values()])
left_pos = PchipInterpolator(key_t, np.array([k[1] for k in KEYFRAMES.values()]))
right_pos = PchipInterpolator(key_t, np.array([k[3] for k in KEYFRAMES.values()]))
left_rot = Slerp(key_t, Rotation.from_matrix(np.stack([k[2] for k in KEYFRAMES.values()])))
right_rot = Slerp(key_t, Rotation.from_matrix(np.stack([k[4] for k in KEYFRAMES.values()])))
left_weight = PchipInterpolator(key_t, np.array([k[5] for k in KEYFRAMES.values()]))
# Steps that lift a foot in an arc instead of sliding it: (foot, from keyframe, to keyframe).
SWINGS = (("left", "approach", "plant"), ("right", "follow", "right_back"), ("left", "shift", "left_back"))
arm_swing = PchipInterpolator(list(ARM_SWING_KEYS), list(ARM_SWING_KEYS.values()))


def targets(t: float) -> dict:
    pos = {"left": left_pos(t).copy(), "right": right_pos(t).copy()}
    for foot, k0, k1 in SWINGS:
        t0, t1 = KEYFRAMES[k0][0], KEYFRAMES[k1][0]
        if t0 < t < t1:
            pos[foot][2] += SWING_HEIGHT * np.sin(np.pi * (t - t0) / (t1 - t0))
    # Settle into the standing posture once both feet are back down.
    ramp = np.clip((t - KEYFRAMES["left_back"][0]) / (KEYFRAMES["steady"][0] - KEYFRAMES["left_back"][0]), 0.0, 1.0)
    return dict(left_pos=pos["left"], left_rot=left_rot(t).as_matrix(), right_pos=pos["right"],
                right_rot=right_rot(t).as_matrix(), w_left=float(np.clip(left_weight(t), 0.0, 1.0)),
                w_posture=WEIGHTS["posture"] + ramp * (POSTURE_STAND - WEIGHTS["posture"]),
                arms=arm_nominal + float(np.clip(arm_swing(t), 0.0, 1.0)) * arm_delta)


model = build_spec(foot_boxes=True, ball_radius=r).compile()
data = mujoco.MjData(model)
hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
hinge_names = [model.joint(j).name for j in hinges]
hinge_adr = np.array([model.jnt_qposadr[j] for j in hinges])
lo = np.where(model.jnt_limited[hinges], model.jnt_range[hinges, 0], -np.pi)
hi = np.where(model.jnt_limited[hinges], model.jnt_range[hinges, 1], np.pi)
LIMIT_MARGIN = 0.1 * (hi - lo)
nominal = np.array([NOMINAL.get(n, 0.0) for n in hinge_names])
robot_root_adr = model.jnt_qposadr[model.body(PELVIS).jntadr[0]]
ball_adr = model.jnt_qposadr[model.body("ball").jntadr[0]]
kick_id, plant_id, pelvis_id, torso_id = (model.body(n).id for n in (KICK, PLANT, PELVIS, TORSO))
# Upright is strong so balance comes from shifting the hips over the support foot, not leaning the torso (the
# torso is the pelvis plus waist yaw, so a pelvis tilt is a torso tilt).
WEIGHTS = {"left_pos": 40, "left_rot": 20, "right_pos": 40, "right_rot": 10, "flat": 20, "com": 40, "upright": 30, "posture": 0.1,
           "arms": 3, "facing": 5, "limits": 0.5}
WEIGHTS.update({k: float(v) for k, v in (w.split("=") for w in args.weight)})
# Arm joints follow their own targets (relaxed pose plus counter-swing), strongly enough that they don't drift up.
arm_idx = np.array([i for i, n in enumerate(hinge_names) if any(k in n for k in ("shoulder", "elbow", "wrist"))])
arm_nominal = nominal[arm_idx]
arm_delta = np.array([ARM_SWING.get(hinge_names[i], 0.0) for i in arm_idx])


def ball_pose(t: float) -> np.ndarray:
    d = max(0.0, t - T_CONTACT) * args.ball_speed
    quat = np.roll(Rotation.from_rotvec([0.0, d / r, 0.0]).as_quat(), 1)
    return np.concatenate([[d, 0.0, r], quat])


def set_pose(x: np.ndarray, t: float = 0.0) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[robot_root_adr : robot_root_adr + 3] = x[:3]
    data.qpos[robot_root_adr + 3 : robot_root_adr + 7] = np.roll(Rotation.from_euler("xyz", x[3:6]).as_quat(), 1)
    data.qpos[hinge_adr] = x[6:]
    data.qpos[ball_adr : ball_adr + 7] = ball_pose(t)
    mujoco.mj_fwdPosition(model, data)


def terms(x: np.ndarray, tg: dict) -> dict[str, np.ndarray]:
    set_pose(x)
    lp, lr = data.xpos[plant_id], data.xmat[plant_id].reshape(3, 3)
    rp, rr = data.xpos[kick_id], data.xmat[kick_id].reshape(3, 3)
    support = tg["w_left"] * (lp + lr @ FOOT_CENTER) + (1 - tg["w_left"]) * (rp + rr @ FOOT_CENTER)
    return {
        "left_pos": lp - tg["left_pos"],
        "left_rot": Rotation.from_matrix(tg["left_rot"].T @ lr).as_rotvec(),
        "right_pos": rp - tg["right_pos"],
        "right_rot": Rotation.from_matrix(tg["right_rot"].T @ rr).as_rotvec(),
        # Every foot target is level, and a tilted sole would dig into the ground.
        "flat": np.concatenate([lr[:2, 2], rr[:2, 2]]),
        "com": data.subtree_com[pelvis_id][:2] - support[:2],
        "upright": data.xmat[pelvis_id].reshape(3, 3)[:2, 2],
        # Head and torso face the way the plant foot points: yaw between their forward axes.
        "facing": np.array([yaw_between(data.xmat[torso_id].reshape(3, 3)[:, 0], lr[:, 0])]),
        "posture": x[6:] - nominal,
        "arms": x[6:][arm_idx] - tg["arms"],
        # Soft barrier: penalise using the last 10% of any joint's range.
        "limits": np.maximum(0.0, LIMIT_MARGIN - np.minimum(x[6:] - lo, hi - x[6:])) / LIMIT_MARGIN,
    }


def yaw_between(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from b to a in the horizontal plane."""
    return float(np.arctan2(b[0] * a[1] - b[1] * a[0], a[0] * b[0] + a[1] * b[1]))


def residuals(x: np.ndarray, tg: dict) -> np.ndarray:
    return np.concatenate([(tg["w_posture"] if k == "posture" else WEIGHTS[k]) * v for k, v in terms(x, tg).items()])


x_lo = np.concatenate([[-2, -2, 0.2, -0.4, -0.4, -np.pi], lo])
x_hi = np.concatenate([[2, 2, 0.9, 0.4, 0.4, np.pi], hi])
x = np.concatenate([[-0.25, 0.0, 0.6, 0.0, 0.0, -0.5], np.clip(nominal, lo, hi)])
times = np.arange(int(round(KEYFRAMES["steady"][0] * FPS)) + 1) / FPS
CORNERS = BOX_CENTER + BOX_HALF * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])


def box_distance(points: np.ndarray, body_id: int) -> float:
    """Distance from world points to a foot box (0 if inside)."""
    local = (points - data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3) - BOX_CENTER
    return float(np.min(np.linalg.norm(local - np.clip(local, -BOX_HALF, BOX_HALF), axis=-1)))


def clearances(t: float) -> tuple[float, float]:
    """Kick-foot box corners to the plant-foot box, and ball surface to the plant-foot box."""
    kick_corners = data.xpos[kick_id] + CORNERS @ data.xmat[kick_id].reshape(3, 3).T
    return box_distance(kick_corners, plant_id), box_distance(ball_pose(t)[None, :3], plant_id) - r


solution = []
worst = dict(pos=0.0, rot=0.0, com=0.0, feet_gap=(np.inf, 0.0), ball_gap=(np.inf, 0.0))
for t in times:
    tg = targets(t)
    x = least_squares(residuals, x, args=(tg,), bounds=(x_lo, x_hi), x_scale="jac",
                      max_nfev=2000 if t == 0 else 300).x
    solution.append(x.copy())
    tm = terms(x, tg)
    worst["pos"] = max(worst["pos"], np.linalg.norm(tm["left_pos"]), np.linalg.norm(tm["right_pos"]))
    worst["rot"] = max(worst["rot"], np.linalg.norm(tm["left_rot"]), np.linalg.norm(tm["right_rot"]))
    worst["com"] = max(worst["com"], np.linalg.norm(tm["com"]))
    worst["facing"] = max(worst.get("facing", 0.0), abs(tm["facing"][0]))
    plant_yaw = abs(np.degrees(yaw_between(data.xmat[plant_id].reshape(3, 3)[:, 0], [1.0, 0.0, 0.0])))
    if KEYFRAMES["plant"][0] <= t <= KEYFRAMES["right_back"][0]:
        worst["plant_twist"] = max(worst.get("plant_twist", 0.0), plant_yaw)
    worst["lean"] = max(worst.get("lean", 0.0), np.degrees(np.arcsin(min(1.0, np.linalg.norm(tm["upright"])))))
    feet_gap, ball_gap = clearances(t)
    worst["feet_gap"] = min(worst["feet_gap"], (feet_gap, t))
    worst["ball_gap"] = min(worst["ball_gap"], (ball_gap, t))
    for name, k in KEYFRAMES.items():
        if abs(t - k[0]) < 0.5 / FPS:
            set_pose(x)
            sole = min(data.geom_xpos[g][2] for g in range(model.ngeom)
                       if model.geom_bodyid[g] == kick_id and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
            print(f"[motion] {name:9s} t={t:.2f}s feet {max(np.linalg.norm(tm['left_pos']), np.linalg.norm(tm['right_pos']))*100:4.1f} cm"
                  f" / {np.degrees(max(np.linalg.norm(tm['left_rot']), np.linalg.norm(tm['right_rot']))):4.1f} deg"
                  f" | CoM off support {np.linalg.norm(tm['com'])*100:4.1f} cm | kick sole {sole*100:4.1f} cm", flush=True)
solution = np.stack(solution)
print(f"[motion] worst over clip: feet {worst['pos']*100:.1f} cm / {np.degrees(worst['rot']):.1f} deg, "
      f"CoM {worst['com']*100:.1f} cm, head off plant-foot direction {np.degrees(worst['facing']):.1f} deg, torso lean {worst['lean']:.1f} deg, plant-foot twist while planted {worst['plant_twist']:.1f} deg", flush=True)
print(f"[motion] min clearance to plant foot: kick foot {worst['feet_gap'][0]*100:.1f} cm (t={worst['feet_gap'][1]:.2f}s), "
      f"ball {worst['ball_gap'][0]*100:.1f} cm (t={worst['ball_gap'][1]:.2f}s)", flush=True)


def finite_diff_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a (T, B, 4) wxyz quaternion series, as a backward difference
    (omega_t = rotvec(R_t R_{t-1}^T) / dt, first frame copied), the convention of the walking clip."""
    rot = Rotation.from_quat(np.roll(quat_wxyz, -1, axis=-1).reshape(-1, 4)).as_matrix().reshape(*quat_wxyz.shape[:2], 3, 3)
    rel = np.einsum("tbij,tbkj->tbik", rot[1:], rot[:-1])
    omega = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(len(rel), -1, 3) / dt
    return np.concatenate([omega[:1], omega], axis=0)


walk = np.load(WALK_CLIP)
walk_err = np.abs(finite_diff_ang_vel(walk["body_quat_w"][:200], 1.0 / FPS)[1:] - walk["body_ang_vel_w"][1:200]).max()
print(f"[motion] angular-velocity convention check vs walking clip: max error {walk_err:.2e} rad/s", flush=True)

joint_names = [str(n) for n in walk["joint_names"]]
body_names = [str(n) for n in walk["body_names"]]
joint_cols = [hinge_names.index(n) for n in joint_names]
body_ids = [model.body(n).id for n in body_names]
body_pos, body_quat = [], []
for x in solution:
    set_pose(x)
    body_pos.append(data.xpos[body_ids].copy())
    body_quat.append(data.xquat[body_ids].copy())
body_pos, body_quat = np.stack(body_pos).astype(np.float32), np.stack(body_quat).astype(np.float32)
joint_pos = solution[:, 6:][:, joint_cols].astype(np.float32)
root_rot = np.roll(body_quat[:, 0], -1, axis=-1)
local = np.einsum("tij,tbj->tbi", Rotation.from_quat(root_rot).inv().as_matrix(), body_pos - body_pos[:, :1])

os.makedirs(args.out_dir, exist_ok=True)
tag = f"r{r:g}{args.tag}"
clip = os.path.join(args.out_dir, f"pass_{tag}.npz")
np.savez(
    clip, fps=np.array([float(FPS)]), joint_pos=joint_pos,
    joint_vel=np.gradient(joint_pos, 1.0 / FPS, axis=0).astype(np.float32),
    body_pos_w=body_pos, body_quat_w=body_quat,
    body_lin_vel_w=np.gradient(body_pos, 1.0 / FPS, axis=0).astype(np.float32),
    body_ang_vel_w=finite_diff_ang_vel(body_quat, 1.0 / FPS).astype(np.float32),
    root_pos=body_pos[:, 0], root_rot=root_rot.astype(np.float32), dof_pos=joint_pos,
    local_body_pos=local.astype(np.float32), link_body_list=np.array(body_names), body_names=np.array(body_names),
    joint_names=np.array(joint_names), body_quat_w_format=np.array("wxyz"), clip_lengths=np.array([len(times)]),
    ball_radius=r, contact_time=T_CONTACT,
)
print(f"[motion] wrote {clip} ({len(times)} frames, {len(times) / FPS:.2f} s)", flush=True)

renderer = mujoco.Renderer(model, height=360, width=480)
option = mujoco.MjvOption()
option.geomgroup[:] = [1, 1, 1, 0, 0, 0]


def render(x: np.ndarray, t: float, azimuth: float) -> np.ndarray:
    set_pose(x, t)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (-0.05, 0.05, 0.35)
    camera.distance, camera.elevation, camera.azimuth = 2.0, -12.0, azimuth
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


# GIF: front view (as in the reference photos) beside a side-on view of the pass line, played at half speed.
frames = [np.concatenate([render(x, t, 135.0), render(x, t, 90.0)], axis=1) for x, t in zip(solution, times)]
frames += [frames[-1]] * 15
gif = os.path.join(args.out_dir, f"pass_{tag}.gif")
imageio.mimsave(gif, frames, duration=2000.0 / FPS, loop=0)
tiles = []
for name, k in KEYFRAMES.items():
    i = int(round(k[0] * FPS))
    tile = Image.fromarray(np.concatenate([render(solution[i], times[i], 135.0), render(solution[i], times[i], 90.0)]))
    ImageDraw.Draw(tile).text((8, 8), f"{name} {k[0]:.2f}s", fill=(255, 255, 255))
    tiles.append(np.asarray(tile))
panel = os.path.join(args.out_dir, f"pass_{tag}_keyframes.png")
imageio.imwrite(panel, np.concatenate(tiles, axis=1))
print(f"[motion] wrote {gif} and {panel}", flush=True)
