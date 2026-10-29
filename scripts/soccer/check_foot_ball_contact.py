"""Physics check for inside-foot ball contact, no RL.

The robot is held in place (fixed base) with the left foot turned out 45 degrees and the right leg folded back out of
the way. A ball is rolled into the inside (medial) face of the left foot, and the first foot-ball contact is reported:
contact point in the foot frame, the elevation angle phi of the contact on the ball (0 = ball middle), and the ball's
velocity before and after.

    python scripts/soccer/check_foot_ball_contact.py --robot soccer --ball_radius 0.11
    python scripts/soccer/check_foot_ball_contact.py --robot stock     # expect no contact: the ball passes through
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", choices=("soccer", "stock"), default="soccer")
parser.add_argument("--ball_radius", type=float, default=0.11)
parser.add_argument("--ball_speed", type=float, default=2.0)
parser.add_argument("--render", action="store_true", help="Save a GIF and key-frame PNGs of the contact.")
parser.add_argument("--record", action="store_true", help="Save robot/ball states to an .npz for replay_mujoco.py.")
parser.add_argument("--out_dir", default="outputs/soccer_checks")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = args.render
simulation_app = AppLauncher(args).app

import os  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_DELAYED_CFG  # noqa: E402
from cyclotron.assets.robots.asimov_1_soccer import ASIMOV_1_SOCCER_CFG, BALL_CFG  # noqa: E402

FOOT = "left_ankle_roll_link"
# Arch of the left foot's inside face, in the foot frame (inside of the left foot is -y).
MEDIAL_PATCH = (0.04, -0.038, -0.013)
MEDIAL_NORMAL = (0.0, -1.0, 0.0)

robot_cfg = ASIMOV_1_SOCCER_CFG if args.robot == "soccer" else ASIMOV_1_DELAYED_CFG
robot_cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot", spawn=robot_cfg.spawn.replace(fix_base=True))
# Left foot turned out; right leg folded back so it is not in the ball's path.
TEST_POSE = {"left_hip_yaw_joint": 0.785, "right_hip_pitch_joint": -0.6, "right_knee_joint": -1.4}


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = robot_cfg
    ball = BALL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Ball", spawn=BALL_CFG.spawn.replace(radius=args.ball_radius)
    )
    # Every robot body, so a ball that misses the foot shows what it hit instead (e.g. the shin).
    foot_ball = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
        track_contact_points=True,
    )
    camera = (
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=640,
            height=480,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        )
        if args.render
        else None
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(CheckSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot, ball, sensor = scene["robot"], scene["ball"], scene["foot_ball"]
    dt = sim.get_physics_dt()
    targets = robot.data.default_joint_pos.clone()
    for name, value in TEST_POSE.items():
        targets[:, robot.find_joints(name)[0][0]] = value
    robot.write_joint_state_to_sim(targets, torch.zeros_like(targets))
    device = robot.device

    def step(n):
        for _ in range(n):
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

    # Park the ball far away while the pose settles.
    state = ball.data.default_root_state.clone()
    state[:, :3] = torch.tensor([[5.0, 5.0, args.ball_radius]], device=device) + scene.env_origins
    ball.write_root_state_to_sim(state)
    step(100)

    foot_id = robot.find_bodies(FOOT)[0][0]
    foot_pos, foot_quat = robot.data.body_pos_w[:, foot_id], robot.data.body_quat_w[:, foot_id]
    patch_w = foot_pos + math_utils.quat_apply(foot_quat, torch.tensor([MEDIAL_PATCH], device=device))
    normal_w = math_utils.quat_apply(foot_quat, torch.tensor([MEDIAL_NORMAL], device=device))
    normal_w[:, 2] = 0.0
    normal_w = normal_w / normal_w.norm(dim=-1, keepdim=True)
    print(f"[check] robot={args.robot} ball_radius={args.ball_radius}", flush=True)
    print(f"[check] left foot origin height {foot_pos[0, 2].item():.3f} m, inside-face normal (world) "
          f"{normal_w[0].tolist()}", flush=True)

    frames = []
    if args.render:
        # Side-on view of the ball's path, slightly above and behind the foot.
        up_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        side = torch.cross(normal_w, up_axis, dim=-1)
        target = patch_w + normal_w * 0.25
        target[:, 2] = 0.12
        scene["camera"].set_world_poses_from_view(target + side * 1.3 + up_axis * 0.35, target)

    # Roll the ball at the inside face, starting 0.8 m out, rolling without slipping.
    start = patch_w + normal_w * (0.8 + args.ball_radius)
    start[:, 2] = args.ball_radius
    vel = -normal_w * args.ball_speed
    up = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    state = ball.data.default_root_state.clone()
    state[:, :3] = start
    state[:, 7:10] = vel
    state[:, 10:13] = torch.cross(up, vel, dim=-1) / args.ball_radius
    ball.write_root_state_to_sim(state)

    body_idx = sensor.body_names.index(FOOT)
    first = None
    touched = {}
    record = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "body_pos", "ball_pos", "ball_quat")}
    for i in range(int(1.5 / dt)):
        before = ball.data.root_lin_vel_w[0].clone()
        step(1)
        if args.render and i % 4 == 0:
            frames.append(scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy())
        if args.record and i % 4 == 0:
            origin = scene.env_origins[0]
            record["root_pos"].append((robot.data.root_pos_w[0] - origin).cpu().numpy())
            record["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy())
            record["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
            record["body_pos"].append((robot.data.body_pos_w[0] - origin).cpu().numpy())
            record["ball_pos"].append((ball.data.root_pos_w[0] - origin).cpu().numpy())
            record["ball_quat"].append(ball.data.root_quat_w[0].cpu().numpy())
        forces = sensor.data.force_matrix_w[0, :, 0].norm(dim=-1)
        for b in torch.nonzero(forces > 1e-3).flatten().tolist():
            if sensor.body_names[b] not in touched:
                touched[sensor.body_names[b]] = i * dt
                print(f"[check]   t={i * dt:.3f}s ball touches {sensor.body_names[b]} ({forces[b].item():.1f} N)",
                      flush=True)
        force = sensor.data.force_matrix_w[0, body_idx, 0]
        if first is None and force.norm() > 1e-3:
            point_w = sensor.data.contact_pos_w[0, body_idx, 0]
            point_f = math_utils.quat_apply_inverse(foot_quat[0], point_w - robot.data.body_pos_w[0, foot_id])
            ball_c = ball.data.root_pos_w[0]
            phi = math.degrees(math.asin(max(-1.0, min(1.0, ((point_w[2] - ball_c[2]) / args.ball_radius).item()))))
            first = dict(t=i * dt, force=force.norm().item(), point_f=point_f.tolist(),
                         point_w=(point_w - scene.env_origins[0]).tolist(),
                         point_h=point_w[2].item(), phi=phi, v_in=before.tolist())
    v_out = ball.data.root_lin_vel_w[0]

    if first is None:
        print("[check] RESULT: no foot-ball contact (ball passed through or missed the foot)")
    else:
        print(f"[check] RESULT: contact at t={first['t']:.3f}s force={first['force']:.1f} N")
        print(f"[check]   contact point in foot frame (x fwd, y left, z up) = "
              f"({first['point_f'][0]:.3f}, {first['point_f'][1]:.3f}, {first['point_f'][2]:.3f}) m, "
              f"height above ground {first['point_h']:.3f} m")
        print(f"[check]   phi (contact elevation on ball, 0 = middle) = {first['phi']:.1f} deg")
        print(f"[check]   ball velocity in  {[round(v, 2) for v in first['v_in']]}")
    print(f"[check]   ball velocity out {[round(v, 2) for v in v_out.tolist()]}, "
          f"ball height {ball.data.root_pos_w[0, 2].item():.3f} m", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.join(args.out_dir, f"{args.robot}_r{args.ball_radius:g}")
    if args.record:
        import numpy as np

        np.savez(
            f"{name}.npz",
            **{k: np.stack(v) for k, v in record.items()},
            joint_names=np.array(robot.joint_names),
            body_names=np.array(robot.body_names),
            robot=args.robot,
            ball_radius=args.ball_radius,
            frame_dt=4 * dt,
            contact_t=first["t"] if first else -1.0,
            contact_point=first["point_w"] if first else np.full(3, np.nan),
        )
        print(f"[check] wrote {name}.npz", flush=True)
    if args.render:
        imageio.mimsave(f"{name}.gif", frames, duration=4 * dt * 1000, loop=0)
        # Key frames: 0.1 s before, at, and 0.2 s after first contact (or the middle of the clip without one).
        hit = int(first["t"] / dt) // 4 if first else len(frames) // 2
        for label, k in (("before", hit - 5), ("contact", hit), ("after", hit + 10)):
            imageio.imwrite(f"{name}_{label}.png", frames[max(0, min(k, len(frames) - 1))])
        print(f"[check] wrote {name}.gif and _before/_contact/_after.png", flush=True)


if __name__ == "__main__":
    main()
    # simulation_app.close() hangs on these nodes after a headless run; nothing is left to flush, so exit directly.
    os._exit(0)
