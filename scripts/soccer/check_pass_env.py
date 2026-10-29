"""Sanity check for Asimov1-Pass: reset geometry and ball-contact detection, no policy.

Resets the pass environment (reference-state init), then rolls the ball into each robot's right foot while holding the
default pose, and reports what the robot-ball contact sensor and the first-touch reward see.

    python scripts/soccer/check_pass_env.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--no_replicate", action="store_true", help="Build each env's physics separately.")
parser.add_argument("--out", default="outputs/soccer_checks/pass_env_check.npz", help="Recording of env 0 for replay.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cyclotron.tasks  # noqa: E402, F401
from cyclotron.tasks.soccer.mdp.pass_terms import KICK_FOOT, pass_state  # noqa: E402

TASK = "Asimov1-Pass-Play-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    if args.no_replicate:
        env_cfg.scene.replicate_physics = False
    env_cfg.scene.ball_any = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Ball")  # everything the ball touches
    env = gym.make(TASK, cfg=env_cfg)
    uenv = env.unwrapped
    robot, ball, sensor = uenv.scene["robot"], uenv.scene["ball"], uenv.scene.sensors["ball_contact"]
    env.reset()

    foot = robot.body_names.index(KICK_FOOT)
    plant = robot.body_names.index("left_ankle_roll_link")
    d_kick = (robot.data.body_pos_w[:, foot] - ball.data.root_pos_w).norm(dim=-1)
    d_plant = (robot.data.body_pos_w[:, plant] - ball.data.root_pos_w).norm(dim=-1)
    print(f"[check] sensor bodies ({len(sensor.body_names)}): {sensor.body_names}", flush=True)
    print(f"[check] force_matrix_w shape {tuple(sensor.data.force_matrix_w.shape)}, contact_pos_w "
          f"{None if sensor.data.contact_pos_w is None else tuple(sensor.data.contact_pos_w.shape)}", flush=True)
    print(f"[check] after reset: ball-to-right-foot {d_kick.cpu().numpy().round(3)} m, "
          f"ball-to-left-foot {d_plant.cpu().numpy().round(3)} m, base height "
          f"{robot.data.root_pos_w[:, 2].cpu().numpy().round(3)} m", flush=True)

    # Roll the ball at the right foot from 0.4 m away.
    target = robot.data.body_pos_w[:, foot].clone()
    target[:, 2] = ball.data.root_pos_w[:, 2]
    away = ball.data.root_pos_w - target
    away[:, 2] = 0.0
    away = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    pose = ball.data.root_pose_w.clone()
    pose[:, :3] = target + 0.4 * away
    vel = torch.zeros(uenv.num_envs, 6, device=uenv.device)
    vel[:, :3] = -2.0 * away
    ball.write_root_pose_to_sim(pose)
    ball.write_root_velocity_to_sim(vel)

    actions = torch.zeros(uenv.num_envs, uenv.action_manager.total_action_dim, device=uenv.device)
    max_filtered = torch.zeros(uenv.num_envs, device=uenv.device)
    min_dist = torch.full((uenv.num_envs,), 9.0, device=uenv.device)
    ball_any = uenv.scene.sensors["ball_any"]
    # Horizontal contact force on the ball (its weight on the ground is vertical).
    max_ball_side = torch.zeros(uenv.num_envs, device=uenv.device)
    resets = torch.zeros(uenv.num_envs, dtype=torch.long, device=uenv.device)
    record = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "body_pos", "ball_pos", "ball_quat")}
    origin = uenv.scene.env_origins[0]
    first_touch_sum = torch.zeros(uenv.num_envs, device=uenv.device)
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(actions)
        resets += (terminated | truncated).long()
        max_ball_side = torch.maximum(max_ball_side, ball_any.data.net_forces_w[:, 0, :2].norm(dim=-1))
        record["root_pos"].append((robot.data.root_pos_w[0] - origin).cpu().numpy())
        record["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy())
        record["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
        record["body_pos"].append((robot.data.body_pos_w[0] - origin).cpu().numpy())
        record["ball_pos"].append((ball.data.root_pos_w[0] - origin).cpu().numpy())
        record["ball_quat"].append(ball.data.root_quat_w[0].cpu().numpy())
        max_filtered = torch.maximum(max_filtered, sensor.data.force_matrix_w.norm(dim=-1).amax(dim=(1, 2)))
        min_dist = torch.minimum(min_dist, (robot.data.body_pos_w[:, foot] - ball.data.root_pos_w).norm(dim=-1))
        first_touch_sum += uenv.reward_manager._step_reward[:, uenv.reward_manager.active_terms.index("first_touch")]
    print(f"[check] max ball-robot force (ball sensor filtered per link) per env: {max_filtered.cpu().numpy().round(1)}", flush=True)
    print(f"[check] closest ball centre to right-foot origin: {min_dist.cpu().numpy().round(3)} m", flush=True)
    print(f"[check] episode resets during the roll: {resets.cpu().numpy()}", flush=True)
    print(f"[check] max horizontal force on the ball (ball sensor, unfiltered): {max_ball_side.cpu().numpy().round(1)}",
          flush=True)
    print(f"[check] touched flags: {pass_state(uenv)['touched'].cpu().numpy()}", flush=True)
    print(f"[check] first_touch reward summed per env: {first_touch_sum.cpu().numpy().round(3)}", flush=True)
    print(f"[check] ball speed now: {ball.data.root_lin_vel_w[:, :2].norm(dim=-1).cpu().numpy().round(2)} m/s",
          flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{k: np.stack(v) for k, v in record.items()}, joint_names=np.array(robot.joint_names),
             body_names=np.array(robot.body_names), robot="soccer", ball_radius=float(ball.cfg.spawn.radius),
             frame_dt=uenv.step_dt, contact_t=-1.0, contact_point=np.full(3, np.nan))
    print(f"[check] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
