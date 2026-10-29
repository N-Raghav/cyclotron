"""Check an exported walking policy on flat ground with fixed velocity commands, and record it for MuJoCo replay.

One environment per command, each held for the whole run. Reports, per command, the achieved velocity (mean over
the last half, in the base frame) and whether the robot fell, and writes one recording per command for
replay_mujoco.py.

    python scripts/soccer/record_walk.py --policy logs/rsl_rl/<exp>/<run>/exported/policy.pt --robot soccer
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="TorchScript policy.pt written by --export.")
parser.add_argument("--robot", choices=("soccer", "stock"), default="soccer")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--out_dir", default="outputs/walk_check")
parser.add_argument("--debug", action="store_true", help="Print the observed commands and action sizes early on.")
parser.add_argument("--no_self_collision", action="store_true", help="Disable the robot's self-collisions.")
parser.add_argument("--box_lift", type=float, default=0.0, help="Raise the soccer foot-box bottom by this much (m).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cyclotron.tasks  # noqa: E402, F401
from cyclotron.assets.robots.asimov_1_soccer import (  # noqa: E402
    ASIMOV_1_SOCCER_CFG,
    FOOT_BOX_CENTER,
    FOOT_BOX_SIZE,
    write_soccer_urdf,
)

TASK = "Asimov1-Velocity-AMP-Play-v0"
# name: (vx, vy, wz) in the base frame, within the trained ranges.
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd_0.4": (0.4, 0.0, 0.0),
    "fwd_0.8": (0.8, 0.0, 0.0),
    "back_0.4": (-0.4, 0.0, 0.0),
    "side_0.4": (0.0, 0.4, 0.0),
    "turn_0.6": (0.0, 0.0, 0.6),
    "arc": (0.4, 0.0, 0.4),
}
RECORD_EVERY = 2  # 25 fps


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=len(COMMANDS))
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if args.robot == "soccer":
        env_cfg.scene.robot = ASIMOV_1_SOCCER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if args.box_lift:
            # Same box top, bottom raised: tests whether the box catches the ground.
            center = (FOOT_BOX_CENTER[0], FOOT_BOX_CENTER[1], FOOT_BOX_CENTER[2] + args.box_lift / 2)
            size = (FOOT_BOX_SIZE[0], FOOT_BOX_SIZE[1], FOOT_BOX_SIZE[2] - args.box_lift)
            urdf = write_soccer_urdf(box_center=center, box_size=size, name=f"asimov_1_soccer_lift{args.box_lift:g}")
            env_cfg.scene.robot = env_cfg.scene.robot.replace(spawn=env_cfg.scene.robot.spawn.replace(asset_path=urdf))
    if args.no_self_collision:
        spawn = env_cfg.scene.robot.spawn
        env_cfg.scene.robot = env_cfg.scene.robot.replace(spawn=spawn.replace(
            articulation_props=spawn.articulation_props.replace(enabled_self_collisions=False)))
    twist = env_cfg.commands.twist
    twist.resampling_time_range = (1e6, 1e6)
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    env = gym.make(TASK, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=uenv.device).eval()

    commands = torch.tensor(list(COMMANDS.values()), device=uenv.device)
    env.reset()
    uenv.command_manager.get_term("twist").vel_command_b[:] = commands
    obs = uenv.observation_manager.compute()

    steps = int(args.seconds / uenv.step_dt)
    fell_at = torch.full((len(COMMANDS),), -1.0, device=uenv.device)
    vel_sum = torch.zeros(len(COMMANDS), 3, device=uenv.device)
    vel_count = 0
    origins = uenv.scene.env_origins
    record = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "body_pos")}
    for i in range(steps):
        with torch.inference_mode():
            obs, _, terminated, _, _ = env.step(policy(obs["policy"]))
        uenv.command_manager.get_term("twist").vel_command_b[:] = commands
        if args.debug and i in (0, 1, 50):
            with torch.inference_mode():
                act = policy(obs["policy"])
            print(f"[debug] step {i} obs command (6:9):\n{obs['policy'][:, 6:9].cpu().numpy().round(3)}\n"
                  f"vel_command_b:\n{uenv.command_manager.get_term('twist').vel_command_b.cpu().numpy().round(3)}\n"
                  f"|action| per env: {act.norm(dim=-1).cpu().numpy().round(3)}", flush=True)
        newly = terminated & (fell_at < 0)
        fell_at[newly] = i * uenv.step_dt
        if i >= steps // 2:
            vel_sum += torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=-1)
            vel_count += 1
        if i % RECORD_EVERY == 0:
            record["root_pos"].append((robot.data.root_pos_w - origins).cpu().numpy())
            record["root_quat"].append(robot.data.root_quat_w.cpu().numpy())
            record["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
            record["body_pos"].append((robot.data.body_pos_w - origins[:, None]).cpu().numpy())

    achieved = (vel_sum / max(vel_count, 1)).cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[walk] robot={args.robot} policy={args.policy}", flush=True)
    print(f"[walk] {'command':9s} {'target vx vy wz':>20s} {'achieved vx vy wz':>22s}  fell", flush=True)
    for k, (name, cmd) in enumerate(COMMANDS.items()):
        t = fell_at[k].item()
        print(f"[walk] {name:9s} {cmd[0]:6.2f} {cmd[1]:6.2f} {cmd[2]:6.2f}   {achieved[k, 0]:6.2f} {achieved[k, 1]:6.2f} "
              f"{achieved[k, 2]:6.2f}   {'no' if t < 0 else f'yes at {t:.1f}s'}", flush=True)
        np.savez(
            os.path.join(args.out_dir, f"{args.robot}_{name}.npz"),
            **{key: np.stack(v)[:, k] for key, v in record.items()},
            joint_names=np.array(robot.joint_names), body_names=np.array(robot.body_names), robot=args.robot,
            frame_dt=RECORD_EVERY * uenv.step_dt,
        )
    print(f"[walk] wrote recordings to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() hangs after headless runs on this cluster.
    os._exit(0)
