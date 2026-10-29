"""Run an exported pass policy for one episode per environment, score the passes, and record them for MuJoCo replay.

Per environment it reports what first touched the ball (clean inside-foot touch on the ball's middle, another right-foot
touch, or another body, named), the contact elevation phi, the ball's travel along and across the pass direction, its
peak speed toward the target, and whether the robot fell. Writes one recording per environment for replay_mujoco.py.

    python scripts/soccer/record_pass.py --policy logs/rsl_rl/asimov1_pass/<run>/exported/policy.pt
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="TorchScript policy.pt written by --export.")
parser.add_argument("--task", default="Asimov1-Pass-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_record", type=int, default=6, help="Write recordings for this many environments.")
parser.add_argument("--out_dir", default="outputs/pass_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import cyclotron.tasks  # noqa: E402, F401
from cyclotron.tasks.soccer.mdp.pass_terms import pass_state  # noqa: E402

RECORD_EVERY = 2  # 25 fps
OUTCOME = {1.0: "clean", 0.3: "foot", -1.0: "other"}


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot, ball, sensor = uenv.scene["robot"], uenv.scene["ball"], uenv.scene.sensors["ball_contact"]
    ft_cfg = uenv.reward_manager.get_term_cfg("first_touch")
    links, radius = ft_cfg.params["robot_links"], ft_cfg.params["ball_radius"]
    ft_index = uenv.reward_manager.active_terms.index("first_touch")
    policy = torch.jit.load(args.policy, map_location=uenv.device).eval()

    obs, _ = env.reset()
    n, dev, origins = uenv.num_envs, uenv.device, uenv.scene.env_origins
    state = pass_state(uenv)
    target = state["target_dir"].clone()
    ball_start = ball.data.root_pos_w.clone()
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    end_len = torch.zeros(n, device=dev)
    ball_end = ball_start.clone()
    peak_speed = torch.zeros(n, device=dev)
    touch = [None] * n  # (time, outcome, link, phi)
    record = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "body_pos", "ball_pos", "ball_quat")}
    steps = int(uenv.max_episode_length)
    for i in range(steps):
        # Read the state before stepping: a terminated env is reset inside env.step.
        if i % RECORD_EVERY == 0:
            record["root_pos"].append((robot.data.root_pos_w - origins).cpu().numpy())
            record["root_quat"].append(robot.data.root_quat_w.cpu().numpy())
            record["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
            record["body_pos"].append((robot.data.body_pos_w - origins[:, None]).cpu().numpy())
            record["ball_pos"].append((ball.data.root_pos_w - origins).cpu().numpy())
            record["ball_quat"].append(ball.data.root_quat_w.cpu().numpy())
        ball_end = torch.where(done[:, None], ball_end, ball.data.root_pos_w)
        v_along = (ball.data.root_lin_vel_w * target).sum(-1)
        peak_speed = torch.where(done, peak_speed, torch.maximum(peak_speed, v_along))
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(policy(obs["policy"]))
        ft = uenv.reward_manager._step_reward[:, ft_index] / ft_cfg.weight
        # Strongest contact over the physics steps since the last policy step (the sensor keeps each one).
        force = sensor.data.force_matrix_w_history[:, :, 0].norm(dim=-1).amax(dim=1)
        for e in torch.nonzero((ft != 0) & ~done).flatten().tolist():
            if touch[e] is None:
                link = links[int(force[e].argmax())]
                # Contact elevation from the touch point the reward used (the nearest foot-box point to the ball).
                point = sensor.data.contact_pos_w[e, 0, int(force[e].argmax())]
                if not torch.isfinite(point).all():
                    point = ball.data.root_pos_w[e] - radius * torch.nn.functional.normalize(
                        ball.data.root_pos_w[e] - robot.data.body_pos_w[e, robot.body_names.index(link)], dim=0)
                phi = math.degrees(math.asin(max(-1.0, min(1.0, ((point[2] - ball.data.root_pos_w[e, 2]) / radius).item()))))
                outcome = min(OUTCOME, key=lambda q: abs(q - ft[e].item()))
                touch[e] = (i * uenv.step_dt, OUTCOME[outcome], link.removesuffix("_link"), phi)
        newly = (terminated | truncated) & ~done
        fell |= terminated & ~done
        end_len[newly] = (i + 1) * uenv.step_dt
        done |= newly
        if done.all():
            break
    end_len[~done] = steps * uenv.step_dt

    disp = ball_end - ball_start
    disp[:, 2] = 0.0
    along = (disp * target).sum(-1)
    across = (disp - along[:, None] * target).norm(dim=-1)
    print(f"[pass] policy={args.policy}", flush=True)
    print(f"[pass] {'env':>3s} {'touch':>6s} {'t':>5s} {'link':>22s} {'phi':>6s} {'along':>6s} {'across':>6s} "
          f"{'peak v':>6s} {'fell':>5s} {'len':>4s}", flush=True)
    for e in range(n):
        t, outcome, link, phi = touch[e] or (float("nan"), "none", "-", float("nan"))
        print(f"[pass] {e:3d} {outcome:>6s} {t:5.2f} {link:>22s} {phi:6.1f} {along[e].item():6.2f} "
              f"{across[e].item():6.2f} {peak_speed[e].item():6.2f} {'yes' if fell[e] else 'no':>5s} "
              f"{end_len[e].item():4.1f}", flush=True)
    counts = {k: sum(1 for x in touch if (x[1] if x else "none") == k) for k in ("clean", "foot", "other", "none")}
    print(f"[pass] summary over {n} envs: first touch {counts}, falls {int(fell.sum())}, "
          f"mean ball travel along {along.mean().item():.2f} m / across {across.mean().item():.2f} m, "
          f"mean peak speed {peak_speed.mean().item():.2f} m/s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    frames = len(record["root_pos"])
    for e in range(min(args.num_record, n)):
        k = min(frames, int(math.ceil(end_len[e].item() / (RECORD_EVERY * uenv.step_dt))))
        contact_t = touch[e][0] if touch[e] else -1.0
        np.savez(
            os.path.join(args.out_dir, f"pass_env{e}.npz"),
            **{key: np.stack(v)[:k, e] for key, v in record.items()},
            joint_names=np.array(robot.joint_names), body_names=np.array(robot.body_names), robot="soccer",
            ball_radius=radius, frame_dt=RECORD_EVERY * uenv.step_dt, contact_t=contact_t,
            contact_point=np.full(3, np.nan),
        )
    print(f"[pass] wrote {min(args.num_record, n)} recordings to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() hangs after headless runs on this cluster.
    os._exit(0)
