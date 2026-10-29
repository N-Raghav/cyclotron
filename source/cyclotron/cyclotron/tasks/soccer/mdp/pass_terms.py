# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Inside-foot pass terms: reference-state resets, ball observations and pass rewards.

Per-episode pass state (target direction, whether and how the ball was first touched) lives on the env as
``env.pass_state`` and is created on first use, since observation terms are evaluated before the first reset.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from cyclotron.assets.robots.asimov_1_soccer import FOOT_BOX_CENTER, FOOT_BOX_SIZE, FOOT_MEDIAL_PATCH

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

KICK_FOOT = "right_ankle_roll_link"
PLANT_FOOT = "left_ankle_roll_link"
# Inside (medial) face of the right foot in its link frame: +y side, along the arch.
MEDIAL_Y_MIN = 0.02
MEDIAL_X_RANGE = (-0.03, 0.13)
# Standing pose: feet apart (y, from the MJCF at the standing joint angles), and foot-origin height with the sole down.
STANCE_WIDTH = 0.215
SOLE_Z = 0.034


def pass_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    if not hasattr(env, "pass_state"):
        n, dev = env.num_envs, env.device
        env.pass_state = {
            "target_dir": torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1),
            "touched": torch.zeros(n, dtype=torch.bool, device=dev),
            "clean_touch": torch.zeros(n, dtype=torch.bool, device=dev),
            "touch_step": torch.zeros(n, dtype=torch.long, device=dev),
            "launch_speed": torch.zeros(n, device=dev),
            "launch_angle": torch.zeros(n, device=dev),
        }
    return env.pass_state


# Events


_CLIP_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_clip(path: str) -> dict[str, np.ndarray]:
    if path not in _CLIP_CACHE:
        d = np.load(path)
        _CLIP_CACHE[path] = {k: d[k] for k in ("root_pos", "root_rot", "joint_pos", "joint_vel", "body_lin_vel_w",
                                               "body_ang_vel_w", "joint_names", "contact_time", "fps")}
    return _CLIP_CACHE[path]


def reset_from_pass_clip(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    clip_path: str,
    ball_radius: float,
    frames_before_contact: int = 5,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    ball_offset_range: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reference state initialisation: start each episode at a random pre-contact frame of the pass clip.

    The clip has the ball at the origin and passes along +x; the whole setup (robot, pass direction) is rotated by a
    random yaw about the ball, which sits at the env origin.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    clip = _load_clip(clip_path)
    dev, n = env.device, len(env_ids)
    last = int(round(float(clip["contact_time"]) * float(clip["fps"][0]))) - frames_before_contact
    frame = torch.randint(0, max(last, 1), (n,), device="cpu").numpy()

    yaw = torch.empty(n, device=dev).uniform_(*yaw_range)
    q_yaw = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    origins = env.scene.env_origins[env_ids]

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=dev)

    root_pos = math_utils.quat_apply(q_yaw, t(clip["root_pos"][frame])) + origins
    root_quat = math_utils.quat_mul(q_yaw, t(np.roll(clip["root_rot"][frame], 1, axis=-1)))  # xyzw -> wxyz
    lin_vel = math_utils.quat_apply(q_yaw, t(clip["body_lin_vel_w"][frame, 0]))
    ang_vel = math_utils.quat_apply(q_yaw, t(clip["body_ang_vel_w"][frame, 0]))
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)

    names = [str(x) for x in clip["joint_names"]]
    cols = [names.index(j) for j in robot.joint_names]
    robot.write_joint_state_to_sim(t(clip["joint_pos"][frame][:, cols]), t(clip["joint_vel"][frame][:, cols]),
                                   env_ids=env_ids)

    ball_pos = origins.clone()
    ball_pos[:, :2] += torch.empty(n, 2, device=dev).uniform_(-ball_offset_range, ball_offset_range)
    ball_pos[:, 2] = ball_radius
    ball_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1)
    ball.write_root_pose_to_sim(torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    state = pass_state(env)
    state["target_dir"][env_ids] = math_utils.quat_apply(q_yaw, torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1))
    state["touched"][env_ids] = False
    state["clean_touch"][env_ids] = False
    state["launch_speed"][env_ids] = 0.0
    state["launch_angle"][env_ids] = 0.0


# Observations


def ball_pos_b(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot"), ball_cfg=SceneEntityCfg("ball")):
    robot, ball = env.scene[asset_cfg.name], env.scene[ball_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, ball.data.root_pos_w - robot.data.root_pos_w)


def ball_vel_b(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot"), ball_cfg=SceneEntityCfg("ball")):
    robot, ball = env.scene[asset_cfg.name], env.scene[ball_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, ball.data.root_lin_vel_w)


def target_dir_b(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, pass_state(env)["target_dir"])[:, :2]


def ball_touched(env: ManagerBasedRLEnv):
    return pass_state(env)["touched"].float().unsqueeze(-1)


# Rewards


def ball_launch_speed(
    env: ManagerBasedRLEnv, target_speed: float, std: float = 0.5, ball_cfg=SceneEntityCfg("ball")
):
    """After the first touch: the ball's peak speed along the pass direction close to the target speed.

    Peak rather than current speed, so the ball slowing as it rolls doesn't push the policy to kick harder; kicking
    it much harder or softer than the target earns little.
    """
    state = _update_launch(env, ball_cfg)
    reward = torch.exp(-(((state["launch_speed"] - target_speed) / std) ** 2))
    return torch.where(state["touched"], reward, torch.zeros_like(reward))


def ball_launch_direction(env: ManagerBasedRLEnv, std: float = 0.15, ball_cfg=SceneEntityCfg("ball")):
    """After the first touch: the ball's direction at its peak speed close to the pass direction (std in rad)."""
    state = _update_launch(env, ball_cfg)
    reward = torch.exp(-((state["launch_angle"] / std) ** 2))
    return torch.where(state["touched"], reward, torch.zeros_like(reward))


def _update_launch(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> dict[str, torch.Tensor]:
    """Track the ball's peak speed along the pass direction since the first touch, and its heading error then."""
    state = pass_state(env)
    vel = env.scene[ball_cfg.name].data.root_lin_vel_w
    d = state["target_dir"]
    along = (vel * d).sum(-1)
    across = d[:, 0] * vel[:, 1] - d[:, 1] * vel[:, 0]
    peak = state["touched"] & (along > state["launch_speed"])
    state["launch_speed"] = torch.where(peak, along, state["launch_speed"])
    state["launch_angle"] = torch.where(peak, torch.atan2(across, along).abs(), state["launch_angle"])
    return state


def ball_lateral_velocity(env: ManagerBasedRLEnv, ball_cfg=SceneEntityCfg("ball")):
    """Ball speed across the pass direction (penalty)."""
    v = env.scene[ball_cfg.name].data.root_lin_vel_w.clone()
    v[:, 2] = 0.0
    d = pass_state(env)["target_dir"]
    return (v - (v * d).sum(-1, keepdim=True) * d).norm(dim=-1)


def ball_airborne(env: ManagerBasedRLEnv, ball_radius: float, ball_cfg=SceneEntityCfg("ball")):
    """Ball height above rolling contact (penalty): an inside-foot pass stays on the ground."""
    return (env.scene[ball_cfg.name].data.root_pos_w[:, 2] - ball_radius).clamp(min=0.0)


def first_touch_quality(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    robot_links: list[str],
    ball_radius: float,
    max_phi_deg: float = 25.0,
    asset_cfg=SceneEntityCfg("robot"),
    ball_cfg=SceneEntityCfg("ball"),
):
    """Sparse reward on the step the robot first touches the ball.

    +1 for the inside of the right foot meeting the middle of the ball (contact elevation |phi| below max_phi_deg),
    +0.3 for any other right-foot touch, -1 when another body (shin, the plant foot, ...) touches it first.
    `sensor_name` is a contact sensor on the ball filtered against `robot_links`, in that order, with a history as
    long as the env's decimation: a kick can start and end between two policy steps, so a touch is any contact in any
    physics step since the last policy step. By then the contact point may be gone, so where the foot met the ball is
    taken from geometry: the point of the right-foot box closest to the ball centre.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    robot, ball = env.scene[asset_cfg.name], env.scene[ball_cfg.name]
    state = pass_state(env)
    force = sensor.data.force_matrix_w_history[:, :, 0].norm(dim=-1).amax(dim=1)  # (N, links)
    in_contact = force > 1.0
    new = in_contact.any(dim=1) & ~state["touched"]

    foot_s = robot_links.index(KICK_FOOT)
    foot_r = robot.body_names.index(KICK_FOOT)
    foot_first = in_contact[:, foot_s]
    foot_pos, foot_quat = robot.data.body_pos_w[:, foot_r], robot.data.body_quat_w[:, foot_r]
    center = torch.tensor(FOOT_BOX_CENTER, device=env.device)
    half = torch.tensor(FOOT_BOX_SIZE, device=env.device) / 2
    ball_local = math_utils.quat_apply_inverse(foot_quat, ball.data.root_pos_w - foot_pos) - center
    local = torch.maximum(torch.minimum(ball_local, half), -half) + center
    point = foot_pos + math_utils.quat_apply(foot_quat, local)
    on_patch = (local[:, 1] > MEDIAL_Y_MIN) & (local[:, 0] > MEDIAL_X_RANGE[0]) & (local[:, 0] < MEDIAL_X_RANGE[1])
    sin_phi = ((point[:, 2] - ball.data.root_pos_w[:, 2]) / ball_radius).clamp(-1.0, 1.0)
    middle = torch.asin(sin_phi).abs() < math.radians(max_phi_deg)
    clean = foot_first & on_patch & middle

    quality = torch.where(clean, 1.0, torch.where(foot_first, 0.3, -1.0))
    state["touched"] |= new
    state["touch_step"] = torch.where(new, env.episode_length_buf, state["touch_step"])
    state["clean_touch"] |= new & clean
    return torch.where(new, quality, torch.zeros_like(quality))


def kick_foot_approach(
    env: ManagerBasedRLEnv, ball_radius: float, std: float = 0.1, asset_cfg=SceneEntityCfg("robot"),
    ball_cfg=SceneEntityCfg("ball"),
):
    """Before the first touch: the right foot's inside patch approaching the ball's back (opposite the pass)."""
    robot, ball = env.scene[asset_cfg.name], env.scene[ball_cfg.name]
    state = pass_state(env)
    foot = robot.body_names.index(KICK_FOOT)
    patch_local = torch.tensor([FOOT_MEDIAL_PATCH], device=env.device).repeat(env.num_envs, 1)
    patch = robot.data.body_pos_w[:, foot] + math_utils.quat_apply(robot.data.body_quat_w[:, foot], patch_local)
    target = ball.data.root_pos_w - ball_radius * state["target_dir"]
    reward = torch.exp(-((patch - target).norm(dim=-1) / std) ** 2)
    return torch.where(state["touched"], torch.zeros_like(reward), reward)


# Recovery after the pass: rewards that switch on `delay` seconds after the first touch, for standing still and
# upright with both feet down side by side. They earn nothing before a touch, so standing instead of kicking pays less.


def _settle_gate(env: ManagerBasedRLEnv, delay: float) -> torch.Tensor:
    state = pass_state(env)
    return state["touched"] & ((env.episode_length_buf - state["touch_step"]) * env.step_dt >= delay)


def settle_upright(env: ManagerBasedRLEnv, delay: float = 0.3, std: float = 0.1, asset_cfg=SceneEntityCfg("robot")):
    """Torso upright: gravity straight down in the base frame."""
    tilt = env.scene[asset_cfg.name].data.projected_gravity_b[:, :2].norm(dim=-1)
    return _settle_gate(env, delay).float() * torch.exp(-((tilt / std) ** 2))


def settle_feet(env: ManagerBasedRLEnv, delay: float = 0.3, std: float = 0.05, asset_cfg=SceneEntityCfg("robot")):
    """Both feet down, level with each other (along the heading) and standing width apart."""
    robot = env.scene[asset_cfg.name]
    right, left = robot.body_names.index(KICK_FOOT), robot.body_names.index(PLANT_FOOT)
    d = math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w),
                                      robot.data.body_pos_w[:, left] - robot.data.body_pos_w[:, right])
    err = (d[:, 0] / std) ** 2 + ((d[:, 1] - STANCE_WIDTH) / std) ** 2
    down = (robot.data.body_pos_w[:, [left, right], 2] < SOLE_Z + 0.02).all(dim=1)
    return (_settle_gate(env, delay) & down).float() * torch.exp(-err)


def settle_still(env: ManagerBasedRLEnv, delay: float = 0.3, std: float = 0.2, asset_cfg=SceneEntityCfg("robot")):
    """Base not moving: no stepping, drifting or turning after the pass."""
    robot = env.scene[asset_cfg.name]
    speed2 = robot.data.root_lin_vel_b[:, :2].square().sum(-1) + 0.2 * robot.data.root_ang_vel_b[:, 2].square()
    return _settle_gate(env, delay).float() * torch.exp(-speed2 / std**2)


def settle_pose(env: ManagerBasedRLEnv, delay: float = 0.3, std: float = 0.3, asset_cfg=SceneEntityCfg("robot")):
    """Joints back near the standing pose: RMS joint error, so each joint brought back earns something."""
    robot = env.scene[asset_cfg.name]
    err = (robot.data.joint_pos - robot.data.default_joint_pos).square().mean(-1)
    return _settle_gate(env, delay).float() * torch.exp(-err / std**2)


def settle_heading(env: ManagerBasedRLEnv, delay: float = 0.3, std: float = 0.3, asset_cfg=SceneEntityCfg("robot")):
    """Torso facing the pass direction again, as in the reference stance (std in rad)."""
    robot = env.scene[asset_cfg.name]
    fwd = math_utils.quat_apply(robot.data.root_quat_w, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(env.num_envs, 3))
    d = pass_state(env)["target_dir"]
    yaw_err = torch.atan2(d[:, 0] * fwd[:, 1] - d[:, 1] * fwd[:, 0], (d[:, :2] * fwd[:, :2]).sum(-1))
    return _settle_gate(env, delay).float() * torch.exp(-((yaw_err / std) ** 2))


# Upper body, whole episode.


def arm_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.3):
    """Arms near the relaxed standing pose (RMS joint error, std in rad): arms balance the kick, not fly up."""
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    err = (robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]).square().mean(-1)
    return torch.exp(-err / std**2)


def hands_above_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_height: float = 0.15):
    """Hands higher than `max_height` above the pelvis (penalty, m). Standing, they hang at about pelvis height;
    the shoulders are 0.34 m above it."""
    robot = env.scene[asset_cfg.name]
    height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - robot.data.root_pos_w[:, 2:3]
    return (height - max_height).clamp(min=0.0).sum(-1)


# Facing: the head (fixed to the torso, waist_yaw_link) looks where the plant foot points, and the plant foot points
# along the pass. That keeps the body square to the pass and the ball going where intended.


def _heading_error(quat: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) between a body's forward axis and a horizontal direction."""
    fwd = math_utils.quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=quat.device).expand(quat.shape[0], 3))
    return torch.atan2(direction[:, 0] * fwd[:, 1] - direction[:, 1] * fwd[:, 0], (direction[:, :2] * fwd[:, :2]).sum(-1))


def _forward(quat: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=quat.device).expand(quat.shape[0], 3))


def head_facing_plant_foot(env: ManagerBasedRLEnv, std: float = 0.3, asset_cfg=SceneEntityCfg("robot")):
    """Torso/head heading close to the plant (left) foot's heading."""
    robot = env.scene[asset_cfg.name]
    torso, plant = robot.body_names.index("waist_yaw_link"), robot.body_names.index(PLANT_FOOT)
    err = _heading_error(robot.data.body_quat_w[:, torso], _forward(robot.data.body_quat_w[:, plant]))
    return torch.exp(-((err / std) ** 2))


def plant_foot_to_target(env: ManagerBasedRLEnv, std: float = 0.3, asset_cfg=SceneEntityCfg("robot")):
    """Plant (left) foot pointing along the pass direction."""
    robot = env.scene[asset_cfg.name]
    plant = robot.body_names.index(PLANT_FOOT)
    err = _heading_error(robot.data.body_quat_w[:, plant], pass_state(env)["target_dir"])
    return torch.exp(-((err / std) ** 2))
