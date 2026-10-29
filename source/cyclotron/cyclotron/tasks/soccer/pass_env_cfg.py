# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 inside-foot pass: right foot passes a ball along a target direction, from scratch.

Built on the AMP walking config so the robot, actions, proprioceptive observations and regularisers stay the same
(and a walking checkpoint could warm-start it later). The velocity command is held at zero. Episodes start from
random pre-contact frames of the inside-foot pass reference clip (reference state initialisation), and the same clip
is the AMP style reference.
"""

from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from cyclotron.assets.robots.asimov_1_soccer import ASIMOV_1_SOCCER_CFG, BALL_CFG

from ..locomotion.amp_env_cfg import Asimov1AmpEnvCfg
from ..locomotion.velocity_env_cfg import Asimov1SceneCfg
from . import mdp

PASS_BALL_RADIUS = 0.05


def pass_clip(ball_radius: float) -> str:
    """Inside-foot pass reference clip for a ball of this radius (written by scripts/soccer/pass_motion.py)."""
    return str(Path(__file__).resolve().parent / "motions" / f"asimov_1_pass_inside_r{ball_radius:g}.npz")


PASS_CLIP = pass_clip(PASS_BALL_RADIUS)
PASS_TARGET_SPEED = 1.5
ROBOT_LINKS = [
    "pelvis_link", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "waist_yaw_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_yaw_link",
]


@configclass
class PassSceneCfg(Asimov1SceneCfg):
    ball = BALL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=BALL_CFG.spawn.replace(radius=PASS_BALL_RADIUS, activate_contact_sensors=True),
    )
    # The sensor sits on the ball, filtered against each robot link separately: PhysX needs every filter pattern to
    # match one prim per sensor body, so a sensor on all robot links filtered against the one ball reports nothing.
    ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{link}" for link in ROBOT_LINKS],
        track_contact_points=True,
        max_contact_data_count_per_prim=8,
    )


@configclass
class Asimov1PassEnvCfg(Asimov1AmpEnvCfg):
    scene: PassSceneCfg = PassSceneCfg(num_envs=4096, env_spacing=3.0)
    ball_radius: float = PASS_BALL_RADIUS

    def __post_init__(self):
        super().__post_init__()
        r = self.ball_radius
        self.scene.ball.spawn.radius = r
        self.scene.ball.init_state.pos = (0.5, 0.0, r)
        self.episode_length_s = 3.0
        self.scene.robot = ASIMOV_1_SOCCER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Zero velocity command: kept so the policy observation layout matches the walker.
        twist = self.commands.twist
        twist.rel_standing_envs = 1.0
        twist.heading_command = False
        twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)

        ball = {"ball_cfg": mdp.SceneEntityCfg("ball")}
        for group in (self.observations.policy, self.observations.critic):
            group.ball_pos = ObsTerm(func=mdp.ball_pos_b, params=ball)
            group.ball_vel = ObsTerm(func=mdp.ball_vel_b, params=ball)
            group.target_dir = ObsTerm(func=mdp.target_dir_b)
            group.ball_touched = ObsTerm(func=mdp.ball_touched)

        # Events: reference-state resets instead of random poses; no pushes while learning the kick.
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.events.reset_pass = EventTerm(
            func=mdp.reset_from_pass_clip, mode="reset",
            params={"clip_path": pass_clip(r), "ball_radius": r},
        )

        # Rewards: drop the gait and velocity-tracking terms (they assume walking, or pin the standing pose).
        for name in ("track_linear_velocity", "track_angular_velocity", "heading_drift", "pose", "air_time",
                     "foot_clearance", "foot_swing_height", "foot_slip", "soft_landing", "joint_deviation_l1"):
            setattr(self.rewards, name, None)
        # A kick can start and end between two policy steps: keep every physics step's ball contacts.
        self.scene.ball_contact.history_length = self.decimation
        self.rewards.ball_to_target = RewTerm(
            func=mdp.ball_launch_speed, weight=4.0, params={"target_speed": PASS_TARGET_SPEED, **ball}
        )
        self.rewards.ball_direction = RewTerm(func=mdp.ball_launch_direction, weight=3.0, params=ball)
        self.rewards.ball_lateral = RewTerm(func=mdp.ball_lateral_velocity, weight=-1.0, params=ball)
        self.rewards.ball_airborne = RewTerm(
            func=mdp.ball_airborne, weight=-20.0, params={"ball_radius": r, **ball}
        )
        self.rewards.first_touch = RewTerm(
            func=mdp.first_touch_quality, weight=50.0,
            params={"sensor_name": "ball_contact", "robot_links": ROBOT_LINKS, "ball_radius": r, **ball},
        )
        self.rewards.kick_foot_approach = RewTerm(
            func=mdp.kick_foot_approach, weight=0.5, params={"ball_radius": r, **ball}
        )
        # Reward weights are multiplied by the 0.02 s step: staying up earns ~6 over an episode, a fall costs 4.
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=2.0)
        self.rewards.fell = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "fell_over"})
        # Recovery after the pass (from 0.3 s after the touch): small next to the kick terms, so standing still
        # doesn't outweigh kicking; together at most ~6 per episode against ~15 for the pass itself.
        for name, func in (("settle_upright", mdp.settle_upright), ("settle_feet", mdp.settle_feet),
                           ("settle_still", mdp.settle_still), ("settle_pose", mdp.settle_pose),
                           ("settle_heading", mdp.settle_heading)):
            setattr(self.rewards, name, RewTerm(func=func, weight=0.5))
        # No leaning after the kick: upright counts double.
        self.rewards.settle_upright.weight = 1.0
        # Upper body, whole episode: arms relaxed near the standing pose, hands below the chest, torso not rocking.
        # The joint/body selections must be passed in params: the reward manager only resolves SceneEntityCfgs there
        # (a function default keeps every joint/body).
        arms = mdp.SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])
        hands = mdp.SceneEntityCfg("robot", body_names=[".*_wrist_yaw_link"])
        self.rewards.arm_pose = RewTerm(func=mdp.arm_pose, weight=0.5, params={"asset_cfg": arms})
        self.rewards.hands_high = RewTerm(func=mdp.hands_above_height, weight=-5.0, params={"asset_cfg": hands})
        self.rewards.body_ang_vel.weight = -0.2
        # Head (torso) faces where the plant foot points, and the plant foot points along the pass.
        self.rewards.head_facing = RewTerm(func=mdp.head_facing_plant_foot, weight=0.5)
        self.rewards.plant_foot_aim = RewTerm(func=mdp.plant_foot_to_target, weight=0.5)


@configclass
class Asimov1PassEnvCfg_PLAY(Asimov1PassEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.pd_gains_rand = None
        self.events.base_com = None
        self.events.foot_friction = None
        self.events.qpos0_rand = None


@configclass
class Asimov1PassR07EnvCfg(Asimov1PassEnvCfg):
    ball_radius: float = 0.07


@configclass
class Asimov1PassR07EnvCfg_PLAY(Asimov1PassEnvCfg_PLAY):
    ball_radius: float = 0.07


@configclass
class Asimov1PassR09EnvCfg(Asimov1PassEnvCfg):
    ball_radius: float = 0.09


@configclass
class Asimov1PassR09EnvCfg_PLAY(Asimov1PassEnvCfg_PLAY):
    ball_radius: float = 0.09


@configclass
class Asimov1PassR11EnvCfg(Asimov1PassEnvCfg):
    ball_radius: float = 0.11


@configclass
class Asimov1PassR11EnvCfg_PLAY(Asimov1PassEnvCfg_PLAY):
    ball_radius: float = 0.11
