# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""PPO + AMP configuration for the inside-foot pass."""

from isaaclab.utils import configclass

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_JOINT_NAMES

from ...locomotion.agents.rsl_rl_ppo_cfg import Asimov1AMPRunnerCfg
from ...locomotion.amp_env_cfg import ASIMOV_1_AMP_OBS_TERMS, ASIMOV_1_ANCHOR_NAME, ASIMOV_1_KEY_BODY_NAMES
from ...locomotion.motion_dataset import MotionDatasetCfg
from ..pass_env_cfg import PASS_CLIP, pass_clip


@configclass
class Asimov1PassRunnerCfg(Asimov1AMPRunnerCfg):
    experiment_name = "asimov1_pass"
    run_name = "pass"
    max_iterations = 5000
    save_interval = 250
    algorithm = Asimov1AMPRunnerCfg().algorithm.replace(
        amp_data=MotionDatasetCfg(
            motion_files=[PASS_CLIP],
            joint_names=list(ASIMOV_1_JOINT_NAMES),
            body_names=list(ASIMOV_1_KEY_BODY_NAMES),
            amp_obs_terms=list(ASIMOV_1_AMP_OBS_TERMS),
            anchor_name=ASIMOV_1_ANCHOR_NAME,
        ),
        # The command is always zero here, so the walker's command gate would switch the style reward off.
        amp_reward_command_gate=False,
    )


@configclass
class Asimov1PassR07RunnerCfg(Asimov1PassRunnerCfg):
    run_name = "pass_r0.07"
    algorithm = Asimov1PassRunnerCfg().algorithm.replace(
        amp_data=Asimov1PassRunnerCfg().algorithm.amp_data.replace(motion_files=[pass_clip(0.07)])
    )


@configclass
class Asimov1PassR09RunnerCfg(Asimov1PassRunnerCfg):
    run_name = "pass_r0.09"
    algorithm = Asimov1PassRunnerCfg().algorithm.replace(
        amp_data=Asimov1PassRunnerCfg().algorithm.amp_data.replace(motion_files=[pass_clip(0.09)])
    )


@configclass
class Asimov1PassR11RunnerCfg(Asimov1PassRunnerCfg):
    run_name = "pass_r0.11"
    algorithm = Asimov1PassRunnerCfg().algorithm.replace(
        amp_data=Asimov1PassRunnerCfg().algorithm.amp_data.replace(motion_files=[pass_clip(0.11)])
    )
