# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg

@configclass
class FrankaRobustEnvCfg(joint_pos_env_cfg.FrankaRobustEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # stiffer PD gains for IK tracking; finger contact sensors detect the grasp
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.spawn = robot.spawn.replace(activate_contact_sensors=True)
        self.scene.robot = robot

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.scene.contact_forces_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_Pick_1"],
        )

        self.scene.contact_forces_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_Pick_1"],
        )

        self.scene.contact_bottom_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_Pick_Bottom"],
        )

        self.scene.contact_bottom_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_Pick_Bottom"],
        )

        self.scene.contact_target_bottom = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_Pick_1",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_Pick_Bottom"],
        )
