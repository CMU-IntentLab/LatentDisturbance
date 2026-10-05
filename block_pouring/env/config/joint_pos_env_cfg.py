# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from .. import mdp
from ..env_cfg import RobustEnvCfg


@configclass
class EventCfg:
	"""Configuration for events."""

	init_franka_arm_pose = EventTerm(
		func=franka_stack_events.set_default_joint_pose,
		mode="startup",
		params={
			"default_pose": [-0.0340,  0.1919,  0.2527, -2.3772, -0.0903,  2.5468,  2.6410,  0.0400, 0.0400],
		},
	)

	reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

	randomize_franka_joint_state = EventTerm(
		func=franka_stack_events.randomize_joint_by_gaussian_offset,
		mode="reset",
		params={
			"mean": 0.0,
			"std": 0.02,
			"asset_cfg": SceneEntityCfg("robot"),
		},
	)

	reset_object_position = EventTerm(
		func=mdp.reset_root_states_uniform,
		mode="reset",
		params={
			"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.0, 0.0), "yaw": (-0.2, 0.2)},
			"velocity_range": {},
			"asset_cfgs": [SceneEntityCfg("cube_1")],
		},
	)

	reset_partial_and_randomize_physics = EventTerm(
		func=mdp.reset_partial_and_randomize_physics,
		mode="reset",
		params={
			"cube1_xy_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
			"pick_bottom_pose": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.0, 0.0), "yaw": (-0.1, 0.1)},
			"pick_top_pose": {"x": (-0.005, 0.005), "y": (0, 0.02), "z": (-0.0, 0.0), "yaw": (-0.3, 0.3)},
			"randomize_physics": True
		}
	)


@configclass
class FrankaRobustEnvCfg(RobustEnvCfg):
	def __post_init__(self):
		# post init of parent
		super().__post_init__()

		# Set events
		self.events = EventCfg()

		# Set Franka as robot
		self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

		# Set actions for the specific robot type (franka)
		self.actions.arm_action = mdp.JointPositionActionCfg(
			asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
		)
		self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
			asset_name="robot",
			joint_names=["panda_finger.*"],
			open_command_expr={"panda_finger_.*": 0.04},
			close_command_expr={"panda_finger_.*": 0.0},
		)

		# Listens to the required transforms
		marker_cfg = FRAME_MARKER_CFG.copy()
		marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
		marker_cfg.prim_path = "/Visuals/FrameTransformer"
		self.scene.ee_frame = FrameTransformerCfg(
			prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
			debug_vis=False,
			visualizer_cfg=marker_cfg,
			target_frames=[
				FrameTransformerCfg.FrameCfg(
					prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
					name="end_effector",
					offset=OffsetCfg(
						pos=[0.0, 0.0, 0.1034],
					),
				),
			],
		)

		
		self.scene.front_cam = TiledCameraCfg(
			prim_path="{ENV_REGEX_NS}/front_cam",
			update_period=0.01,
			height=128,
			width=128,
			data_types=["rgb"],
			spawn=PinholeCameraCfg(
				focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
			),
			offset=TiledCameraCfg.OffsetCfg(pos=(0.67, 0.032, 0.3), rot=(0.65328,0.2706,0.2706,0.65328), convention="opengl"),
		)

		self.scene.wrist_cam = TiledCameraCfg(
			prim_path="{ENV_REGEX_NS}/wrist_cam",
			update_period=0.01,
			height=128,
			width=128,
			data_types=["rgb"],
			spawn=PinholeCameraCfg(
				focal_length=32.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
			),
			offset=TiledCameraCfg.OffsetCfg(pos=(0.5, -0.25, 0.3), rot=(0.8870108, 0.4617486, 0, 0), convention="opengl"),
		)

		

