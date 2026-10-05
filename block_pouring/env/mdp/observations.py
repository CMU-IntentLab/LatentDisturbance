# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


def cube_positions_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The position of the cubes in the world frame."""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]

	return torch.cat((cube_1.data.root_link_pos_w, cube_2.data.root_link_pos_w, cube_3.data.root_link_pos_w), dim=1)

def instance_randomize_cube_positions_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The position of the cubes in the world frame."""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

	cube_1_pos_w = []
	cube_2_pos_w = []
	cube_3_pos_w = []
	for env_id in range(env.num_envs):
		cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
		cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
		cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
	cube_1_pos_w = torch.stack(cube_1_pos_w)
	cube_2_pos_w = torch.stack(cube_2_pos_w)
	cube_3_pos_w = torch.stack(cube_3_pos_w)

	return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
	"""The orientation of the cubes in the world frame."""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]

	return torch.cat((cube_1.data.root_link_quat_w, cube_2.data.root_link_quat_w, cube_3.data.root_link_quat_w), dim=1)


def instance_randomize_cube_orientations_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The orientation of the cubes in the world frame."""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

	cube_1_quat_w = []
	cube_2_quat_w = []
	cube_3_quat_w = []
	for env_id in range(env.num_envs):
		cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
		cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
		cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
	cube_1_quat_w = torch.stack(cube_1_quat_w)
	cube_2_quat_w = torch.stack(cube_2_quat_w)
	cube_3_quat_w = torch.stack(cube_3_quat_w)

	return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
	"""Object observations (in world frame):
	cube_1 pos,
	cube_1 quat,
	cube_2 pos,
	cube_2 quat,
	cube_3 pos,
	cube_3 quat,
	gripper to cube_1,
	gripper to cube_2,
	gripper to cube_3,
	cube_1 to cube_2,
	cube_2 to cube_3,
	cube_1 to cube_3,
	"""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

	cube_1_pos_w = cube_1.data.root_link_pos_w
	cube_1_quat_w = cube_1.data.root_link_quat_w

	cube_2_pos_w = cube_2.data.root_link_pos_w
	cube_2_quat_w = cube_2.data.root_link_quat_w

	cube_3_pos_w = cube_3.data.root_link_pos_w
	cube_3_quat_w = cube_3.data.root_link_quat_w

	ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
	gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
	gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
	gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

	cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
	cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
	cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

	return torch.cat(
		(
			cube_1_pos_w,
			cube_1_quat_w,
			cube_2_pos_w,
			cube_2_quat_w,
			cube_3_pos_w,
			cube_3_quat_w,
			gripper_to_cube_1,
			gripper_to_cube_2,
			gripper_to_cube_3,
			cube_1_to_2,
			cube_2_to_3,
			cube_1_to_3,
		),
		dim=1,
	)


def instance_randomize_object_obs(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
	"""Object observations (in world frame):
	cube_1 pos,
	cube_1 quat,
	cube_2 pos,
	cube_2 quat,
	cube_3 pos,
	cube_3 quat,
	gripper to cube_1,
	gripper to cube_2,
	gripper to cube_3,
	cube_1 to cube_2,
	cube_2 to cube_3,
	cube_1 to cube_3,
	"""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

	cube_1_pos_w = []
	cube_2_pos_w = []
	cube_3_pos_w = []
	cube_1_quat_w = []
	cube_2_quat_w = []
	cube_3_quat_w = []
	for env_id in range(env.num_envs):
		cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
		cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
		cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
		cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
		cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
		cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
	cube_1_pos_w = torch.stack(cube_1_pos_w)
	cube_2_pos_w = torch.stack(cube_2_pos_w)
	cube_3_pos_w = torch.stack(cube_3_pos_w)
	cube_1_quat_w = torch.stack(cube_1_quat_w)
	cube_2_quat_w = torch.stack(cube_2_quat_w)
	cube_3_quat_w = torch.stack(cube_3_quat_w)

	ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
	gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
	gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
	gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

	cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
	cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
	cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

	return torch.cat(
		(
			cube_1_pos_w,
			cube_1_quat_w,
			cube_2_pos_w,
			cube_2_quat_w,
			cube_3_pos_w,
			cube_3_quat_w,
			gripper_to_cube_1,
			gripper_to_cube_2,
			gripper_to_cube_3,
			cube_1_to_2,
			cube_2_to_3,
			cube_1_to_3,
		),
		dim=1,
	)


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]

	return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_quat = ee_frame.data.target_quat_source[:, 0, :]

	return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	robot: Articulation = env.scene[robot_cfg.name]
	finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
	finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

	return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def gripper_contact(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	contact_left = env.scene.sensors["contact_forces_left"].data.force_matrix_w
	contact_right = env.scene.sensors["contact_forces_right"].data.force_matrix_w

	has_left_contact  = torch.norm(contact_left,  dim=-1) > 0.0
	has_right_contact = torch.norm(contact_right, dim=-1) > 0.0

	return (has_left_contact | has_right_contact).reshape(env.num_envs)

def gripper_contact_bottom(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	contact_left = env.scene.sensors["contact_bottom_left"].data.force_matrix_w
	contact_right = env.scene.sensors["contact_bottom_right"].data.force_matrix_w

	has_left_contact  = torch.norm(contact_left,  dim=-1) > 0.0
	has_right_contact = torch.norm(contact_right, dim=-1) > 0.0

	return (has_left_contact & has_right_contact).reshape(env.num_envs)

def is_poured(env: ManagerBasedRLEnv) -> torch.Tensor:
	cotact =  env.scene.sensors["contact_target_bottom"].data.force_matrix_w
	not_poured  = torch.norm(cotact,  dim=-1) > 0.0

	return (~not_poured).reshape(env.num_envs)

def object_physics(env: ManagerBasedRLEnv) -> torch.Tensor:
	asset = env.scene['pick_cube_1']
	masses = asset.root_physx_view.get_masses()
	frictions = asset.root_physx_view.get_material_properties()[:, 0, :]

	return torch.cat([masses, frictions], dim=1)

def object_grasped(
	env: ManagerBasedRLEnv,
	robot_cfg: SceneEntityCfg,
	ee_frame_cfg: SceneEntityCfg,
	object_cfg: SceneEntityCfg,
	diff_threshold: float = 0.06,
	gripper_open_val: torch.tensor = torch.tensor([0.04]),
	gripper_threshold: float = 0.005,
) -> torch.Tensor:
	"""Check if an object is grasped by the specified robot."""

	robot: Articulation = env.scene[robot_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	object: RigidObject = env.scene[object_cfg.name]

	object_pos = object.data.root_link_pos_w
	end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
	pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

	grasped = torch.logical_and(
		pose_diff < diff_threshold,
		torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
	)
	grasped = torch.logical_and(
		grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
	)

	return grasped


def object_stacked(
	env: ManagerBasedRLEnv,
	robot_cfg: SceneEntityCfg,
	upper_object_cfg: SceneEntityCfg,
	lower_object_cfg: SceneEntityCfg,
	xy_threshold: float = 0.05,
	height_threshold: float = 0.005,
	height_diff: float = 0.0468,
	gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
	"""Check if an object is stacked by the specified robot."""

	robot: Articulation = env.scene[robot_cfg.name]
	upper_object: RigidObject = env.scene[upper_object_cfg.name]
	lower_object: RigidObject = env.scene[lower_object_cfg.name]

	pos_diff = upper_object.data.root_link_pos_w - lower_object.data.root_link_pos_w
	height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

	stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

	stacked = torch.logical_and(
		torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
	)
	stacked = torch.logical_and(
		torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
	)

	return stacked


def ee_frame_out_of_boundary(
	env: ManagerBasedRLEnv,
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
	x_min: float = 0.3, x_max = 0.7, y_min = -0.3, y_max = 0.3
) -> torch.Tensor:

	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]
	
	# Extract x, y positions
	x_pos = ee_frame_pos[:, 0]
	y_pos = ee_frame_pos[:, 1]
	
	# Check if x and y are within the boundary
	within_x = (x_pos >= x_min) & (x_pos <= x_max)
	within_y = (y_pos >= y_min) & (y_pos <= y_max)
	within_boundary = within_x & within_y
		
	return ~within_boundary


def check_stacked(
	env,
	upper_obj_name: str,
	lower_obj_name: str,
	xy_threshold: float,
	height_diff: float,
	height_threshold: float,
	min_height_diff:float
) -> torch.Tensor:
	"""
	Returns a tensor of booleans indicating whether `upper_obj_name` is stacked on `lower_obj_name`
	for each environment instance in the batch.
	"""
	upper_obj = env.scene[upper_obj_name]
	lower_obj = env.scene[lower_obj_name]

	pos_upper = upper_obj.data.root_link_pos_w
	pos_lower = lower_obj.data.root_link_pos_w
	# relative XY offset
	delta = pos_upper[:, :2] - pos_lower[:, :2]

	half_side = xy_threshold / 2.0
	tol       = 1e-3

	# boolean masks per env
	within_x = delta[:, 0].abs() <= (half_side + tol)
	within_y = delta[:, 1].abs() <= (half_side + tol)

	# final “lies on top” test in XY plane
	is_on_top_xy = within_x & within_y

	# Z distance
	pos_diff = upper_obj.data.root_link_pos_w - lower_obj.data.root_link_pos_w
	height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)

	# Condition 1: XY must be close
	cond_xy = is_on_top_xy

	# Condition 2: Z must be near the known stack height
	cond_z = torch.abs(height_dist - height_diff) < height_threshold

	# upper one's height should be bigger than min_height diff -> success
	over_z = (height_dist > min_height_diff)

	stacked = cond_xy & cond_z & over_z
	return stacked

def is_failure(
	env,
) -> torch.Tensor:
	"""Returns a boolean tensor indicating failure in each environment instance."""

	base_height = 0.1
	c3_drop = (env.scene['pick_cube_1'].data.root_link_pos_w[:, 2] < (base_height + 0.01))

	return c3_drop

def check_stacked_3on1(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	stacked_3on1 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return stacked_3on1

def check_not_stacked_2on1(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	
	stacked_2on1 = check_stacked(
		env=env,
		upper_obj_name=cube_2_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return ~stacked_2on1

def check_not_stacked_3on2(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	
	stacked_3on2 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_2_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return ~stacked_3on2

def is_success(
		env
) -> torch.Tensor:

	stacked_block_on_target = check_stacked(
		env=env,
		upper_obj_name="pick_cube_1",
		lower_obj_name="cube_1",
		xy_threshold=0.05,
		height_diff=0.02,
		height_threshold = 0.005,
		min_height_diff = 0.015
	)

	static = torch.norm(env.scene['pick_cube_1'].data.root_link_lin_vel_w, dim=-1) < 0.025

	
	return stacked_block_on_target & static

def rel_pos_12(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
) -> torch.Tensor:
	'''
	xy only
	'''

	pos_diff = env.scene[cube_2_name].data.root_link_pos_w - env.scene[cube_1_name].data.root_link_pos_w
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1) * 100
	reward = 1  / (1 + xy_dist)

	return reward

def rel_pos_xyz_l2(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
) -> torch.Tensor:
	'''
	xyz
	'''

	pos_diff = env.scene[cube_2_name].data.root_link_pos_w - env.scene[cube_1_name].data.root_link_pos_w

	xy_diff = pos_diff[:, :2]
	xy_diff = torch.linalg.vector_norm(xy_diff, dim=1)
	z_diff = torch.abs(pos_diff[:, 2] + 0.02)
	
	reward = -20 * xy_diff - 5 * z_diff + 1
	
	return reward


def abs_z_12(
	env,
	object_name: str = "pick_cube_bottom",
) -> torch.Tensor:

	# cube 2 distance from cube 3
	pos_z = env.scene[object_name].data.root_link_pos_w[:, 2]

	return pos_z