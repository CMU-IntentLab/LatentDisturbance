# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
	"""Configuration for the lift scene with a robot and a object.
	This is the abstract base implementation, the exact scene is defined in the derived classes
	which need to set the target object, robot and end-effector frames
	"""

	# robots: will be populated by agent env cfg
	robot: ArticulationCfg = MISSING
	# end-effector sensor: will be populated by agent env cfg
	ee_frame: FrameTransformerCfg = MISSING

	# Cameras
	wrist_cam: CameraCfg | TiledCameraCfg = MISSING
	front_cam: CameraCfg | TiledCameraCfg = MISSING

	# Table
	table = AssetBaseCfg(
		prim_path="{ENV_REGEX_NS}/Table",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
		spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
	)

	# plane
	plane = AssetBaseCfg(
		prim_path="/World/GroundPlane",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
		spawn=GroundPlaneCfg(),
	)

	# lights
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
	)

	base =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Base",
		spawn=sim_utils.CuboidCfg(
			size=[0.7, 0.7, 0.1],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.00, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_1 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_1",
		spawn=sim_utils.CuboidCfg(
			size=[0.1, 0.1, 0.03],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=2),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.530, -0.00, 0.114), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_2 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_2",
		spawn=sim_utils.CuboidCfg(
			size=[0.11, 0.025, 0.02],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 1.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.01, 0.141), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_3 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_3",
		spawn=sim_utils.CuboidCfg(
			size=[0.04, 0.06, 0.02],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53851, 0.01, 0.16), rot=(0.970, 0.0, 0.0, -0.24)),
	)

base_x = 0.5
base_height = 0.1
cube_1_thickness = 0.02
cube_1_z = base_height + cube_1_thickness / 2.0
cube_1_height = cube_1_thickness + base_height
sphere_radius = 0.03
sphere_z = cube_1_height + sphere_radius
sphere_x_delta = 0.035
sphere_y_delta = 0.035

@configclass
class FourSpherePickBlockSceneCfg(InteractiveSceneCfg):
	"""Scene with four spheres on the table surface and a block with another cube on top for picking."""

	# robot & sensor frames (to be filled by agent env cfg)
	robot: ArticulationCfg = MISSING
	ee_frame: FrameTransformerCfg = MISSING

	wrist_cam: CameraCfg | TiledCameraCfg = MISSING
	front_cam: CameraCfg | TiledCameraCfg = MISSING

	table: AssetBaseCfg = AssetBaseCfg(
		prim_path="{ENV_REGEX_NS}/Table",
		init_state=AssetBaseCfg.InitialStateCfg(
			pos=[0.5, 0.0, 0.0],
			rot=[0.707, 0.0, 0.0, 0.707]
		),
		spawn=UsdFileCfg(
			usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
		),
	)

	# ground plane
	plane: AssetBaseCfg = AssetBaseCfg(
		prim_path="/World/GroundPlane",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
		spawn=GroundPlaneCfg(),
	)

	# dome lighting
	light: AssetBaseCfg = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
	)

	# table base (optional extra support)
	base: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Base",
		spawn=sim_utils.CuboidCfg(
			size=[0.5, 0.5, base_height],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(base_x, 0.0, 0.05),
			rot=(1.0, 0.0, 0.0, 0.0)
		),
	)

	cube_1 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_1",
		spawn=sim_utils.CuboidCfg(
			size=[0.2, 0.2, cube_1_thickness],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=True,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x, -0.00, cube_1_z), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	# four spheres arranged in a square on top of thcub_1_heighte base
	sphere_1: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Sphere_1",
		spawn=sim_utils.SphereCfg(
			radius=sphere_radius,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x + sphere_x_delta,  sphere_y_delta, sphere_z)),
	)

	sphere_2: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Sphere_2",
		spawn=sim_utils.SphereCfg(
			radius=sphere_radius,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(**{ }),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x - sphere_x_delta,  sphere_y_delta, sphere_z)),
	)

	sphere_3: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Sphere_3",
		spawn=sim_utils.SphereCfg(
			radius=sphere_radius,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(**{ }),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x + sphere_x_delta, -sphere_y_delta, sphere_z)),
	)

	sphere_4: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Sphere_4",
		spawn=sim_utils.SphereCfg(
			radius=sphere_radius,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(**{ }),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x - sphere_x_delta, -sphere_y_delta, sphere_z)),
	)


cube_1_y = -0.015
cube_pick_thickness = 0.02
cube_pick_bottom_z = base_height + cube_pick_thickness / 2.0

cube_2_thickness = 0.015
cube_2_z = base_height + cube_pick_thickness + cube_2_thickness / 2.0

class RobustBlockScene(InteractiveSceneCfg):
	"""Scene with four spheres on the table surface and a block with another cube on top for picking."""

	# robot & sensor frames (to be filled by agent env cfg)
	robot: ArticulationCfg = MISSING
	ee_frame: FrameTransformerCfg = MISSING

	wrist_cam: CameraCfg | TiledCameraCfg = MISSING
	front_cam: CameraCfg | TiledCameraCfg = MISSING
	contact_forces_left: ContactSensorCfg= MISSING
	contact_forces_right: ContactSensorCfg= MISSING
	contact_bottom_left: ContactSensorCfg= MISSING
	contact_bottom_right: ContactSensorCfg= MISSING
	contact_target_bottom: ContactSensorCfg= MISSING

	table: AssetBaseCfg = AssetBaseCfg(
		prim_path="{ENV_REGEX_NS}/Table",
		init_state=AssetBaseCfg.InitialStateCfg(
			pos=[0.5, 0.0, 0.0],
			rot=[0.707, 0.0, 0.0, 0.707]
		),
		spawn=UsdFileCfg(
			usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
		),
	)

	# ground plane
	plane: AssetBaseCfg = AssetBaseCfg(
		prim_path="/World/GroundPlane",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
		spawn=GroundPlaneCfg(),
	)

	# dome lighting
	light: AssetBaseCfg = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
	)

	base: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Base",
		spawn=sim_utils.CuboidCfg(
			size=[0.5, 0.5, base_height],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(base_x, 0.0, 0.05),
			rot=(1.0, 0.0, 0.0, 0.0)
		),
	)

	cube_1 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_1",
		spawn=sim_utils.CuboidCfg(
			size=[0.05, 0.05, cube_1_thickness],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			physics_material=RigidBodyMaterialCfg(
				static_friction=0.4,
				dynamic_friction=0.2,
				restitution=0.5
        	),
			mass_props=sim_utils.MassPropertiesCfg(mass=100),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.044, 0.249), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x, cube_1_y, cube_1_z), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	pick_cube_bottom =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_Pick_Bottom",
		spawn=sim_utils.CuboidCfg(
			size=[0.025, 0.1, cube_pick_thickness],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.0), metallic=0.2)
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x, 0.08, cube_pick_bottom_z), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	pick_cube_1 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_Pick_1",
		spawn=sim_utils.CuboidCfg(
			size=[0.018, 0.018, 0.018],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			physics_material=RigidBodyMaterialCfg(
				static_friction=0.3,
				dynamic_friction=0.15,
				restitution=0.3
        	),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0266, 0.2742, 0.133), metallic=0.2),
			activate_contact_sensors=True
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(base_x, 0.05, cube_2_z), rot=(1.0, 0.0, 0.0, 0.0)),
	)


@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""

	# will be set by agent env cfg
	arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
	gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
	"""Observation specifications for the MDP."""

	@configclass
	class PolicyCfg(ObsGroup):
		"""Observations for policy group with state values."""

		actions = ObsTerm(func=mdp.last_action)
		joint_pos = ObsTerm(func=mdp.joint_pos_rel)
		joint_pos_abs = ObsTerm(func=mdp.joint_pos)
		joint_vel = ObsTerm(func=mdp.joint_vel_rel)
		eef_pos = ObsTerm(func=mdp.ee_frame_pos)
		eef_quat = ObsTerm(func=mdp.ee_frame_quat)
		gripper_pos = ObsTerm(func=mdp.gripper_pos)
		invalid_contact = ObsTerm(func=mdp.gripper_contact)
		physics = ObsTerm(func=mdp.object_physics)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	@configclass
	class RGBCameraPolicyCfg(ObsGroup):
		"""Observations for policy group with RGB images."""

		front_cam = ObsTerm(
			func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False}
		)
		wrist_cam = ObsTerm(
			func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
		)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	@configclass
	class SubtaskCfg(ObsGroup):
		"""Observations for subtask group."""

		failure = ObsTerm(
			func=mdp.is_failure,
		)

		success = ObsTerm(
			func=mdp.is_success,
		)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	# observation groups
	policy: PolicyCfg = PolicyCfg()
	rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
	subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class RewardsCfg:
	"""Reward terms for the MDP."""
	# action penalty
	action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

	joint_vel = RewTerm(
		func=mdp.joint_vel_l2,
		weight= -0.5,
		params={"asset_cfg": SceneEntityCfg("robot")},
	)


	target_cube_relative_position =RewTerm(
		func=mdp.rel_pos_12, weight=1, params={"cube_1_name": "cube_1", "cube_2_name": "pick_cube_1" },
	)


	target_cube_relative_xyz_position =RewTerm(
		func=mdp.rel_pos_xyz_l2, weight=2, params={"cube_1_name": "cube_1", "cube_2_name": "pick_cube_1" },
	)

	pick_cube_z = RewTerm(
		func=mdp.abs_z_12, weight=1, params={"object_name": "pick_cube_bottom"}
	)
	
	ee_position = RewTerm(
		func=mdp.ee_frame_out_of_boundary, weight=-10, params={'x_min': 0.37, \
														'x_max': 0.6, 'y_min': -0.1, 'y_max': 0.22}
	)

	failure = RewTerm(
			func=mdp.is_failure, weight = 0
	)

	success = RewTerm(
			func=mdp.is_success, weight = 15
	)

	poured = RewTerm(
			func=mdp.is_poured, weight = 2
	)

	touch_target_cube = RewTerm(
		func=mdp.gripper_contact, weight=-10
	)

	touch_bottom_cube = RewTerm(
		func=mdp.gripper_contact_bottom, weight=2
	)


@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""

	time_out = DoneTerm(func=mdp.time_out, time_out=True)

	ee_position = DoneTerm(func=mdp.ee_frame_out_of_boundary)


@configclass
class RobustEnvCfg(ManagerBasedRLEnvCfg):
	"""Configuration for the stacking environment."""

	# Scene settings
	scene: RobustBlockScene = RobustBlockScene(num_envs=4096, env_spacing=2.5, replicate_physics=False)
	# Basic settings
	observations: ObservationsCfg = ObservationsCfg()
	actions: ActionsCfg = ActionsCfg()
	# MDP settings
	terminations: TerminationsCfg = TerminationsCfg()

	# Unused managers
	commands = None
	rewards: RewardsCfg = RewardsCfg()
	events = None
	curriculum = None

	def __post_init__(self):
		"""Post initialization."""
		# general settings
		self.decimation = 5
		self.episode_length_s = 10.0
		self.rerender_on_reset = True
		
		# simulation settings
		self.sim.dt = 0.01
		self.sim.render_interval = self.decimation

		self.sim.physx.bounce_threshold_velocity = 0.2
		self.sim.physx.bounce_threshold_velocity = 0.01
		self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
		self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
		self.sim.physx.friction_correlation_distance = 0.00625

		self.seed = 0
