"""Isaac Lab environment for block pouring with the world model's observation format.

Import after the Omniverse app is launched (see `launch`).
"""

import numpy as np
import torch

# normalized action in [-1, 1]^7 -> (dx, dy, dz, droll, dpitch, dyaw, gripper)
ACTION_SCALE = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1)


def launch(parser, gui=False):
	"""Parse Isaac app arguments and start the simulator (headless unless --gui). Returns (args, app)."""
	from isaaclab.app import AppLauncher

	AppLauncher.add_app_launcher_args(parser)
	parser.add_argument("--gui", action="store_true", default=gui)
	parser.set_defaults(enable_cameras=True)
	args = parser.parse_args()
	args.headless = not args.gui
	app = AppLauncher(args).app
	return args, app


class BlockPouringEnv:
	def __init__(self, device="cuda", randomize_pose=True, randomize_physics=True, time_out=True, terminate=True,
				 replay=False, hd_res=0):
		"""`replay`: no randomization and no terminations, for re-simulating recorded states (see `restore`).
		`hd_res` > 0 adds square front / wrist recording cameras at that resolution (see `hd_frames`)."""
		import copy

		import gymnasium as gym
		from isaaclab.managers import TerminationTermCfg as DoneTerm
		from isaaclab.sensors import CameraCfg
		from isaaclab_tasks.utils import parse_env_cfg

		import block_pouring.env as task
		from block_pouring.env import mdp

		cfg = parse_env_cfg(task.TASK_ID, device=device, num_envs=1, use_fabric=True)
		params = cfg.events.reset_partial_and_randomize_physics.params
		params["randomize_pose"] = randomize_pose and not replay
		params["randomize_physics"] = randomize_physics and not replay
		if replay:
			cfg.sim.physx.enable_enhanced_determinism = True
			for name in ("success", "failure", "time_out", "ee_position"):
				setattr(cfg.terminations, name, None)
		else:
			if terminate:
				cfg.terminations.success = DoneTerm(func=mdp.is_success)
				cfg.terminations.failure = DoneTerm(func=mdp.is_failure)
			if not time_out:
				cfg.terminations.time_out = None
		for cam in ("front_cam", "wrist_cam") if hd_res else ():
			base = getattr(cfg.scene, cam)
			setattr(cfg.scene, f"hd_{cam}", CameraCfg(
				prim_path="{ENV_REGEX_NS}/hd_" + cam, update_period=base.update_period, height=hd_res, width=hd_res,
				data_types=["rgb"], spawn=copy.deepcopy(base.spawn),
				offset=CameraCfg.OffsetCfg(pos=tuple(base.offset.pos), rot=tuple(base.offset.rot),
										   convention=base.offset.convention)))
		self.env = gym.make(task.TASK_ID, cfg=cfg)
		self.device = device
		self.scale = torch.tensor(ACTION_SCALE, device=device)

	def restore(self, record, step):
		"""Write recorded state `step` (robot, objects and their mass / material) into the simulator."""
		scene = self.env.unwrapped.scene
		ids = torch.tensor([0], device=self.device)

		def t(x):
			return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)[None]

		robot = scene["robot"]
		robot.write_root_state_to_sim(t(record["robot_root_state"][step]), env_ids=ids)
		robot.write_joint_state_to_sim(t(record["robot_joint_pos"][step]), t(record["robot_joint_vel"][step]), env_ids=ids)
		for name in (str(n) for n in record["object_names"]):
			obj = scene[name]
			obj.write_root_state_to_sim(t(record[f"obj__{name}__root_state"][step]), env_ids=ids)
			view = obj.root_physx_view
			view.set_masses(t(record[f"obj__{name}__mass"][step]).cpu(), ids.cpu())
			materials = view.get_material_properties()
			materials[0, :] = t(record[f"obj__{name}__material"][step]).cpu()
			view.set_material_properties(materials, ids.cpu())
		scene.write_data_to_sim()
		self.env.unwrapped.sim.forward()

	def hd_frames(self):
		"""(front, wrist) uint8 HxWx3 from the recording cameras."""
		scene = self.env.unwrapped.scene
		return tuple(scene[f"hd_{cam}"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
					 for cam in ("front_cam", "wrist_cam"))

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def _obs(self, obs, first, last=None, terminal=None):
		out = {k: v[..., :3] for k, v in obs["rgb_camera"].items()}
		for k in ("eef_pos", "eef_quat", "gripper_pos", "physics"):
			out[k] = obs["policy"][k]
		out["failure"] = obs["subtask_terms"]["failure"]
		out["success"] = obs["subtask_terms"]["success"]
		n = out["eef_pos"].shape[0]
		zeros = torch.zeros(n, dtype=torch.int32, device=self.device)
		out["is_first"] = zeros + int(first)
		out["is_last"] = zeros if last is None else last.int()
		out["is_terminal"] = zeros if terminal is None else terminal.int()
		return out

	def reset(self):
		obs, _ = self.env.reset()
		return self._obs(obs, True)

	def step(self, action):
		"""`action` in [-1, 1]^7 (last entry: gripper, negative closes)."""
		action = torch.clamp(action.to(self.device), -1, 1) * self.scale
		obs, reward, terminated, truncated, info = self.env.step(action)
		done = terminated | truncated
		return self._obs(obs, False, done, terminated), reward, done, info

	@staticmethod
	def outcome(info):
		log = info.get("log", {}) if isinstance(info, dict) else {}
		if log.get("Episode_Termination/success", 0):
			return "success"
		if log.get("Episode_Termination/failure", 0):
			return "failure"
		return "timeout"

	def close(self):
		self.env.close()


def frame(obs):
	"""Front and wrist cameras side by side, uint8 HxWx3."""
	cams = [obs[k][0].detach().cpu().numpy().astype(np.uint8) for k in ("front_cam", "wrist_cam")]
	return np.concatenate(cams, 1)
