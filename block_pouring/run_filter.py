"""Safety filtering demo in Isaac Sim with a diffusion policy or SpaceMouse teleoperation.

    # base policy, headless, videos + per-step signals in --out
    python block_pouring/run_filter.py --policy diffusion --episodes 5
    # teleoperation (needs a display and a 3Dconnexion SpaceMouse)
    python block_pouring/run_filter.py --policy spacemouse --gui
    # same, without the filter
    python block_pouring/run_filter.py --policy diffusion --use_filter False
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=["diffusion", "spacemouse"], default="diffusion")
parser.add_argument("--use_filter", type=lambda s: s == "True", default=True)
parser.add_argument("--reach_config", default="block_pouring/configs/reach.yaml")
parser.add_argument("--filter_run", default="checkpoints/block_pouring/filter")
parser.add_argument("--filter_step", type=int, default=400000)
parser.add_argument("--value_thr", type=float, default=0.0)
parser.add_argument("--uq_thr", type=float, default=3.7)
parser.add_argument("--hold_steps", type=int, default=3)
parser.add_argument("--policy_run", default="checkpoints/block_pouring/diffusion_policy")
parser.add_argument("--policy_step", type=int, default=1000)
parser.add_argument("--policy_norm", default="checkpoints/block_pouring/diffusion_policy/normalization.npz")
parser.add_argument("--act_steps", type=int, default=4)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=250)
parser.add_argument("--out", default="logs/block_pouring/demo")
parser.add_argument("--seed", type=int, default=0)

from block_pouring.sim import launch  # noqa: E402

args, app = launch(parser)

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from block_pouring.sim import BlockPouringEnv, frame  # noqa: E402
from latentdisturbance import config as config_lib  # noqa: E402
from latentdisturbance.plot import ROBUST, rgb  # noqa: E402
from latentdisturbance.train import DEFAULTS  # noqa: E402


class DiffusionPolicy:
	"""Image diffusion policy; executes `act_steps` of each predicted 8-step chunk."""

	def __init__(self, run_dir, step, norm_path, act_steps, device="cuda"):
		import hydra
		from omegaconf import OmegaConf

		OmegaConf.register_new_resolver("eval", eval, replace=True)
		run_dir = config_lib.resolve(run_dir)
		cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
		self.model = hydra.utils.instantiate(cfg.model)
		state = torch.load(run_dir / "checkpoint" / f"state_{step}.pt", map_location="cpu", weights_only=True)
		self.model.load_state_dict(state["ema"])
		self.model.to(device).eval()
		norm = np.load(config_lib.resolve(norm_path))
		self.stats = {k: torch.tensor(norm[k], dtype=torch.float32, device=device) for k in norm.files}
		self.size = int(cfg.shape_meta.obs.rgb.shape[-1])
		self.act_steps, self.device, self.queue = act_steps, device, []

	def reset(self):
		self.queue = []

	def replan(self):
		self.queue = []

	def _resize(self, img):
		img = cv2.resize(img[0].cpu().numpy().astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_AREA)
		return img.transpose(2, 0, 1)

	@torch.no_grad()
	def __call__(self, obs):
		if not self.queue:
			s = self.stats
			rgb = np.concatenate([self._resize(obs["front_cam"]), self._resize(obs["wrist_cam"])], 0)
			state = torch.cat([obs["eef_pos"][0], obs["eef_quat"][0], obs["gripper_pos"][0]]).float()
			state = 2 * (state - s["obs_min"]) / (s["obs_max"] - s["obs_min"] + 1e-6) - 1
			cond = {"state": state[None, None], "rgb": torch.from_numpy(rgb).to(self.device)[None, None]}
			traj = self.model(cond=cond, deterministic=True).trajectories
			traj = (traj + 1) / 2 * (s["action_max"] - s["action_min"]) + s["action_min"]
			self.queue = [traj[:, i] for i in range(min(self.act_steps, traj.shape[1]))]
		return torch.clamp(self.queue.pop(0), -1, 1)


class SpaceMousePolicy:
	def __init__(self, device="cuda"):
		from isaaclab.devices import Se3SpaceMouse

		self.device = Se3SpaceMouse(pos_sensitivity=1.0, rot_sensitivity=1.0)
		self.torch_device = device

	def reset(self):
		self.device.reset()

	def replan(self):
		pass

	def __call__(self, obs):
		delta, close = self.device.advance()
		action = np.concatenate([delta.astype(np.float32), [-1.0 if close else 1.0]])
		return torch.clamp(torch.tensor(action, device=self.torch_device)[None], -1, 1)


def overlay(img, info):
	img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
	if info is None:
		return img
	color = rgb(ROBUST) if info["filtered"] else (255, 255, 255)
	text = f"V {info['value']:+.2f}  u {info['uncertainty']:.2f}" + ("  FILTER" if info["filtered"] else "")
	cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
	cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
	return img


def main():
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	env = BlockPouringEnv()
	if args.policy == "diffusion":
		policy = DiffusionPolicy(args.policy_run, args.policy_step, args.policy_norm, args.act_steps)
	else:
		policy = SpaceMousePolicy()
	safety = None
	if args.use_filter:
		from block_pouring.filter import SafetyFilter

		cfg = config_lib.parse(["--config", args.reach_config], defaults=DEFAULTS)
		safety = SafetyFilter(cfg, config_lib.resolve(args.filter_run) / "model", args.filter_step,
							  args.value_thr, args.uq_thr, args.hold_steps)
	out = config_lib.resolve(args.out)
	out.mkdir(parents=True, exist_ok=True)

	for ep in range(args.episodes):
		obs = env.reset()
		policy.reset()
		if safety is not None:
			safety.reset(obs)
		frames, log, outcome = [], [], "timeout"
		for _ in range(args.max_steps):
			if not app.is_running():
				break
			action = policy(obs)
			info = None
			if safety is not None:
				action, info = safety(action)
				if info["filtered"]:
					policy.replan()
				log.append(info)
			frames.append(overlay(frame(obs), info))
			obs, _, done, step_info = env.step(action)
			if safety is not None:
				safety.update(action, obs)
			if bool(done[0]):
				outcome = env.outcome(step_info)
				break
		name = f"ep{ep:03d}_{outcome}"
		imageio.mimsave(out / f"{name}.mp4", frames, fps=20)
		if log:
			np.savez(out / f"{name}.npz", **{k: np.array([x[k] for x in log]) for k in log[0]})
		n_filtered = sum(x["filtered"] for x in log)
		print(f"[{ep + 1}/{args.episodes}] {outcome}, {len(frames)} steps, filtered {n_filtered}", flush=True)
	env.close()
	app.close()


if __name__ == "__main__":
	main()
