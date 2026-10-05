"""Collect block-pouring demonstrations with a 3Dconnexion SpaceMouse.

    python block_pouring/collect_demos.py --out data/block_pouring/my_demos

SpaceMouse: translate/rotate the gripper, left button toggles the gripper.
Keyboard (sim window): K saves the episode and resets, L discards and resets.
Episodes are saved as success_*.npz / failure_*.npz in the world-model format,
where action[t] is the command that led to observation t (action[0] = 0).
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/block_pouring/my_demos")
parser.add_argument("--randomize_physics", type=lambda s: s == "True", default=True)

from block_pouring.sim import launch  # noqa: E402

args, app = launch(parser, gui=True)

import datetime  # noqa: E402

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse  # noqa: E402

from block_pouring.sim import BlockPouringEnv  # noqa: E402
from latentdisturbance.config import resolve  # noqa: E402

KEYS = ("front_cam", "wrist_cam", "eef_pos", "eef_quat", "gripper_pos", "physics", "success", "failure",
		"is_first", "is_last", "is_terminal")


class Recorder:
	def __init__(self, out):
		self.out = resolve(out)
		self.out.mkdir(parents=True, exist_ok=True)
		self.count = 0
		self.clear()

	def clear(self):
		self.ep = {k: [] for k in (*KEYS, "action", "reward", "discount", "done")}

	def add(self, obs, action, reward, done):
		for k in KEYS:
			self.ep[k].append(obs[k][0].cpu().numpy())
		self.ep["action"].append(action)
		self.ep["reward"].append(np.float32(reward))
		self.ep["discount"].append(np.float32(1.0))
		self.ep["done"].append(np.float32(done))

	def save(self, success):
		self.count += 1
		stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		name = f"{'success' if success else 'failure'}_{self.count:03d}_{stamp}"
		ep = {k: np.asarray(v, dtype=np.uint8 if k.endswith("cam") else np.float32) for k, v in self.ep.items()}
		np.savez_compressed(self.out / f"{name}.npz", **ep)
		imageio.mimsave(self.out / f"{name}.mp4", [np.concatenate([f, w], 1) for f, w in zip(ep["front_cam"], ep["wrist_cam"])], fps=20)
		print(f"saved {name} ({len(ep['action'])} steps)", flush=True)


def main():
	env = BlockPouringEnv(randomize_physics=args.randomize_physics, time_out=False, terminate=False)
	mouse = Se3SpaceMouse(pos_sensitivity=1.0, rot_sensitivity=1.0)
	keyboard = Se3Keyboard()
	rec = Recorder(args.out)
	state = {"obs": None}

	def reset(save):
		if save and len(rec.ep["action"]) > 1:
			rec.save(bool(rec.ep["success"][-1]))
		rec.clear()
		obs = env.reset()
		mouse.reset()
		rec.add(obs, np.zeros(7, np.float32), 0.0, 0.0)
		state["obs"] = obs

	keyboard.add_callback("K", lambda: reset(True))
	keyboard.add_callback("L", lambda: reset(False))
	reset(False)

	while app.is_running():
		with torch.inference_mode():
			delta, close = mouse.advance()
			action = np.clip(np.concatenate([delta.astype(np.float32), [-1.0 if close else 1.0]]), -1, 1).astype(np.float32)
			obs, reward, done, _ = env.step(torch.tensor(action, device=env.device)[None])
			if bool(done[0]):
				rec.ep["is_terminal"][-1] = obs["is_terminal"][0].cpu().numpy()
				rec.ep["done"][-1] = np.float32(1.0)
				rec.save(bool(obs["success"][0]))
				rec.clear()
				obs = env.reset()
				mouse.reset()
				rec.add(obs, np.zeros(7, np.float32), 0.0, 0.0)
			else:
				rec.add(obs, action, float(reward[0]), 0.0)
	env.close()
	app.close()


if __name__ == "__main__":
	main()
