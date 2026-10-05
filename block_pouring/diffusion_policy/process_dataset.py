"""Successful block-pouring episodes -> DPPO stitched dataset.

    python block_pouring/diffusion_policy/process_dataset.py --src data/block_pouring/rollouts_0804 --out data/block_pouring/diffusion_policy/train.npz

states  (N, 9)        eef_pos + eef_quat + gripper_pos, scaled to [-1, 1]
actions (N, 7)        scaled to [-1, 1]
images  (N, 6, H, W)  uint8, [front, wrist] resized to H x W
A normalization.npz with the min/max statistics is written next to the output.
"""

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

STATE_KEYS = ("eef_pos", "eef_quat", "gripper_pos")


def resize(frames, size):
	return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for f in frames])


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--src", nargs="+", required=True)
	p.add_argument("--out", required=True)
	p.add_argument("--img_size", type=int, default=96)
	args = p.parse_args()

	files = [f for d in args.src for f in sorted(glob.glob(os.path.join(d, "success_*.npz")))]
	assert files, f"no success_*.npz under {args.src}"
	states, actions, images, lengths = [], [], [], []
	for path in tqdm(files):
		ep = np.load(path)
		states.append(np.concatenate([ep[k].astype(np.float32) for k in STATE_KEYS], 1))
		actions.append(ep["action"].astype(np.float32))
		images.append(np.concatenate([resize(ep["front_cam"], args.img_size), resize(ep["wrist_cam"], args.img_size)], 1))
		lengths.append(len(actions[-1]))
	states, actions, images = np.concatenate(states), np.concatenate(actions), np.concatenate(images)
	stats = dict(obs_min=states.min(0), obs_max=states.max(0), action_min=actions.min(0), action_max=actions.max(0))
	scale = lambda x, lo, hi: (2 * (x - lo) / (hi - lo + 1e-6) - 1).astype(np.float32)

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	np.savez_compressed(args.out, states=scale(states, stats["obs_min"], stats["obs_max"]),
						actions=scale(actions, stats["action_min"], stats["action_max"]), images=images,
						traj_lengths=np.array(lengths, np.int64))
	np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(args.out)), "normalization.npz"), **stats)
	print(f"{len(files)} episodes, {len(states)} steps -> {args.out}")


if __name__ == "__main__":
	main()
