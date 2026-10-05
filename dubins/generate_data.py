"""Offline Dubins car datasets under system disturbances.

naughty: the executed steering is flipped with probability `flip_prob`, while
         the dataset records the commanded steering.
positional: a uniform positional disturbance in [-dist_bound, dist_bound]^2 is
         added at every step.

    python dubins/generate_data.py --mode naughty --out data/dubins/naughty
    python dubins/generate_data.py --mode positional --out data/dubins/positional
"""

import argparse
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

from dubins import dynamics as dyn


def rollout(rng, render, mode, length, flip_prob, dist_bound):
	state = dyn.sample_init_state(rng)
	images, states, actions, dones = [], [], [], []
	for t in range(length):
		action = rng.uniform(-dyn.TURN_RATE, dyn.TURN_RATE)
		executed, disturbance = action, None
		if mode == "naughty" and rng.random() < flip_prob:
			executed = -action
		if mode == "positional":
			disturbance = rng.uniform(-dist_bound, dist_bound, size=2)
		images.append(render(state))
		states.append(state)
		actions.append([action])
		done = t == length - 1 or bool(dyn.out_of_bounds(state))
		dones.append(float(done))
		state = dyn.step(state, executed, disturbance)
		if done:
			break
	return np.array(images, np.uint8), np.array(states, np.float32), np.array(actions, np.float32), np.array(dones, np.float32)


def generate(path, n, args, rng, render):
	path.parent.mkdir(parents=True, exist_ok=True)
	with h5py.File(path, "w") as f:
		f.attrs["num_trajectories"] = n
		for i in tqdm(range(n), desc=path.name):
			img, s, a, d = rollout(rng, render, args.mode, args.length, args.flip_prob, args.dist_bound)
			name = f"traj_{i:06d}"
			f.create_dataset(f"images/{name}", data=img, compression="gzip")
			f.create_dataset(f"states/{name}", data=s, compression="gzip")
			f.create_dataset(f"actions/{name}", data=a, compression="gzip")
			f.create_dataset(f"dones/{name}", data=d, compression="gzip")
	print(f"wrote {n} trajectories to {path}")


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--mode", choices=["naughty", "positional"], required=True)
	p.add_argument("--out", required=True, help="output prefix")
	p.add_argument("--num_trajs", type=int, default=5000)
	p.add_argument("--num_calib", type=int, default=500)
	p.add_argument("--length", type=int, default=100)
	p.add_argument("--flip_prob", type=float, default=0.4)
	p.add_argument("--dist_bound", type=float, default=0.3)
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args()

	rng = np.random.default_rng(args.seed)
	render = dyn.Renderer()
	generate(pathlib.Path(f"{args.out}_train.hdf5"), args.num_trajs, args, rng, render)
	if args.num_calib:
		generate(pathlib.Path(f"{args.out}_calib.hdf5"), args.num_calib, args, rng, render)


if __name__ == "__main__":
	main()
