"""Normalization statistics for the egg hdf5 episodes.

    python egg/norm_stats.py --dirs data/egg/260709_egg ... --out checkpoints/egg/norm_stats.json

Delta-pose bounds are the symmetric `clip_pct` percentile of |action| per dimension,
so outlier teleop spikes do not compress the usable range; the gripper keeps its
min / max. State bounds are min / max of joints + gripper width.
"""

import argparse
import json
import pathlib

import cv2
import h5py
import numpy as np


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--dirs", nargs="+", required=True)
	p.add_argument("--out", required=True)
	p.add_argument("--clip_pct", type=float, default=99.99)
	args = p.parse_args()

	files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("*.hdf5"))]
	actions, states, cams = [], [], None
	for path in files:
		with h5py.File(path, "r") as f:
			actions.append(np.asarray(f["data/actions"][()], np.float64))
			states.append(np.concatenate([f["data/joint_states"][()], f["data/gripper_states"][()][:, None]], 1))
			if cams is None:
				cams = sorted(k[len("camera_"):] for k in f["data"] if k.startswith("camera_"))
				hw = list(cv2.imdecode(np.asarray(f[f"data/camera_{cams[0]}"][0], np.uint8), cv2.IMREAD_COLOR).shape[:2])
	a, s = np.concatenate(actions), np.concatenate(states)
	lo, hi = a.min(0), a.max(0)
	bound = np.percentile(np.abs(a[:, :6]), args.clip_pct, axis=0)
	lo[:6], hi[:6] = -bound, bound
	stats = {
		"action": {"low": lo.tolist(), "high": hi.tolist(), "clip_pct": args.clip_pct,
				   "clipped_frac": ((a < lo) | (a > hi)).mean(0).tolist()},
		"state": {"low": s.min(0).tolist(), "high": s.max(0).tolist()},
		"state_keys": ["joint_states x7", "gripper_width"],
		"cameras": cams, "image_hw": hw, "color": "rgb",
		"n_episodes": len(files), "n_frames": len(a),
	}
	pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	pathlib.Path(args.out).write_text(json.dumps(stats, indent=2))
	print(f"{len(files)} episodes, {len(a)} frames -> {args.out}")


if __name__ == "__main__":
	main()
