"""Plot one replayed egg episode: key frames and the safety value V(x).

    python egg/visualize.py data/egg/traj/traj_0000_20260814_113436_success.npz \
        --signals logs/egg/videos/traj_0000_20260814_113436_success.npz
"""

import argparse
import pathlib

import numpy as np

from latentdisturbance.plot import filter_trace


def main():
	p = argparse.ArgumentParser()
	p.add_argument("episode", help="recorded episode .npz")
	p.add_argument("--signals", required=True, help="signals written by filter_video.py")
	p.add_argument("--value_thr", type=float, default=0.2)
	p.add_argument("--out", default="")
	args = p.parse_args()
	ep, sig = np.load(args.episode, allow_pickle=True), np.load(args.signals)
	frames = np.concatenate([ep["rs_0_cam"], ep["zed_1_cam"]], 2)
	out = args.out or str(pathlib.Path(args.signals).with_suffix(".png"))
	filter_trace(frames, sig["value"], args.value_thr, ep["is_filtered"], out=out)


if __name__ == "__main__":
	main()
