"""Plot one episode recorded by run_filter.py: key frames and the safety value V(x).

    python block_pouring/visualize.py logs/block_pouring/demo/ep000_success
"""

import argparse

import imageio
import numpy as np

from latentdisturbance.plot import filter_trace


def main():
	p = argparse.ArgumentParser()
	p.add_argument("episode", help="path prefix of the .mp4 / .npz pair")
	p.add_argument("--value_thr", type=float, default=0.0)
	args = p.parse_args()
	sig = np.load(f"{args.episode}.npz")
	frames = [f for f in imageio.get_reader(f"{args.episode}.mp4")][:len(sig["value"])]
	filter_trace(frames, sig["value"], args.value_thr, sig["filtered"], out=f"{args.episode}.png")


if __name__ == "__main__":
	main()
