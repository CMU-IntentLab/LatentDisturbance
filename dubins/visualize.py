"""Value-function slices of a learned Dubins safety filter with the BRT boundary.

    python dubins/visualize.py --config dubins/configs/reach_naughty.yaml \
        --run logs/dubins/reach/<run> --step 90000 --out figures/dubins_naughty.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dubins.eval_value import BRT_CELLS, encode_states, grid_coords, load_agent, value
from dubins.task import load_world_model
from latentdisturbance import config as config_lib
from latentdisturbance.train import DEFAULTS


def main():
	cfg = config_lib.parse(defaults={**DEFAULTS, "run": "", "step": 0, "out": "dubins_value.png", "extent": 1.3,
									 "thetas": [0.0, 0.785, 1.571]})
	wm, _ = load_world_model(cfg)
	wm.eval()
	agent = load_agent(cfg, cfg.run, cfg.step)
	xs, idx = grid_coords(cfg.extent)
	gt = np.load(config_lib.resolve(cfg.gt_brt))
	brt_thetas = np.linspace(0, 2 * np.pi, BRT_CELLS)
	X, Y = np.meshgrid(xs, xs, indexing="ij")

	fig, axes = plt.subplots(2, len(cfg.thetas), figsize=(3.2 * len(cfg.thetas), 6.4))
	for i, th in enumerate(cfg.thetas):
		states = np.stack([X, Y, np.full_like(X, th)], -1).reshape(-1, 3)
		V = value(agent, encode_states(wm, states)).reshape(X.shape)
		k = int(np.argmin(np.abs(brt_thetas - th)))
		gt_slice = gt[np.ix_(idx, idx, [k])][..., 0]
		for row, img in enumerate((V, np.where(V > 0, 1.0, -1.0))):
			ax = axes[row, i]
			ax.imshow(img.T, origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap="seismic", vmin=-1, vmax=1)
			ax.contour(X, Y, gt_slice, levels=[0], colors="k", linewidths=2)
			ax.set_xticks([])
			ax.set_yticks([])
		axes[0, i].set_title(f"theta = {th:.2f}")
	axes[0, 0].set_ylabel("V(x)")
	axes[1, 0].set_ylabel("V(x) > 0")
	fig.tight_layout()
	out = config_lib.resolve(cfg.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=150)
	print(f"saved {out}")


if __name__ == "__main__":
	main()
