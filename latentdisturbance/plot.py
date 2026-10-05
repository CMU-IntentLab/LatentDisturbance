import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROBUST = "#ff9500"   # robust (LUCID) filter interventions
NOMINAL = "#0048a6"  # nominal filter interventions


def rgb(color):
	return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def filter_trace(frames, value, value_thr, filtered, uncertainty=None, uq_thr=None, out="trace.png", n_frames=6,
				 color=ROBUST):
	"""Key frames above the value (and OOD score) of one filtered episode; filtered steps in `color`."""
	T = len(value)
	idx = np.linspace(0, T - 1, n_frames).astype(int)
	rows = 2 if uncertainty is None else 3
	h, w = np.asarray(frames[0]).shape[:2]
	img_h = 2.2 * h / w
	fig = plt.figure(figsize=(2.2 * n_frames, img_h + 1.8 * (rows - 1) + 0.6))
	grid = fig.add_gridspec(rows, n_frames, height_ratios=[img_h] + [1.8] * (rows - 1), hspace=0.35, wspace=0.05)
	for i, t in enumerate(idx):
		ax = fig.add_subplot(grid[0, i])
		ax.imshow(frames[t])
		ax.set_title(f"t={t}" + ("  filtered" if filtered[t] else ""), fontsize=9, color=color if filtered[t] else "k")
		ax.axis("off")
	series = [("V(x)", value, value_thr)] + ([("OOD score", uncertainty, uq_thr)] if uncertainty is not None else [])
	for r, (label, y, thr) in enumerate(series, start=1):
		ax = fig.add_subplot(grid[r, :])
		ax.plot(np.arange(T), y, color="k", lw=1.8)
		for s in np.flatnonzero(filtered[1:]) + 1:
			ax.plot([s - 1, s], y[s - 1:s + 1], color=color, lw=2.4)
		if thr is not None:
			ax.axhline(thr, color="0.5", ls="--", lw=1)
		ax.vlines(idx, *ax.get_ylim(), colors="0.6", lw=0.6)
		ax.set_xlim(0, T - 1)
		ax.set_ylabel(label)
	ax.set_xlabel("step")
	fig.savefig(out, dpi=130, bbox_inches="tight")
	plt.close(fig)
	print(f"saved {out}")
