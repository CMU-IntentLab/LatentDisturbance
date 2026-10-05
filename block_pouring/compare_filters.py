"""Nominal vs. robust safety filter on the same recorded teleoperation, side by side.

Each recorded trajectory (data/block_pouring/teleop_replay) holds the teleoperator's actions and the full
simulator state, including the unobserved mass and friction of the blocks. For each filter, the initial state is
restored and the recorded actions are proposed step by step as the task policy; the filter overrides them with
its safety policy when V < value_thr or the ensemble uncertainty exceeds uq_thr (held for `hold_steps`), and the
simulator executes whatever the filter chose. The two runs therefore share the initial state, physics and
proposed actions, and differ only in the filter.

    # the released cases where the nominal filter fails and the robust one succeeds
    python block_pouring/compare_filters.py
    # every recorded trajectory; writes videos for all and lists the differing cases in summary.json
    python block_pouring/compare_filters.py --cases all

Per trajectory: <out>/<name>.mp4 (nominal | robust, synchronized steps, value traces below) and <name>.npz.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trajs", default="data/block_pouring/teleop_replay")
parser.add_argument("--cases", default="block_pouring/configs/compare_cases.txt",
					help="text file with one trajectory name per line, or 'all'")
parser.add_argument("--reach_config", default="block_pouring/configs/reach.yaml")
parser.add_argument("--nominal_run", default="checkpoints/block_pouring/filter_nominal")
parser.add_argument("--robust_run", default="checkpoints/block_pouring/filter")
parser.add_argument("--filter_step", type=int, default=400000)
parser.add_argument("--value_thr", type=float, default=0.0)
parser.add_argument("--uq_thr", type=float, default=3.7)
parser.add_argument("--hold_steps", type=int, default=3)
parser.add_argument("--select_window", type=int, default=5, help="trailing steps that decide the outcome")
parser.add_argument("--hd_res", type=int, default=512)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--no_video", action="store_true", help="only outcomes and signals (fast)")
parser.add_argument("--out", default="logs/block_pouring/compare")
parser.add_argument("--seed", type=int, default=0)

from block_pouring.sim import launch  # noqa: E402

args, app = launch(parser)

import json  # noqa: E402

import cv2  # noqa: E402
import imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from block_pouring.filter import SafetyFilter  # noqa: E402
from block_pouring.sim import BlockPouringEnv  # noqa: E402
from latentdisturbance import config as config_lib  # noqa: E402
from latentdisturbance.plot import NOMINAL, ROBUST, rgb  # noqa: E402
from latentdisturbance.train import DEFAULTS  # noqa: E402

FILTERS = ("nominal", "robust")
TITLES = {"nominal": "Nominal", "robust": "LUCID (robust)"}
COLORS = {"nominal": NOMINAL, "robust": ROBUST}
SKIP_FIRST = 1     # index 0 of a recording holds the previous episode's state
RENDER_FLUSH = 30  # renders of the restored scene, so no image history carries over from the previous run
FLUSH_STEPS = 3    # zero-action steps after restoring, so the cameras render the restored state


def outcome(success, failure, window):
	s, f = bool(success[-window:].any()), bool(failure[-window:].any())
	if s and f:
		return "success" if success[-1] else "failure"
	return "success" if s else "failure" if f else "incomplete"


@torch.no_grad()
def run(env, safety, record):
	"""Re-simulate `record` under `safety`; the recorded actions are the proposed (task) actions.
	The filter samples the predicted next latent, so every run is seeded with `--seed` to be reproducible."""
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	env.reset()
	env.restore(record, SKIP_FIRST)
	for _ in range(RENDER_FLUSH):
		env.unwrapped.sim.render()
	zero = torch.zeros(1, 7, device=env.device)
	for _ in range(FLUSH_STEPS):
		obs, _, _, _ = env.step(zero)
	obs["is_first"] = torch.ones_like(obs["is_first"])
	safety.reset(obs)
	log = {k: [] for k in ("front", "wrist", "value", "uncertainty", "filtered", "success", "failure")}
	for proposed in np.asarray(record["action"])[SKIP_FIRST:]:
		proposed = torch.clamp(torch.tensor(proposed, dtype=torch.float32, device=env.device)[None], -1, 1)
		action, info = safety(proposed)
		front, wrist = env.hd_frames()
		log["front"].append(front)
		log["wrist"].append(wrist)
		for k in ("value", "uncertainty", "filtered"):
			log[k].append(info[k])
		log["success"].append(bool(obs["success"][0]))
		log["failure"].append(bool(obs["failure"][0]))
		obs, _, _, _ = env.step(action)
		safety.update(action, obs)
	return {k: np.asarray(v) for k, v in log.items()}


def value_panel(value, filtered, upto, ylim, size, color):
	"""Value trace up to step `upto`, in `color` where the filter overrode the proposed action."""
	w, h = size
	fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
	n = len(value)
	for t in range(1, upto + 1):
		ax.plot([t - 1, t], value[t - 1:t + 1], color=color if filtered[t] else "k", linewidth=1.8)
	idx = np.flatnonzero(filtered[:upto + 1])
	ax.plot(idx, value[idx], linestyle="", marker="o", markersize=3, color=color)
	ax.axhline(args.value_thr, linestyle=":", linewidth=1.4, color="0.5")
	ax.axvline(upto, linewidth=0.8, color="gray", alpha=0.5)
	ax.set_xlim(0, max(1, n - 1))
	ax.set_ylim(*ylim)
	ax.set_xlabel("step")
	ax.set_ylabel("safety value V")
	ax.grid(True, linewidth=0.3)
	fig.tight_layout()
	fig.canvas.draw()
	img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
	plt.close(fig)
	return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def header(text, width, color):
	bar = np.full((36, width, 3), 255, np.uint8)
	cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
	return bar


def write_video(path, logs, outcomes):
	res = args.hd_res
	values = np.concatenate([logs[k]["value"] for k in FILTERS] + [[args.value_thr]])
	pad = 0.05 * max(values.max() - values.min(), 1e-6)
	ylim = (values.min() - pad, values.max() + pad)
	T = len(logs["nominal"]["value"])
	with imageio.get_writer(str(path), fps=args.fps, codec="libx264", macro_block_size=None,
							pixelformat="yuv420p", ffmpeg_params=["-crf", "20"]) as writer:
		for t in range(T):
			cols = []
			for k in FILTERS:
				log = logs[k]
				row = np.hstack([log["front"][t], log["wrist"][t]])
				if log["filtered"][t]:
					cv2.rectangle(row, (0, 0), (2 * res - 1, res - 1), rgb(COLORS[k]), 6)
				title = header(f"{TITLES[k]}: {outcomes[k]}    step {t}", 2 * res, rgb(COLORS[k]))
				graph = value_panel(log["value"], log["filtered"], t, ylim, (2 * res, res // 2), COLORS[k])
				cols.append(np.vstack([title, row, graph]))
			gap = np.full((cols[0].shape[0], 12, 3), 255, np.uint8)
			writer.append_data(np.hstack([cols[0], gap, cols[1]]))


def main():
	trajs = config_lib.resolve(args.trajs)
	if args.cases == "all":
		names = sorted(p.stem for p in trajs.glob("*.npz"))
	else:
		names = [n.strip() for n in config_lib.resolve(args.cases).read_text().splitlines() if n.strip()]

	env = BlockPouringEnv(replay=True, hd_res=args.hd_res)
	cfg = config_lib.parse(["--config", args.reach_config], defaults=DEFAULTS)
	kw = dict(value_thr=args.value_thr, uq_thr=args.uq_thr, hold_steps=args.hold_steps, worst_case=False)
	nominal = SafetyFilter(cfg, config_lib.resolve(args.nominal_run) / "model", args.filter_step, **kw)
	robust = SafetyFilter(cfg, config_lib.resolve(args.robust_run) / "model", args.filter_step,
						  world_model=(nominal.wm, nominal.ensemble), **kw)
	out = config_lib.resolve(args.out)
	out.mkdir(parents=True, exist_ok=True)

	summary = []
	for i, name in enumerate(names):
		if not app.is_running():
			break
		record = np.load(trajs / f"{name}.npz", allow_pickle=True)
		logs = {"nominal": run(env, nominal, record), "robust": run(env, robust, record)}
		outcomes = {k: outcome(logs[k]["success"], logs[k]["failure"], args.select_window) for k in FILTERS}
		differs = outcomes["nominal"] == "failure" and outcomes["robust"] == "success"
		print(f"[{i + 1}/{len(names)}] {name}: nominal {outcomes['nominal']}, robust {outcomes['robust']}"
			  + ("  <- differs" if differs else ""), flush=True)
		if not args.no_video:
			write_video(out / f"{name}.mp4", logs, outcomes)
		np.savez(out / f"{name}.npz", **{f"{k}_{key}": v for k in FILTERS for key, v in logs[k].items()
										 if key not in ("front", "wrist")})
		summary.append({"name": name, **{f"{k}_outcome": outcomes[k] for k in FILTERS},
						**{f"{k}_filtered_steps": int(logs[k]["filtered"].sum()) for k in FILTERS},
						"nominal_fails_robust_succeeds": differs})
		(out / "summary.json").write_text(json.dumps(summary, indent=2))
	n = sum(s["nominal_fails_robust_succeeds"] for s in summary)
	print(f"{n} / {len(summary)} trajectories: nominal fails, robust succeeds. Summary: {out / 'summary.json'}")
	env.close()
	app.close()


if __name__ == "__main__":
	main()
