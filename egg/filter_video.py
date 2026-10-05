"""Replay recorded robot episodes through the learned egg safety filter and render videos.

    python egg/filter_video.py --episodes data/egg/traj --out logs/egg/videos

Each step t re-grounds the world model on the recorded camera frames and joint
state, scores the base policy's command a_t as the robot did,
  V_t = min_i Q_i(z', pi(z')),  z' ~ p(z_{t+1} | z_t, a_t)
  u_t = logpZO_za(z_t, a_t)
and marks the step unsafe when V_t < value_thr or u_t > uq_thr. The video shows
the two camera views above the value trace; orange frames are steps the filter
overrode on the robot.
"""

import argparse

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from egg.data import normalize
from egg.task import load_world_model, norm_stats
from latentdisturbance import config as config_lib
from latentdisturbance.plot import ROBUST, rgb
from latentdisturbance.sac import SAC
from latentdisturbance.train import DEFAULTS


def load(args):
	cfg = config_lib.parse(["--config", args.config] + args.overrides, defaults=DEFAULTS)
	wm, models = load_world_model(cfg)
	wm.eval()
	for m in models.values():
		m.eval()
	agent = SAC(cfg, cfg.dyn_stoch + cfg.dyn_deter, cfg.policy_action_dim)
	agent.load(args.step, config_lib.resolve(args.filter_run) / "model")
	agent.eval()
	return cfg, wm, models, agent, norm_stats(cfg)


def observation(ep, t, cams):
	obs = {"state": ep["state"][t][None, None].astype(np.float32), "is_first": np.zeros((1, 1)),
		   "is_terminal": np.zeros((1, 1))}
	for c in cams:
		obs[c] = ep[c][t][None, None]
	return obs


@torch.no_grad()
def ground(wm, density, ep, cams, tol=1.0):
	"""Posterior latents along the episode. Teleop pauses on the robot restart the
	latent and are not recorded, so at each step the continued and the restarted
	latent are both computed and the one matching the recorded state density is kept."""
	encode = lambda t: wm.encoder(wm.preprocess(observation(ep, t, cams)))[:, 0]
	ones = torch.ones(1, device=wm._config.device)
	restart = lambda e: wm.dynamics.obs_step(None, None, e, ones, sample=False)[0]
	latents = [restart(encode(0))]
	for t in range(1, len(ep["state"])):
		e = encode(t)
		a = torch.as_tensor(ep["action_norm"][t - 1][None], device=e.device, dtype=torch.float32)
		latent = wm.dynamics.obs_step(latents[-1], a, e, 0 * ones, sample=False)[0]
		if "density" in ep:
			score = lambda lat: float(density(wm.dynamics.get_feat(lat)))
			err = abs(score(latent) - ep["density"][t])
			if err > tol:
				fresh = restart(e)
				if abs(score(fresh) - ep["density"][t]) < err:
					latent = fresh
		latents.append(latent)
	return latents


@torch.no_grad()
def score(wm, models, agent, latents, actions):
	out = {k: [] for k in ("value", "uncertainty", "density", "p_fallen", "p_flipped")}
	for latent, a in zip(latents, actions):
		feat = wm.dynamics.get_feat(latent)
		nxt = wm.dynamics.get_feat(wm.dynamics.img_step(latent, a[None]))
		out["value"].append(agent.value(nxt).item())
		out["uncertainty"].append(models["sa_density"].score_flat(torch.cat([feat, a[None]], -1)).item())
		out["density"].append(models["density"](feat).item())
		out["p_fallen"].append(wm.heads["fallen"](feat).mean.item())
		out["p_flipped"].append(wm.heads["flipped"](feat).mean.item())
	return {k: np.array(v) for k, v in out.items()}


def value_panel(values, t, thr, filtered, width, height=220):
	fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
	ax = fig.add_axes([0.14, 0.24, 0.83, 0.7])
	steps = np.arange(len(values))
	ax.plot(steps[:t + 1], values[:t + 1], color="k", lw=2)
	for s in np.flatnonzero(filtered[1:t + 1]) + 1:
		ax.plot(steps[s - 1:s + 1], values[s - 1:s + 1], color=ROBUST, lw=2.4)
	ax.axhline(thr, color="0.5", ls="--", lw=1)
	ax.set_xlim(0, len(values) - 1)
	ax.set_ylim(min(-1.0, values.min() - 0.1), max(1.0, values.max() + 0.1))
	ax.set_ylabel("V")
	ax.set_xlabel("step")
	fig.canvas.draw()
	img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
	plt.close(fig)
	return img


def render(ep, cams, sig, thr, path, fps):
	filtered = ep["is_filtered"].astype(bool)
	frames = []
	for t in range(len(sig["value"])):
		row = np.concatenate([ep[c][t] for c in cams], 1)
		if filtered[t]:
			cv2.rectangle(row, (0, 0), (row.shape[1] - 1, row.shape[0] - 1), rgb(ROBUST), 6)
		frames.append(np.concatenate([row, value_panel(sig["value"], t, thr, filtered, row.shape[1])], 0))
	imageio.mimsave(path, frames, fps=fps, macro_block_size=8)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--config", default="egg/configs/reach.yaml")
	p.add_argument("--filter_run", default="checkpoints/egg/filter")
	p.add_argument("--step", type=int, default=400000)
	p.add_argument("--episodes", default="data/egg/traj")
	p.add_argument("--out", default="logs/egg/videos")
	p.add_argument("--value_thr", type=float, default=0.2)
	p.add_argument("--uq_thr", type=float, default=1000.0)
	p.add_argument("--max_episodes", type=int, default=0)
	p.add_argument("--fps", type=int, default=15)
	args, args.overrides = p.parse_known_args()

	cfg, wm, models, agent, stats = load(args)
	cams = [f"{c}_cam" for c in cfg.cameras]
	lo, hi = (np.asarray(stats["action"][k], np.float64) for k in ("low", "high"))
	out = config_lib.resolve(args.out)
	out.mkdir(parents=True, exist_ok=True)
	files = sorted(config_lib.resolve(args.episodes).rglob("*.npz"))
	files = files[:args.max_episodes] if args.max_episodes else files

	for path in files:
		ep = dict(np.load(path, allow_pickle=True))
		raw = ep["action_original"].astype(np.float32).copy()
		raw[:, -1] = np.where(raw[:, -1] > 0, 1.0, -1.0)
		actions = torch.as_tensor(normalize(np.clip(raw, lo, hi), lo, hi), device=cfg.device)
		latents = ground(wm, models["density"], ep, cams)
		sig = score(wm, models, agent, latents, actions)
		sig["unsafe"] = (sig["value"] < args.value_thr) | (sig["uncertainty"] > args.uq_thr)
		np.savez(out / f"{path.stem}.npz", **sig)
		render(ep, cams, sig, args.value_thr, out / f"{path.stem}.mp4", args.fps)
		print(f"{path.stem}: {len(sig['value'])} steps, filtered {int(ep['is_filtered'].sum())}", flush=True)


if __name__ == "__main__":
	main()
