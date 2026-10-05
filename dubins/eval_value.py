"""Score a learned value function as a classifier of the ground-truth BRT.

    python dubins/eval_value.py --config dubins/configs/reach_naughty.yaml \
        --run logs/dubins/reach/<run> --step 90000

The grid is x, y in [-1.3, 1.3] on the BRT's native spacing and 51 headings.
V(x) < 0 is predicted unsafe; the BRT value < 0 is the true unsafe set.
Naughty uses the zero-steering worst-case BRT, positional the d=0.3 disturbance BRT.
"""

import json

import numpy as np
import torch
from tqdm import tqdm

from dubins import dynamics as dyn
from dubins.task import load_world_model
from latentdisturbance import config as config_lib
from latentdisturbance.sac import SAC
from latentdisturbance.train import DEFAULTS

BRT_EXTENT = 2.0
BRT_CELLS = 51


def grid_coords(extent):
	dx = 2 * BRT_EXTENT / (BRT_CELLS - 1)
	ks = np.arange(-int(np.floor(extent / dx + 1e-9)), int(np.floor(extent / dx + 1e-9)) + 1)
	return (ks * dx).astype(np.float32), ks + BRT_CELLS // 2


@torch.no_grad()
def encode_states(wm, states, chunk=1024):
	"""Posterior latent of a single rendered frame per state."""
	render = dyn.Renderer()
	feats = []
	for i in tqdm(range(0, len(states), chunk), desc="encoding grid"):
		imgs = np.stack([render(s) for s in states[i:i + chunk]])[:, None]
		n = len(imgs)
		data = wm.preprocess({"image": imgs, "action": np.zeros((n, 1, 1)), "is_first": np.ones((n, 1)),
							  "is_terminal": np.zeros((n, 1))})
		post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
		feats.append(wm.dynamics.get_feat(post)[:, 0])
	return torch.cat(feats)


@torch.no_grad()
def value(agent, feat, chunk=8192):
	out = []
	for i in range(0, len(feat), chunk):
		f = feat[i:i + chunk]
		(mu, _), _ = agent.actor(f)
		out.append(agent.critic1(f, torch.tanh(mu)).squeeze(-1))
	return torch.cat(out).cpu().numpy()


def load_agent(cfg, run, step):
	stoch = cfg.dyn_stoch * cfg.dyn_discrete if cfg.dyn_discrete else cfg.dyn_stoch
	agent = SAC(cfg, stoch + cfg.dyn_deter, cfg.policy_action_dim)
	agent.load(step, config_lib.resolve(run) / "model")
	agent.eval()
	return agent


def metrics(pred_unsafe, true_unsafe):
	p, t = pred_unsafe.reshape(-1), true_unsafe.reshape(-1)
	tp, tn = int((p & t).sum()), int((~p & ~t).sum())
	fp, fn = int((p & ~t).sum()), int((~p & t).sum())
	tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
	return {"accuracy": (tp + tn) / len(p), "balanced_accuracy": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr,
			"fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)}


def main():
	cfg = config_lib.parse(defaults={**DEFAULTS, "run": "", "step": 0, "eval_extent": 1.3, "feat_cache": "", "out": ""})
	wm, _ = load_world_model(cfg)
	wm.eval()
	xs, idx = grid_coords(cfg.eval_extent)
	thetas = np.linspace(0, 2 * np.pi, BRT_CELLS)
	X, Y, TH = np.meshgrid(xs, xs, thetas, indexing="ij")
	states = np.stack([X, Y, TH], -1).reshape(-1, 3)

	cache = config_lib.resolve(cfg.feat_cache) if cfg.feat_cache else None
	if cache is not None and cache.exists():
		feat = torch.as_tensor(np.load(cache), device=cfg.device)
	else:
		feat = encode_states(wm, states)
		if cache is not None:
			cache.parent.mkdir(parents=True, exist_ok=True)
			np.save(cache, feat.cpu().numpy())

	agent = load_agent(cfg, cfg.run, cfg.step)
	V = value(agent, feat).reshape(X.shape)
	gt = np.load(config_lib.resolve(cfg.gt_brt))[np.ix_(idx, idx, np.arange(BRT_CELLS))]
	result = metrics(V < 0, gt < 0)
	print(json.dumps({k: round(v, 4) for k, v in result.items()}))
	if cfg.out:
		out = config_lib.resolve(cfg.out)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
