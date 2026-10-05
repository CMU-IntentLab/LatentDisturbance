"""Trajectory-level conformal calibration of the uncertainty set.

    python -m latentdisturbance.calibrate --config dubins/configs/reach_naughty.yaml \
        --calib_data data/dubins/naughty_calib.hdf5 --alpha 0.01

Per transition, the nonconformity scores are
  kl_radius            KL(encoder posterior || latent dynamics prior), mean over latent dims
  ood_threshold        OOD score (logpZO) of the encoded latent
  epistemic_threshold  state-action epistemic uncertainty, if the config uses the UNISafe margin
Each trajectory is scored by its maximum, and the threshold is the conformal
(1 - alpha) quantile over held-out trajectories.
"""

import importlib
import math

import numpy as np
import torch

from . import config as config_lib
from .sac import _cat_dist, _normal_dist
from .train import DEFAULTS


def conformal_quantile(scores, alpha):
	s = np.sort(np.asarray(scores))
	k = min(max(math.ceil((len(s) + 1) * (1 - alpha)) - 1, 0), len(s) - 1)
	return float(s[k])


def latent_kl(post, prior):
	"""KL(post || prior) averaged over latent dimensions, as in the KL ball of the disturbance."""
	if "logit" in post:
		p, q = _cat_dist(post["logit"], event_dims=0), _cat_dist(prior["logit"], event_dims=0)
	else:
		p, q = _normal_dist(post["mean"], post["std"], 0), _normal_dist(prior["mean"], prior["std"], 0)
	return torch.distributions.kl.kl_divergence(p, q).mean(-1)


@torch.no_grad()
def trajectory_scores(cfg, wm, models, data):
	data = wm.preprocess(data)
	post, prior = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
	feat = wm.dynamics.get_feat(post)[0]
	out = {"kl_radius": latent_kl(post, prior)[0, 1:]}
	if models.get("density") is not None:
		out["ood_threshold"] = models["density"](feat)
	sa = {"ensemble": models.get("ensemble"), "density": models.get("sa_density")}.get(cfg.epistemic_margin)
	if sa is not None:
		x = torch.cat([feat[:-1], data["action"][0, 1:]], -1)
		out["epistemic_threshold"] = sa.score_flat(x) if hasattr(sa, "score_flat") else sa.intrinsic_reward_penn(x)[:, 0]
	return {k: float(v.max()) for k, v in out.items()}


def main():
	cfg = config_lib.parse(defaults={**DEFAULTS, "calib_data": [""], "alpha": 0.01})
	task = importlib.import_module(f"{cfg.task}.task")
	wm, models = task.load_world_model(cfg)
	wm.eval()
	scores = {}
	for data in task.calibration_trajectories(cfg, [config_lib.resolve(p) for p in cfg.calib_data]):
		for k, v in trajectory_scores(cfg, wm, models, data).items():
			scores.setdefault(k, []).append(v)
	n = len(next(iter(scores.values())))
	print(f"{n} calibration trajectories, alpha = {cfg.alpha}")
	for k, v in scores.items():
		print(f"{k:22s} {conformal_quantile(v, cfg.alpha):.4g}")


if __name__ == "__main__":
	main()
