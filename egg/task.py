"""Hooks used by latentdisturbance.train for the fried-egg world model."""

import numpy as np
import torch
from gymnasium import spaces

from egg.data import EpisodeStore, load_norm_stats, make_dataset as _make_dataset
from latentdisturbance.config import resolve
from latentdisturbance.wm import load_world_model as _load


def build_spaces(cfg, stats):
	h, w = stats["image_hw"]
	obs = {f"{c}_cam": spaces.Box(0, 255, (h, w, 3), np.uint8) for c in cfg.cameras}
	obs["state"] = spaces.Box(-1.0, 1.0, (len(stats["state"]["low"]),), np.float32)
	return spaces.Dict(obs), spaces.Box(-1.0, 1.0, (len(stats["action"]["low"]),), np.float32)


def norm_stats(cfg, checkpoint=None):
	path = resolve(checkpoint or cfg.wm_checkpoint)
	if cfg.norm_stats:
		return load_norm_stats(resolve(cfg.norm_stats))
	return torch.load(path, map_location="cpu", weights_only=False)["norm_stats"]


def load_world_model(cfg, checkpoint=None):
	weights = resolve(cfg.dino["weights_path"])
	assert weights.is_file(), f"DINOv3 ViT-S/16+ weights not found at {weights}; see INSTALL.md"
	cfg.dino = dict(cfg.dino, weights_path=str(weights))
	obs_space, act_space = build_spaces(cfg, norm_stats(cfg, checkpoint))
	cfg.num_actions = act_space.shape[0]
	return _load(cfg, obs_space, act_space, checkpoint)


def make_dataset(cfg):
	dirs = [resolve(d) for d in cfg.offline_traindir]
	return _make_dataset(cfg, norm_stats(cfg), dirs)


def make_value_reg_dataset(cfg):
	"""Known-safe episodes, plus the last `value_reg_lastframe_n` frames of every
	failure-free episode in the rollout dataset."""
	saved = cfg.num_workers
	cfg.num_workers = cfg.value_reg_num_workers
	tail_dirs = [resolve(d) for d in cfg.offline_traindir] if cfg.value_reg_lastframe_n else []
	data = _make_dataset(cfg, norm_stats(cfg), [resolve(d) for d in cfg.value_reg_traindir], seed=cfg.seed + 7717,
						 tail_dirs=tail_dirs, tail_frames=cfg.value_reg_lastframe_n,
						 tail_frac=cfg.value_reg_lastframe_frac if cfg.value_reg_lastframe_n else 0.0,
						 batch_length=cfg.value_reg_batch_length)
	cfg.num_workers = saved
	return data


def calibration_trajectories(cfg, dirs):
	store = EpisodeStore(dirs, [f"{c}_cam" for c in cfg.cameras], norm_stats(cfg))
	for ep in store.episodes:
		n = len(ep["action"])
		data = {k: v[None] for k, v in ep.items() if not k.endswith("_cam")}
		data.update({k: store.frames(ep, k, 0, n)[None] for k in store.cam_keys})
		yield data
